from typing import Dict, Union, Literal
import logging
import os, sys
import cv2
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor
import torch.nn.functional as F

from models.gaussians.basics import *
from datasets.base.scene_dataset import ModelType
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from utils.visualization import get_layout
from utils.geometry import transform_points
from utils.camera import get_interp_novel_trajectories
from utils.misc import export_points_to_ply, import_str
from utils.lib import pointnet2_utils as pointutils
from utils.o3d_vis import visualize_point_cloud, bcpdreg
from datasets.voxel_utils import build_voxel_grid, smpl_to_voxel
from scipy.spatial.transform import Rotation 
from utils.grid_c4 import get_smpl_template, smpl_based_completion_and_densification 
from pytorch3d.ops import knn_points, sample_farthest_points, knn_gather

# sys.path.append("./third_party/LiveHPS/smpl")
# from smpl import SMPL, SMPL_MODEL_DIR
# from third_party.LiveHPS.models import LiveHPS, gen_smpl, gen_smpl_quats

from third_party.smplx.smplx import SMPLLayer
from third_party.smplx.smplx.utils import SMPLOutput
from third_party.smplx.smplx.lbs import vertices2joints, batch_rigid_transform

from models.human_body import get_predefined_human_rest_pose, init_xyz_on_mesh, init_qso_on_mesh
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    axis_angle_to_matrix,
    axis_angle_to_quaternion
)

# import open3d.ml.torch as ml3d

logger = logging.getLogger()

DEBUG_PCD=False
if DEBUG_PCD:
    DEBUG_OUTPUT_DIR="debug"
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

NAME_TO_NODE = {
    "RigidNodes": ModelType.RigidNodes,
    "SMPLNodes": ModelType.SMPLNodes,
    "DeformableNodes": ModelType.DeformableNodes
}

class DrivingDataset(SceneDataset):
    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)
        
        # AVAILABLE DATASETS:
        #   Waymo:    5 Cameras
        #   KITTI:    2 Cameras
        #   NuScenes: 6 Cameras
        #   ArgoVerse:7 Cameras
        #   PandaSet: 6 Cameras
        #   NuPlan:   8 Cameras
        self.type = self.data_cfg.dataset
        try: # For Waymo, NuScenes, ArgoVerse, PandaSet
            self.data_path = os.path.join(
                self.data_cfg.data_root,
                f"{int(self.scene_idx):03d}"
            )
        except: # For KITTI, NuPlan
            self.data_path = os.path.join(self.data_cfg.data_root, self.scene_idx)
            
        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"
        if os.path.exists(os.path.join(self.data_path, "ego_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "ego_pose")))
        elif os.path.exists(os.path.join(self.data_path, "lidar_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "lidar_pose")))
        else:
            raise ValueError("Unable to determine the total number of frames. Neither 'ego_pose' nor 'lidar_pose' directories found.")

        # ---- find the number of synchronized frames ---- #
        if self.data_cfg.end_timestep == -1:
            end_timestep = total_frames - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        # to make sure the last timestep is included
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep
        
        # ---- create layout for visualization ---- #
        self.layout = get_layout(self.type)

        # ---- create data source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()
        assert self.pixel_source is not None and self.lidar_source is not None, \
            "Must have both pixel source and lidar source"
        self.project_lidar_pts_on_images(
            delete_out_of_view_points=True
        )
        self.aabb = self.get_aabb()

        # ---- define train and test indices ---- #
        # note that the timestamps of the pixel source and the lidar source are the same in waymo dataset
        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()

        # ---- create split wrappers ---- #
        image_sets = self.build_split_wrapper()
        self.train_image_set, self.test_image_set, self.full_image_set = image_sets

        # self.livehps_smpl = SMPL(SMPL_MODEL_DIR, create_transl=False).to(self.device)
        # self.livehps_model = LiveHPS()
        # # if torch.cuda.device_count() > 1:
        # #     self.livehps_model = torch.nn.DataParallel(self.livehps_model)
        # #     print(torch.cuda.device_count())
        # self.livehps_model.to(self.device)
        
        # debug use
        # self.seg_dynamic_instances_in_lidar_frame(-1, frame_idx=0)
        # self.get_init_objects()
        
    @property
    def instance_num(self):
        return len(self.pixel_source.instances_pose[0])
    
    @property
    def frame_num(self):
        return self.pixel_source.num_frames
    
    def get_instance_infos(self):
        return (
            self.pixel_source.instances_pose.clone(),
            self.pixel_source.instances_size.clone(),
            self.pixel_source.instances_model_types.clone(),
            self.pixel_source.per_frame_instance_mask.clone()
        )
    
    def safe_cat(self, tensor_list, dim=0, device="cuda"):
        tensor_list = [t for t in tensor_list if t.numel() > 0]
        if len(tensor_list) == 0:
            return torch.empty(0, 3, device=device)
        return torch.cat(tensor_list, dim=dim)
    
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        # return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return F.normalize(x, dim=-1, eps=1e-8)

    def build_split_wrapper(self):
        train_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # train_indices are img indices, so the length is num_cams * num_train_timesteps
            split_indices=self.train_indices,
            split="train",
        )
        full_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # cover all the images
            split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
            split="full",
        )
        test_image_set = None
        if len(self.test_indices) > 0:
            test_image_set = SplitWrapper(
                datasource=self.pixel_source,
                # test_indices are img indices, so the length is num_cams * num_test_timesteps
                split_indices=self.test_indices,
                split="test",
            )
        image_sets = (train_image_set, test_image_set, full_image_set)
        return image_sets

    def build_data_source(self):
        """
        Create the data source for the dataset.
        """
        # ---- create pixel source ---- #
        pixel_source = import_str(self.data_cfg.pixel_source.type)(
            self.data_cfg.dataset,
            self.data_cfg.pixel_source,
            self.data_path,
            self.start_timestep,
            self.end_timestep,
            device=self.device,
        )
        pixel_source.to(self.device)
        
        # ---- create lidar source ---- #
        lidar_source = None
        if self.data_cfg.lidar_source.load_lidar:
            lidar_source = import_str(self.data_cfg.lidar_source.type)(
                self.data_cfg.lidar_source,
                self.data_path,
                self.start_timestep,
                self.end_timestep,
                device=self.device,
            )
            lidar_source.to(self.device)
            assert (pixel_source._unique_normalized_timestamps - lidar_source._unique_normalized_timestamps).abs().sum().item() == 0., \
                "The timestamps of the pixel source and the lidar source are not synchronized"
        return pixel_source, lidar_source
    
    def get_lidar_samples(
        self, 
        num_samples: float = None,
        downsample_factor: float = None,
        return_color=False,
        return_flow=False,
        return_normalized_time=False,
        device: torch.device = torch.device("cpu")
        ) -> Tensor:
        assert self.lidar_source is not None, "Must have lidar source if you want to get init pcd"
        assert (num_samples is None) != (downsample_factor is None), \
            "Must provide either num_samples or downsample_factor, but not both"
        if downsample_factor is not None:
            num_samples = int(len(self.lidar_source.pts_xyz) / downsample_factor)
        if num_samples > len(self.lidar_source.pts_xyz):
            logger.warning(f"num_samples {num_samples} is larger than the number of points {len(self.lidar_source.pts_xyz)}")
            num_samples = len(self.lidar_source.pts_xyz)
        
        # randomly sample points
        if self.data_cfg.lidar_source.use_fps_samplling:
            xyz_t = self.lidar_source.pts_xyz.unsqueeze(0)
            self.sampled_idx = pointutils.furthest_point_sample(xyz_t, num_samples)  # [B, N]
            sampled_pts = pointutils.gather_operation(xyz_t.permute(0, 2, 1).contiguous(), self.sampled_idx)  # [B, C, N]
            sampled_pts = sampled_pts.squeeze(0).permute(1, 0).contiguous().to(device)  # [N, C]
            self.sampled_idx = self.sampled_idx.squeeze(0).contiguous().to(device)  # [N, C]
        else:
            self.sampled_idx = torch.randperm(len(self.lidar_source.pts_xyz))[:num_samples]
            sampled_pts = self.lidar_source.pts_xyz[self.sampled_idx].to(device)

        
        # get color if needed
        sampled_color = None
        if return_color:
            sampled_color = self.lidar_source.colors[self.sampled_idx].to(device)
        
        sampled_time = None
        if return_normalized_time:
            sampled_time = self.lidar_source._normalized_time[self.sampled_idx].to(device)
            sampled_time = sampled_time[..., None]
        

        if return_flow:
            sampled_flow = None
            sampled_flow = self.lidar_source.flows[self.sampled_idx].to(device)
            return sampled_pts, sampled_color, sampled_flow, sampled_time
        else:
            return sampled_pts, sampled_color, sampled_time
    
    def get_pcs(self, device: torch.device = torch.device("cpu")):
        """
        Get the source point clouds from the lidar source.
        Returns:
            src_pts: Tensor, [num_points, 3]
                The source point clouds.
            src_colors: Tensor, [num_points, 3]
                The colors of the source point clouds.
            src_flows: Tensor, [num_points, 3]
                The flows of the source point clouds.
        """
        assert self.lidar_source is not None, "Must have lidar source if you want to get init pcd"
        sampled_pts = self.lidar_source.pts_xyz[self.sampled_idx].to(device)
        
        # get color if needed
        sampled_color = None
        sampled_color = self.lidar_source.colors[self.sampled_idx].to(device)
        
        sampled_time = None
      
        sampled_time = self.lidar_source._normalized_time[self.sampled_idx].to(device)
        sampled_time = sampled_time[..., None]
        
        sampled_flow = None
        sampled_flow = self.lidar_source.flows[self.sampled_idx].to(device)

        return sampled_pts, sampled_color, sampled_flow, sampled_time
        
        
    def seg_dynamic_instances_in_lidar_frame(
        self,
        instance_ids: Union[int, list],
        frame_idx: int
        ):
        if isinstance(instance_ids, int):
            instance_num = len(self.pixel_source.instances_pose[frame_idx])
            assert instance_ids < instance_num, f"instance_id {instance_ids} is larger than the number of instances {instance_num}"
            if instance_ids == -1:
                instance_ids = list(range(instance_num))
            else:
                instance_ids = [instance_ids]
        elif isinstance(instance_ids, list):
            instance_ids = instance_ids
        
        # get the lidar points
        lidar_dict = self.lidar_source.get_lidar_rays(frame_idx)
        lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
        valid_mask = torch.zeros_like(lidar_pts[:, 0]).bool()
        for instance_id in instance_ids:
            is_valid_instance = self.pixel_source.per_frame_instance_mask[frame_idx, instance_id]
            if not is_valid_instance:
                continue
            # get the pose of the instance at the given frame
            o2w = self.pixel_source.instances_pose[frame_idx, instance_id]
            o_size = self.pixel_source.instances_size[instance_id]
            
            # transform the lidar points to the instance's coordinate system
            # instance_pose [4, 4], pts [N, 3]
            w2o = torch.inverse(o2w)
            o_pts = transform_points(lidar_pts, w2o)
            # get the mask of the points that are inside the instance's bounding box
            mask = (
                (o_pts[:, 0] > -o_size[0] / 2)
                & (o_pts[:, 0] < o_size[0] / 2)
                & (o_pts[:, 1] > -o_size[1] / 2)
                & (o_pts[:, 1] < o_size[1] / 2)
                & (o_pts[:, 2] > -o_size[2] / 2)
                & (o_pts[:, 2] < o_size[2] / 2)
            )
            valid_mask = valid_mask | mask

        valid_points = lidar_pts[valid_mask]
        valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][valid_mask]
        
        if DEBUG_PCD:
            export_points_to_ply(
                valid_points,
                valid_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "vehicle_lidar_pts.ply")
            )
            export_points_to_ply(
                lidar_pts,
                self.lidar_source.colors[lidar_dict["lidar_mask"]],
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_pts.ply")
            )
        

    def get_on_mesh_init_geo_values(self, v, f, opacity_init_logit, on_mesh_subdivide = 0, scale_init_factor = 1.0, thickness_init_factor = 0.5, max_scale = 1.0, min_scale = 0.0, s_inv_act = torch.logit):
        x_all, q_all, s_all, o_all = [], [], [], []
        for i in range(len(v)):
            x, mesh = init_xyz_on_mesh(v[i], f, on_mesh_subdivide)
            q, s, o = init_qso_on_mesh(
                mesh,
                scale_init_factor,
                thickness_init_factor,
                max_scale,
                min_scale,
                s_inv_act,
                opacity_init_logit,
            )
            
            x_all.append(x)
            q_all.append(q)
            s_all.append(s)
            o_all.append(o)
        
        x_all = torch.cat(x_all, dim=0)
        q_all = torch.cat(q_all, dim=0)
        s_all = torch.cat(s_all, dim=0)
        o_all = torch.cat(o_all, dim=0)
        return x_all, q_all, s_all, o_all

    def get_smpl_template(self, fi, num_human, init_beta, instances_quats, instances_trans, cano_pose_type="a_pose", smpl_points_num=6890, smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl", use_canical=True):
        assert num_human == init_beta.shape[0], "num_human should be the same as the number of beta"
        
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        cano_pose_type = cano_pose_type

        can_pose = get_predefined_human_rest_pose(cano_pose_type)
        # if use_canical:
        #     instances_quats[1:,] = instances_quats[0,:]
        can_pose = axis_angle_to_matrix(torch.cat([torch.zeros(1, 3), can_pose], 0))
        
        
        _template_layer = SMPLLayer(model_path=smpl_model_path)
        init_smpl_output, _ = _template_layer(
            betas=init_beta,
            body_pose=can_pose[None, 1:].repeat(num_human, 1, 1, 1),
            global_orient=can_pose[None, 0].repeat(num_human, 1, 1, 1),
            return_full_pose=True,
        )
        J_canonical, A0 = init_smpl_output.J, init_smpl_output.A
        A0_inv = torch.inverse(A0)

        v_init = init_smpl_output.vertices  # [B, 6890, 3]
        W_init = _template_layer.lbs_weights  # [6890, 24]
        # instances_quats[1:,1:] = torch.zeros_like(instances_quats[1:,1:])
        faces = _template_layer.faces_tensor
        opacity_init_value = 0.99
        x, q, s, o = self.get_on_mesh_init_geo_values(v_init, faces, opacity_init_logit=torch.logit(torch.tensor(opacity_init_value)))

     
        masked_theta = instances_quats.unsqueeze(1)  # [23,24,4]->[11,24,4]
        masked_theta = masked_theta / masked_theta.norm(dim=-1, keepdim=True)

        assert (
            masked_theta.ndim == 3 and masked_theta.shape[-1] == 4
        ), "pose should have shape Bx24x3, in axis-angle format"
        # nB = len(masked_theta)
        _, A = batch_rigid_transform(quaternion_to_matrix(masked_theta), J_canonical, _template_layer.parents)
        A = torch.einsum("bnij, bnjk->bnik", A, A0_inv)  # B,24,4,4
        W = W_init.unsqueeze(0)
   
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)
        R = T[:, :, :3, :3] # [N, 3, 3]
        t = T[:, :, :3, 3]  # [N, 3]
        
        if use_canical:
            rot_per_pts = quat_to_rotmat(masked_theta[0] / masked_theta[0].norm(dim=-1, keepdim=True)) # (num_instances, 3, 3)
            deformed_means = x @ rot_per_pts[0].T

            pelvis_coord = J_canonical[:, 0, :]   
            neck_coord   = J_canonical[:, 12, :]  
            v = neck_coord - pelvis_coord
            v = v @ rot_per_pts[0].T
            v = v.detach().cpu().numpy()
            v = v / np.linalg.norm(v)

            z_axis = np.array([0, 0, 1])
            axis = np.cross(v, z_axis)
            if np.linalg.norm(axis) < 1e-8:
                rot_mat = np.eye(3)
            else:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(v, z_axis), -1.0, 1.0))
                rot = Rotation.from_rotvec(axis * angle)
                rot_mat = rot.as_matrix()

          
            deformed_means = deformed_means @ rot_mat[0].T
            deformed_means = deformed_means.unsqueeze(0).cpu().to(x.dtype)
        else:
            reshaped_means = x.reshape(num_human, smpl_points_num, 3)
            np_reshaped_means = reshaped_means.detach().cpu().numpy()
            can_rot = R[0,0,:,:]
            np_reshaped_means = np_reshaped_means[0,:,:] @ can_rot.detach().cpu().numpy()
            # np.savetxt("da_can.txt", np_reshaped_means, fmt='%.5f') 
            deformed_means = torch.einsum("bnij,bnj->bni", R, reshaped_means) + t # [N, 6890, 3]
            # np_deformed_means = deformed_means.detach().cpu().numpy()
            # fn_name = "da_deform" + str(fi) +".txt"
            # np.savetxt(fn_name, np_deformed_means[0,:,:], fmt='%.5f') 
    
        bbox_min = deformed_means.min(dim=1)[0]
        bbox_max = deformed_means.max(dim=1)[0]
        local_shift = (bbox_min + bbox_max) / 2
        instances_trans = instances_trans - local_shift
        # visualize_point_cloud([deformed_means[0]], [[0,0,1]])
        deformed_means_o2w = deformed_means[0] + instances_trans
        return deformed_means_o2w, faces
    
    def add_gaussian_noise_to_points(self, points: torch.Tensor, num_new_points: int, std_dev: float = 0.1) -> torch.Tensor:
        """
        Args:
            points (torch.Tensor):  (N, 3)
            num_new_points (int): 
            std_dev (float): 

        Returns:
            torch.Tensor:  (num_new_points, 3)
        """
        if points.numel() == 0:
            return None
        device = points.device
        
        random_indices = torch.randint(0, points.shape[0], (num_new_points,), device=device)
        base_points = points[random_indices]

        noise = torch.randn_like(base_points) * std_dev

        new_points = base_points + noise

        dists,idx = pointutils.three_nn(new_points.unsqueeze(0).contiguous(), points.unsqueeze(0).contiguous())
        dists[dists < 1e-6] = 1e-6
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1,keepdim = True)   # [B,N,3]
        N = new_points.shape[0]
        # M = points.shape[0]
        
        nn_obj = 1.0 * pointutils.grouping_operation(points.unsqueeze(0).transpose(2,1).contiguous(), idx) #[B,C,N,3]
        interpolated_obj = torch.sum(nn_obj * weight.view(1, 1, N, 3), dim = -1) # [B,C,N,3]
        interpolated_obj = interpolated_obj.squeeze(0).transpose(1,0)
        
        return interpolated_obj#.detach().cpu().numpy()
    
    def vis_3d_temp_project_to_img(self, fi, o_type, valid_pts, lidar_pts, first_frame_betas, smpl_quats, smpl_trans, mask):
        # normed_time = self.pixel_source.normalized_time[fi]
        # # get lidar depth on image plane
        # fi = self.lidar_source.find_closest_timestep(normed_time)
        if o_type == ModelType.SMPLNodes and valid_pts.numel() > 0:
            # cam_id = 0
            for cam_id, cam in enumerate(self.pixel_source.camera_data.values()):
                if cam.undistort:
                    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                                cam.intrinsics[fi].cpu().numpy(),
                                cam.distortions[fi].cpu().numpy(),
                                (cam.WIDTH, cam.HEIGHT),
                                alpha=1,
                            )
                    intrinsic_4x4 = torch.nn.functional.pad(
                            torch.from_numpy(new_camera_matrix), (0, 1, 0, 1)
                        ).to(self.device)
                else:
                    intrinsic_4x4 = torch.nn.functional.pad(
                        cam.intrinsics[fi], (0, 1, 0, 1)
                    )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[fi].inverse()
                lidar_points2 = (
                    lidar2img[:3, :3] @ lidar_pts.clone().T + lidar2img[:3, 3:4]
                ).T # (num_pts, 3)
                deformed_can_means, faces = self.get_smpl_template(fi, num_human=1, init_beta=first_frame_betas.unsqueeze(0), instances_quats=smpl_quats[fi], instances_trans=smpl_trans[fi], smpl_points_num=6890, smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl")
                deformed_can_means = deformed_can_means.to(self.device)
                lidar_points = (
                    lidar2img[:3, :3] @ deformed_can_means.clone().T + lidar2img[:3, 3:4]
                ).T # (num_pts, 3)
                
                depth = lidar_points[:, 2]
                depth2 = lidar_points2[:, 2]
                cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6) # (num_pts, 2)
                cam_points2 = lidar_points2[:, :2] / (depth2.unsqueeze(-1) + 1e-6) # (num_pts, 2)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                ) # (num_pts, )
                valid_mask2 = (
                    (cam_points2[:, 0] >= 0)
                    & (cam_points2[:, 0] < cam.WIDTH)
                    & (cam_points2[:, 1] >= 0)
                    & (cam_points2[:, 1] < cam.HEIGHT)
                    & (depth2 > 0)
                ) # (num_pts, )
                visualize_point_cloud([lidar_pts, lidar_pts[valid_mask2], lidar_pts[mask], deformed_can_means], [[1,0,0],[0,0,1], [0,1,0], [0.5, 0.5,0]])
                depth = depth[valid_mask]
                _cam_points = cam_points[valid_mask]
                if _cam_points.numel() == 0:
                    print("Tensor is empty")
                    continue
                
                # depth_map = torch.zeros(
                #         cam.HEIGHT, cam.WIDTH, 3
                #     ).to(self.device)
                images_dir = self.data_path+'/images'
                ori_image = cv2.imread(os.path.join(images_dir, f"{str(fi).zfill(3)}_{cam_id}.jpg"))
                resized_image = cv2.resize(ori_image, (cam.WIDTH, cam.HEIGHT)) 
                depth_map = torch.from_numpy(resized_image).to(self.device, valid_pts.dtype) / 255.0
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long(), 0
                ] = depth.squeeze(-1)
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long(), 1
                ] = 0
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long(), 2
                ] = 0
                mask2 = mask & valid_mask2
                _cam_points22 = cam_points2[mask2]
                depth22 = depth2[mask2]
                # _cam_points2 = cam_points2[valid_mask2]
                # depth2 = depth2[valid_mask2]
                # if _cam_points2.numel() == 0:
                #     print("Tensor is empty")
                #     continue
                # depth_map[
                #     _cam_points2[:, 1].long(), _cam_points2[:, 0].long(), 2
                # ] = depth2.squeeze(-1)
                
                depth_map[
                    _cam_points22[:, 1].long(), _cam_points22[:, 0].long(), 1
                ] = depth22.squeeze(-1)
                depth_map[
                    _cam_points22[:, 1].long(), _cam_points22[:, 0].long(), 0
                ] = 0
                depth_map[
                    _cam_points22[:, 1].long(), _cam_points22[:, 0].long(), 2
                ] = 0

                img = depth_map.squeeze().cpu().numpy() * 255
                img = img.astype(np.uint8)
                # fn_name = str(fi) +  "_gray_pure_image_" + str(cam_id) + ".png"
                fn_name = "gray_pure_image_" + str(cam_id) + ".png"
                cv2.imwrite(fn_name, img)
                # cam_id += 1
                print(f"cam_id:{cam_id}")
            
    def farthest_point_sample(self, xyz, npoint):
        ndataset = xyz.shape[0]
        if ndataset<npoint:
            repeat_n = int(npoint/ndataset)
            xyz = np.tile(xyz,(repeat_n,1))
            xyz = np.append(xyz,xyz[:npoint%ndataset],axis=0)
            return xyz
        centroids = np.zeros(npoint)
        distance = np.ones(ndataset) * 1e10
        farthest =  np.random.randint(0, ndataset)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[int(farthest)]
            dist = np.sum((xyz - centroid) ** 2, 1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        return xyz[np.int32(centroids)]

    def get_init_objects(
        self,
        cur_node_type: Literal["RigidNodes", "DeformableNodes", "CPDDeformableNodes"],
        instance_max_pts: int = 5000,
        only_moving: bool = True,
        traj_length_thres: float = 0.5,
        exclude_smpl: bool = False,
        ):
        """
        return:
            instances_dict: Dict[int, Dict[str, Tensor]]
                keys: instance_id
                values: Dict[str, Tensor]
                    keys: "pts", "colors", "num_pts", "flows"(Optional)
                    values: Tensor

        NOTE: pts are in object coordinate system
        """
        if self.type == "KITTI":
            traj_length_thres = 5.0
            logger.info(f"For KITTI dataset, the trajectory length threshold is set \
                to {traj_length_thres} to filter out noisy short trajectories of static objects")
            

        instance_dict = {}
        target_dir = self.data_cfg.data_root + '/' + str(self.data_cfg.scene_idx).zfill(3) + "/flow/points_id/"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        
        fi_valid_insId_list_full = []
        fi_valid_ins_o2w_list_full = []
        fi_valid_pts_list_full = []
        for fi in range(self.frame_num):
            lidar_dict = self.lidar_source.get_lidar_rays(fi)
            lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
            # lidar_pts2 = lidar_pts.clone()
            fi_valid_insId_list = []
            fi_valid_ins_o2w_list = []
            fi_valid_pts_list = []
            # fi_list = []
            for ins_id in range(self.instance_num):
                instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                o_type = self.pixel_source.instances_model_types[ins_id].item()
                
                if not instance_active:
                    continue
                
                if cur_node_type == "DeformableNodes":
                    if not (
                        o_type == ModelType.DeformableNodes or 
                        o_type == ModelType.SMPLNodes
                    ):
                        continue
                elif cur_node_type == "RigidNodes":
                    if not o_type == ModelType.RigidNodes:
                        continue

                elif cur_node_type == "CPDDeformableNodes":
                    if not (
                        # o_type == ModelType.RigidNodes or
                        o_type == ModelType.DeformableNodes or 
                        o_type == ModelType.SMPLNodes 
                    ):
                        continue
                
                if exclude_smpl:
                    # objects with smpl pose will be modeled by SMPLNodes
                    assert cur_node_type == "DeformableNodes" or cur_node_type=='CPDDeformableNodes', \
                        "Only exclude SMPL for DeformableNodes"
                    true_id = self.pixel_source.instances_true_id[ins_id].item()
                    if true_id in self.pixel_source.smpl_human_all.keys():
                        continue

                true_id = self.pixel_source.instances_true_id[ins_id].item()
                # valid_smpl = False
                if true_id in self.pixel_source.smpl_human_all.keys() and self.pixel_source.smpl_human_all[true_id]["frame_valid"].sum() > 0:
                    # valid_smpl = True
                    smpl_trans = self.pixel_source.smpl_human_all[true_id]["smpl_trans"]
                    frame_info = self.pixel_source.smpl_human_all[true_id]["frame_valid"]
                    smpl_quats = self.pixel_source.smpl_human_all[true_id]["smpl_quats"]
                    smpl_betas = self.pixel_source.smpl_human_all[true_id]["smpl_betas"]
                    # size = self.pixel_source.instances_size[ins_id]
                    # # NOTE: set the first frame's betas as the betas of the instance
                    first_frame_betas = smpl_betas[fi] 
                else:
                    smpl_trans = None
                    frame_info = None
                    smpl_quats = None
                    smpl_betas = None
                    first_frame_betas = None

                if ins_id not in instance_dict:
                    instance_dict[ins_id] = {
                        "node_type": cur_node_type,
                        "pts": [],
                        "ref_pts": [],
                        "pt_ids": None,
                        "ref_fi": [],
                        "ref_colors": [],
                        # "ref_o_type": [],
                        "colors": [],
                        "o_type": [],
                        "o2w": [],
                        "fi": [],
                        "o2w_dict":{},
                        "can_smpl_quats": None,           # [1, 24, 4]
                        "can_smpl_trans": None,           # [1, 3]
                        "smpl_pt_masks": None,
                        "keypt_masks": None,
                        "faces": None,
                        "cpddeformable_mask": True if o_type == ModelType.SMPLNodes and smpl_quats is None else False,
                        "smpl_mask": True if o_type == ModelType.SMPLNodes and smpl_quats is not None else False,
                        # # "fi_pts": {}
                        "smpl_quats": smpl_quats if o_type == ModelType.SMPLNodes and smpl_quats is not None else None,           # [frame_num, 24, 4]
                        "smpl_trans": smpl_trans if o_type == ModelType.SMPLNodes and smpl_trans is not None else None,           # [frame_num, 3]
                        "smpl_betas": first_frame_betas if o_type == ModelType.SMPLNodes and first_frame_betas is not None else None,    # [10]
                    }
                if o_type == ModelType.SMPLNodes and smpl_quats is None:
                    instance_dict[ins_id]["cpddeformable_mask"] = True
                    instance_dict[ins_id]["smpl_mask"] = True
                else:
                    instance_dict[ins_id]["cpddeformable_mask"] = False
                    instance_dict[ins_id]["smpl_mask"] = False
                # get the pose of the instance at the given frame
                o2w = self.pixel_source.instances_pose[fi, ins_id]
                o_size = self.pixel_source.instances_size[ins_id]
                # convert the lidar points to the instance's coordinate system
                w2o = torch.inverse(o2w)
                o_pts = transform_points(lidar_pts, w2o)
                instance_dict[ins_id]["o2w"].append(o2w)
                instance_dict[ins_id]["fi"].append(fi)
                # get the mask of the points that are inside the instance's bounding box
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                valid_pts = o_pts[mask]
                valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][mask]
                instance_dict[ins_id]["pts"].append(valid_pts)
                if len(instance_dict[ins_id]["ref_pts"]) > 0:
                    if instance_dict[ins_id]["ref_pts"][0].shape[0] < valid_pts.shape[0]:
                        instance_dict[ins_id]["ref_pts"][0] = valid_pts
                        instance_dict[ins_id]["ref_colors"][0] = valid_colors
                        instance_dict[ins_id]["ref_fi"][0] = fi
                        # if o_type == ModelType.RigidNodes:
                        #     instance_dict[ins_id]["ref_o_type"][0] = torch.full((valid_pts.shape[0], 1), 1, dtype=torch.long).to(self.device)
                        # if o_type == ModelType.DeformableNodes:
                        #     instance_dict[ins_id]["ref_o_type"][0] = torch.full((valid_pts.shape[0], 1), 4, dtype=torch.long).to(self.device)
                        # if o_type == ModelType.SMPLNodes:
                        #     instance_dict[ins_id]["ref_o_type"][0] = torch.full((valid_pts.shape[0], 1), 2, dtype=torch.long).to(self.device)
                        # instance_dict[ins_id]["smpl_betas"] = first_frame_betas if o_type == ModelType.SMPLNodes else None
                else:
                    instance_dict[ins_id]["ref_pts"].append(valid_pts)
                    instance_dict[ins_id]["ref_fi"].append(fi)
                    instance_dict[ins_id]["ref_colors"].append(valid_colors)
                    # if o_type == ModelType.RigidNodes:
                    #     instance_dict[ins_id]["ref_o_type"].append(torch.full((valid_pts.shape[0], 1), 1, dtype=torch.long).to(self.device))
                    # if o_type == ModelType.DeformableNodes:
                    #     instance_dict[ins_id]["ref_o_type"].append(torch.full((valid_pts.shape[0], 1), 4, dtype=torch.long).to(self.device))
                    # if o_type == ModelType.SMPLNodes:
                    #     instance_dict[ins_id]["ref_o_type"].append(torch.full((valid_pts.shape[0], 1), 2, dtype=torch.long).to(self.device))
                    # instance_dict[ins_id]["smpl_betas"] = first_frame_betas if o_type == ModelType.SMPLNodes else None

                fi_valid_pts_list.append(transform_points(valid_pts, o2w))
                instance_dict[ins_id]["colors"].append(valid_colors)
                # instance_dict[ins_id]["flows"].append(valid_flows)
                # instance_dict[ins_id]["o_type"].append(valid_flow_classes)
                if o_type == ModelType.RigidNodes:
                    instance_dict[ins_id]["o_type"].append(torch.tensor([1], dtype=torch.int8).to(self.device))
                if o_type == ModelType.DeformableNodes:#CPDDeformableNodes=4
                    instance_dict[ins_id]["o_type"].append(torch.tensor([4], dtype=torch.int8).to(self.device))
                if o_type == ModelType.SMPLNodes:
                    instance_dict[ins_id]["o_type"].append(torch.tensor([2], dtype=torch.int8).to(self.device))

                fi_valid_insId_list.append(torch.ones((valid_pts.shape[0]), dtype=torch.int8) * ins_id)
                fi_valid_ins_o2w_list.append(w2o)
            fi_valid_insId_list_full.append(torch.cat(fi_valid_insId_list, dim=0))
            fi_valid_pts_list_full.append(torch.cat(fi_valid_pts_list, dim=0))
            fi_valid_ins_o2w_list_full.append(torch.cat(fi_valid_ins_o2w_list, dim=0))
            
        # nonsmpl_list = []
        logger.info(f"Aggregating lidar points across {self.frame_num} frames")
        P = torch.tensor([[0.,1.,0.],
                            [0.,0.,1.],
                            [1.,0.,0.]]).to(self.device)
        for ins_id in instance_dict:
            instance_max_pts = 7190
            fi_list = instance_dict[ins_id]["fi"]
            o2w_list = instance_dict[ins_id]["o2w"]
            instance_dict[ins_id]["o2w_dict"] = dict(zip(fi_list, o2w_list))
            fi = instance_dict[ins_id]["ref_fi"][0]
            ref_lidar_pts = instance_dict[ins_id]["ref_pts"][0]
            instance_dict[ins_id].pop("fi")
            instance_dict[ins_id].pop("o2w")
            
            if cur_node_type == "CPDDeformableNodes" and not(exclude_smpl):
                init_beta = instance_dict[ins_id]["smpl_betas"]
                instances_quats = instance_dict[ins_id]["smpl_quats"]
                instances_trans = instance_dict[ins_id]["smpl_trans"]
          
                if init_beta is not None and instances_quats is not None:# and ref_lidar_pts.shape[0] > 70
                    o2w = instance_dict[ins_id]["o2w_dict"][fi]
                    w2o = torch.inverse(o2w)
                    deformed_can_means, _, deformed_can_means_faces, _, _ = get_smpl_template(num_human=1, init_beta=init_beta.unsqueeze(0), instances_quats=instances_quats[fi], instances_trans=instances_trans[fi], smpl_points_num=6890, smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl", use_canical=False, use_correction=False)
                    instance_dict[ins_id]["can_smpl_quats"] = instances_quats[fi]
                    instance_dict[ins_id]["can_smpl_trans"] = instances_trans[fi]
                    deformed_can_means       = deformed_can_means.cuda()
                    deformed_can_means_faces = deformed_can_means_faces.cuda()
                    
                    hinge = False
                    if ref_lidar_pts.shape[0] > 70:
                        smpl_verts_w2o = transform_points(deformed_can_means[0], w2o)
                        smpl_verts_w2o = smpl_verts_w2o.unsqueeze(0).contiguous()
                        if ins_id == 61 and self.scene_idx==23:
                            _, source_cuda, _ = bcpdreg(smpl_verts_w2o[0].detach().cpu().numpy(), ref_lidar_pts.detach().cpu().numpy(), normalization_type='n', quiet_mode=True, show_fov=False)
                            hinge = True
                        else:
                            _, source_cuda, _ = bcpdreg(smpl_verts_w2o[0].detach().cpu().numpy(), ref_lidar_pts.detach().cpu().numpy(), normalization_type='e', quiet_mode=True, show_fov=False)
                        # joints = torch.einsum('jv,bv3->bj3', J_regressor, verts)  # [B,24,3]
                        source_cuda = torch.from_numpy(source_cuda).to(dtype=torch.float32, device=self.device)
                        dense_pc, smpl_mask, hinge_pt_ids = smpl_based_completion_and_densification(source_cuda, deformed_can_means_faces, ref_lidar_pts, obj_flow=None, n_samples=7190, k=3, threshold=0.05, hinge=hinge, device=self.device, fusion_mode=True)
                        instance_dict[ins_id]["smpl_mask"] = True
                    else:
                        if torch.linalg.norm(instances_trans[fi]) < 1e-5:
                            rot_cur_frame = quat_to_rotmat(self.quat_act(instances_quats[fi])).squeeze(1)   
                            rot_cur_frame = rot_cur_frame[0].cuda() @ P  # (num_instances, 3, 3)
                            o2w = torch.zeros((4, 4), device=rot_cur_frame.device, dtype=rot_cur_frame.dtype).to(self.device)
                            o2w[:3, :3] = rot_cur_frame
                            o2w[:3, 3]  = instances_trans[fi]
                            o2w[3, 3]   = 1.0
                            w2o = torch.inverse(o2w)
                            # source_cuda = deformed_can_means[0]
                        source_cuda = transform_points(deformed_can_means[0], w2o)                        
                        # smpl_verts_w2o = smpl_verts_w2o.unsqueeze(0).contiguous()
                        # source_cuda = deformed_can_means[0]#deformed_can_means[0]
                        dense_pc, smpl_mask, hinge_pt_ids = smpl_based_completion_and_densification(source_cuda, deformed_can_means_faces, ref_lidar_pts, obj_flow=None, n_samples=7190, k=3, threshold=0.05, hinge=hinge, device=self.device, fusion_mode=False)
                        instance_dict[ins_id]["smpl_mask"] = False

                    instance_dict[ins_id]["cpddeformable_mask"] = False
                    instance_dict[ins_id]["pts"] = dense_pc
                    instance_dict[ins_id]["faces"] = deformed_can_means_faces
                    valid_colors = instance_dict[ins_id]["ref_colors"][0]
                    instance_dict[ins_id]["smpl_pt_masks"] =  smpl_mask
                    instance_dict[ins_id]["pt_ids"] = hinge_pt_ids

                    # instance_dict[ins_id]["keypt_masks"] =  keypt_masks
                    # instance_dict[ins_id]["o_type"] = torch.ones((dense_pc.shape[0], 1), dtype=torch.long).to(self.device) * 2 # SMPLNodes
                    if valid_colors.shape[0] != dense_pc.shape[0]:
                        if ref_lidar_pts.shape[0] > 70:
                            valid_colors = self.add_gaussian_noise_to_points(valid_colors, dense_pc.shape[0], std_dev=0.01)   
                        else:
                            valid_colors = torch.rand((dense_pc.shape[0], 3), device=self.device)  
                        instance_dict[ins_id]["colors"] =  valid_colors
                        instance_dict[ins_id]["o_type"] = torch.tensor([2], dtype=torch.int8).to(self.device) # SMPLNodes
                        instance_dict[ins_id]["ref_pts"][0] = source_cuda
                    else:
                        # nonsmpl_list.append(ins_id)
                        instance_dict[ins_id]["pts"] = dense_pc
                        instance_dict[ins_id]["smpl_mask"] = False
                        instance_dict[ins_id]["colors"] = self.safe_cat(instance_dict[ins_id]["colors"], dim=0, device=self.device)
                        instance_dict[ins_id]["o_type"] = self.safe_cat(instance_dict[ins_id]["o_type"], dim=0, device=self.device)
                        # instance_dict[ins_id]["ref_pts"][0] = source_cuda
                else:
                    # nonsmpl_list.append(ins_id)
                    instance_dict[ins_id]["pts"] = self.safe_cat(instance_dict[ins_id]["pts"], dim=0, device=self.device)
                    instance_dict[ins_id]["smpl_mask"] = False
                    instance_dict[ins_id]["cpddeformable_mask"] = True
                    instance_dict[ins_id]["colors"] = self.safe_cat(instance_dict[ins_id]["colors"], dim=0, device=self.device)
                    instance_dict[ins_id]["o_type"] = self.safe_cat(instance_dict[ins_id]["o_type"], dim=0, device=self.device)
                    instance_dict[ins_id]["pt_ids"] = torch.zeros(instance_dict[ins_id]["pts"].shape[0], dtype=torch.int8, device=self.device)
            else:  
                src_pts = self.safe_cat(instance_dict[ins_id]["pts"], dim=0, device=self.device)
                instance_dict[ins_id]["pts"] = src_pts#self.safe_cat(instance_dict[ins_id]["pts"], dim=0, device=self.device)
                instance_dict[ins_id]["colors"] = self.safe_cat(instance_dict[ins_id]["colors"], dim=0, device=self.device)
                # instance_dict[ins_id]["flows"] = torch.cat(instance_dict[ins_id]["flows"], dim=0)
                instance_dict[ins_id]["o_type"] = self.safe_cat(instance_dict[ins_id]["o_type"], dim=0, device=self.device)
                instance_dict[ins_id]["pt_ids"] = torch.zeros(instance_dict[ins_id]["pts"].shape[0], dtype=torch.int8, device=self.device)
            
            assert instance_dict[ins_id]["pts"].shape[0] == instance_dict[ins_id]["colors"].shape[0], \
                f"Instance {ins_id} has {instance_dict[ins_id]['pts'].shape[0]} points but {instance_dict[ins_id]['colors'].shape[0]} colors"
            # assert instance_dict[ins_id]["o_type"].shape[0] == instance_dict[ins_id]["pts"].shape[0], \
            #     f"Instance {ins_id} has {instance_dict[ins_id]['o_type'].shape[0]} o_type but {instance_dict[ins_id]['pts'].shape[0]} points"
            instance_dict[ins_id]["num_pts"] = instance_dict[ins_id]["pts"].shape[0]
            if not instance_dict[ins_id]["smpl_mask"]:
                instance_max_pts = 5000
            else:
                instance_max_pts = 7190
            if instance_dict[ins_id]["num_pts"] > instance_max_pts:
                # randomly sample points
                if instance_dict[ins_id]["pt_ids"] is not None:
                    # instance_dict[ins_id]["pt_ids"] = instance_dict[ins_id]["pt_ids"][sampled_idx]
                    ins_hinge_pt_mask = instance_dict[ins_id]["pt_ids"] == 1
                    ins_pt_mask = instance_dict[ins_id]["pt_ids"] == 0
                    if ins_hinge_pt_mask.sum() > 0:
                        ins_hinge_pt = instance_dict[ins_id]["pts"][ins_hinge_pt_mask]
                        ins_hinge_pt_num = ins_hinge_pt.shape[0]
                        # sampled_idx = (torch.randperm(instance_dict[ins_id]["num_pts"]-ins_hinge_pt_num)[:(instance_max_pts-ins_hinge_pt_num)]).to(self.device)
                        P1 = instance_dict[ins_id]["pts"][ins_pt_mask].unsqueeze(0).contiguous()
                        fps_node_idx = pointutils.furthest_point_sample(P1, instance_max_pts-ins_hinge_pt_num) 
                        sampled_idx = fps_node_idx[0]
                        instance_dict[ins_id]["pts"] = torch.cat([instance_dict[ins_id]["pts"][ins_pt_mask][sampled_idx], ins_hinge_pt], dim=0)
                        instance_dict[ins_id]["colors"] = torch.cat([instance_dict[ins_id]["colors"][ins_pt_mask][sampled_idx], instance_dict[ins_id]["colors"][ins_hinge_pt_mask]], dim=0)
                        # instance_dict[ins_id]["o_type"] = torch.cat([instance_dict[ins_id]["o_type"][ins_pt_mask][sampled_idx], instance_dict[ins_id]["o_type"][ins_hinge_pt_mask]], dim=0)
                        if instance_dict[ins_id]["smpl_pt_masks"] is not None:
                            instance_dict[ins_id]["smpl_pt_masks"] = torch.cat([instance_dict[ins_id]["smpl_pt_masks"][ins_pt_mask][sampled_idx], instance_dict[ins_id]["smpl_pt_masks"][ins_hinge_pt_mask]], dim=0)
                        if instance_dict[ins_id]["pt_ids"] is not None:
                            instance_dict[ins_id]["pt_ids"] = torch.cat([instance_dict[ins_id]["pt_ids"][ins_pt_mask][sampled_idx], instance_dict[ins_id]["pt_ids"][ins_hinge_pt_mask]], dim=0)
                    else:
                        # sampled_idx = (torch.randperm(instance_dict[ins_id]["num_pts"])[:instance_max_pts]).to(self.device)
                        P1 = instance_dict[ins_id]["pts"].unsqueeze(0).contiguous()
                        fps_node_idx = pointutils.furthest_point_sample(P1, instance_max_pts) 
                        sampled_idx = fps_node_idx[0]
                        instance_dict[ins_id]["pts"] = instance_dict[ins_id]["pts"][sampled_idx]
                        instance_dict[ins_id]["colors"] = instance_dict[ins_id]["colors"][sampled_idx]
                        # instance_dict[ins_id]["flows"] = instance_dict[ins_id]["flows"][sampled_idx]
                        # instance_dict[ins_id]["o_type"] = instance_dict[ins_id]["o_type"][sampled_idx]
                        if instance_dict[ins_id]["smpl_pt_masks"] is not None:
                            instance_dict[ins_id]["smpl_pt_masks"] = instance_dict[ins_id]["smpl_pt_masks"][sampled_idx]
                        if instance_dict[ins_id]["pt_ids"] is not None:
                            instance_dict[ins_id]["pt_ids"] = instance_dict[ins_id]["pt_ids"][sampled_idx]
                else:
                    sampled_idx = (torch.randperm(instance_dict[ins_id]["num_pts"])[:instance_max_pts]).to(self.device)
                    instance_dict[ins_id]["pts"] = instance_dict[ins_id]["pts"][sampled_idx]
                    instance_dict[ins_id]["colors"] = instance_dict[ins_id]["colors"][sampled_idx]
                    # instance_dict[ins_id]["flows"] = instance_dict[ins_id]["flows"][sampled_idx]
                    # instance_dict[ins_id]["o_type"] = instance_dict[ins_id]["o_type"][sampled_idx]
                    if instance_dict[ins_id]["smpl_pt_masks"] is not None:
                        instance_dict[ins_id]["smpl_pt_masks"] = instance_dict[ins_id]["smpl_pt_masks"][sampled_idx]
                    if instance_dict[ins_id]["pt_ids"] is not None:
                        instance_dict[ins_id]["pt_ids"] = instance_dict[ins_id]["pt_ids"][sampled_idx]
                instance_dict[ins_id]["num_pts"] = instance_max_pts
                
            if instance_dict[ins_id]["smpl_pt_masks"] is not None:
                smpl_mask = instance_dict[ins_id]["smpl_pt_masks"]
                keypt_masks = torch.full([smpl_mask.shape[0]], False, dtype=torch.bool, device=smpl_mask.device)
                P1 = instance_dict[ins_id]["pts"].unsqueeze(0).contiguous()
                fps_node_idx = pointutils.furthest_point_sample(P1, 2500) 
                keypt_masks[fps_node_idx[0]] = True
                instance_dict[ins_id]["keypt_masks"] = keypt_masks
            logger.info(f"Instance {ins_id} has {instance_dict[ins_id]['num_pts']} lidar sample points")
        
        if only_moving:
            # consider only the instances with non-zero flows
            logger.info(f"Filtering out the instances with non-moving trajectories")
            new_instance_dict = {}
            for k, v in instance_dict.items():
                if v["num_pts"] > 0:
                    frame_info = self.pixel_source.per_frame_instance_mask[:, k]
                    instances_pose = self.pixel_source.instances_pose[:, k]
                    instances_trans = instances_pose[:, :3, 3]
                    valid_trans = instances_trans[frame_info]
                    traj_length = valid_trans[1:] - valid_trans[:-1]
                    traj_length = torch.norm(traj_length, dim=-1).sum()
                    if traj_length > traj_length_thres:
                        new_instance_dict[k] = v
                        logger.info(f"Instance {k} has {v['num_pts']} lidar sample points")
            instance_dict = new_instance_dict
    
        
        # get instance info
        for ins_id in instance_dict:
            instance_dict[ins_id]["poses"] = self.pixel_source.instances_pose[:, ins_id]
            instance_dict[ins_id]["size"] = self.pixel_source.instances_size[ins_id]
            instance_dict[ins_id]["frame_info"] = self.pixel_source.per_frame_instance_mask[:, ins_id]

        
        DEBUG_PCD = False
        DEBUG_OUTPUT_DIR = 'output_root/cpddeform/PLYs/'
        if DEBUG_PCD:
            output_dir = os.path.join(DEBUG_OUTPUT_DIR, "aggregated_instance_lidar_pts")
            os.makedirs(output_dir, exist_ok=True)
            for ins_id in instance_dict:
                export_points_to_ply(
                    instance_dict[ins_id]["pts"],
                    instance_dict[ins_id]["colors"],
                    save_path=os.path.join(output_dir, f"ID={ins_id}_{torch.unique(instance_dict[ins_id]['o_type']).item()}.ply")
                )
        return instance_dict
    
    def get_init_smpl_objects(self, only_moving: bool = False, traj_length_thres: float = 0.5):
        instance_dict = {}
        """
        instance_dict = {
            ins_id: {
                "node_type": str, 
                "pts": Tensor, [frame_num, num_pts, 3]
                "colors": Tensor, [frame_num, num_pts, 3]
                "quats": Tensor, [frame_num, 4]
                "trans": Tensor, [frame_num, 3]
                "size": Tensor, [3]
                "frame_info": Tensor, [frame_num]
        }
        """
        
        for ins_id in range(self.instance_num):
          true_id = self.pixel_source.instances_true_id[ins_id].item()
          if true_id in self.pixel_source.smpl_human_all.keys():
              if self.pixel_source.smpl_human_all[true_id]["frame_valid"].sum() == 0:
                  continue
              smpl_trans = self.pixel_source.smpl_human_all[true_id]["smpl_trans"]
              frame_info = self.pixel_source.smpl_human_all[true_id]["frame_valid"]
              if only_moving and traj_length_thres > 0:
                  traj_length = smpl_trans[frame_info][1:] - smpl_trans[frame_info][:-1]
                  traj_length = torch.norm(traj_length, dim=-1).sum()
                  if traj_length < traj_length_thres:
                      continue
              smpl_quats = self.pixel_source.smpl_human_all[true_id]["smpl_quats"]
              smpl_betas = self.pixel_source.smpl_human_all[true_id]["smpl_betas"]
              size = self.pixel_source.instances_size[ins_id]
              first_frame_betas = smpl_betas[frame_info][0]

              collected_lidar_pts = []
              collected_lidar_colors = []
              for fi in range(self.frame_num):
                  lidar_dict = self.lidar_source.get_lidar_rays(fi)
                  lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
                  instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                  if not instance_active:
                      continue

                  o2w = self.pixel_source.instances_pose[fi, ins_id]
                  o_size = self.pixel_source.instances_size[ins_id]
                  w2o = torch.inverse(o2w)
                  o_pts = transform_points(lidar_pts, w2o)
                  mask = (
                      (o_pts[:, 0] > -o_size[0] / 2)
                      & (o_pts[:, 0] < o_size[0] / 2)
                      & (o_pts[:, 1] > -o_size[1] / 2)
                      & (o_pts[:, 1] < o_size[1] / 2)
                      & (o_pts[:, 2] > -o_size[2] / 2)
                      & (o_pts[:, 2] < o_size[2] / 2)
                  )
                  valid_pts = o_pts[mask]
                  valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][mask]

                  if valid_pts.shape[0] > 0:
                      collected_lidar_pts.append(valid_pts)
                      collected_lidar_colors.append(valid_colors)

              if len(collected_lidar_pts) == 0:
                  pts_tensor = torch.empty((0, 3), device=smpl_trans.device, dtype=smpl_trans.dtype)
                  colors_tensor = torch.empty((0, 3), device=smpl_trans.device, dtype=smpl_trans.dtype)  # 根据实际颜色维度调整
              else:
                  pts_tensor = torch.cat(collected_lidar_pts, dim=0)
                  colors_tensor = torch.cat(collected_lidar_colors, dim=0)

              instance_dict[ins_id] = {
                  "node_type": "SMPLNodes",
                  "smpl_quats": smpl_quats,
                  "smpl_trans": smpl_trans,
                  "smpl_betas": first_frame_betas,
                  "size":       size,
                  "frame_info": frame_info,
                  "pts": pts_tensor,
                  "colors": colors_tensor,
              }
        return instance_dict
    
   
    def filter_pts_in_boxes(
        self,
        seed_pts: Tensor,
        valid_instances_dict: Dict[int, Dict[str, Tensor]],
        seed_colors: Tensor = None,
        seed_time: Tensor = None,
    ):
        """
        This function is used to filter out the points that are inside the bounding boxes of the instances
        """
        if DEBUG_PCD:
            os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
            export_points_to_ply(
                seed_pts,
                seed_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "original_seed_pts.ply")
            )
        valid_instance_keys = valid_instances_dict.keys()
        
        inside_mask = torch.zeros_like(seed_pts[:, 0]).bool()
        for fi in range(self.frame_num):
            for ins_id in valid_instance_keys:
                instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                if not instance_active:
                    continue
                # get the pose of the instance at the given frame
                o2w = self.pixel_source.instances_pose[fi, ins_id].to(seed_pts.device)
                o_size = self.pixel_source.instances_size[ins_id].to(seed_pts.device)
                # convert the lidar points to the instance's coordinate system
                w2o = torch.inverse(o2w)
                o_pts = transform_points(seed_pts, w2o)
                # get the mask of the points that are inside the instance's bounding box
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                inside_mask = inside_mask | mask
        
        # filter out the points that are inside the bounding boxes
        seed_pts = seed_pts[~inside_mask]
        if seed_colors is not None:
            seed_colors = seed_colors[~inside_mask]
        if seed_time is not None:
            seed_time = seed_time[~inside_mask]
        
        if DEBUG_PCD:
            export_points_to_ply(
                seed_pts,
                seed_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "filtered_seed_pts.ply")
            )
            
            for fi in range(self.frame_num):
                if fi % 10 != 0:
                    continue
                frame_save_dir = os.path.join(DEBUG_OUTPUT_DIR, f"frame_{fi}")
                os.makedirs(frame_save_dir, exist_ok=True)
                for ins_id in valid_instances_dict:
                    # print number of points
                    # print(f"Frame {fi}, Instance {ins_id} has {valid_instances_dict[ins_id]['pts'].shape[0]} points")
                    o2w = self.pixel_source.instances_pose[fi, ins_id]
                    pts_in_obj = valid_instances_dict[ins_id]["pts"]
                    # rotate the points back to the world coordinate system
                    pts_in_world = transform_points(pts_in_obj, o2w)
                    export_points_to_ply(
                        pts_in_world,
                        valid_instances_dict[ins_id]["colors"],
                        save_path=os.path.join(frame_save_dir, f"ID={ins_id}.ply")
                    )
        
        return {
            "pts": seed_pts,
            "colors": seed_colors,
            "time": seed_time
        }

    def check_pts_visibility(self, pts_xyz):
        # filter out the lidar points that are not visible from the camera
        pts_xyz = pts_xyz.to(self.device)
        valid_mask = torch.zeros_like(pts_xyz[:, 0]).bool()
        # project lidar points to the image plane
        for cam in self.pixel_source.camera_data.values():
            for frame_idx in range(len(cam)):
                intrinsic_4x4 = torch.nn.functional.pad(
                    cam.intrinsics[frame_idx], (0, 1, 0, 1)
                )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = (
                    intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                )
                projected_points = (
                    lidar2img[:3, :3] @ pts_xyz.T + lidar2img[:3, 3:4]
                ).T
                depth = projected_points[:, 2]
                cam_points = projected_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)
                current_valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                )
                valid_mask = valid_mask | current_valid_mask
        return valid_mask

    def split_train_test(self):
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                # it makes no sense to have test timesteps before the start timestep
                self.data_cfg.pixel_source.test_image_stride,
                self.num_img_timesteps,
                self.data_cfg.pixel_source.test_image_stride,
            )
        else:
            test_timesteps = []
        train_timesteps = np.array(
            [i for i in range(self.num_img_timesteps) if i not in test_timesteps]
        )
        logger.info(
            f"Train timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[train_timesteps]}"
        )
        logger.info(
            f"Test timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[test_timesteps]}"
        )

        # propagate the train and test timesteps to the train and test indices
        train_indices, test_indices = [], []
        for t in range(self.num_img_timesteps):
            if t in train_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    train_indices.append(t * self.pixel_source.num_cams + cam)
            elif t in test_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    test_indices.append(t * self.pixel_source.num_cams + cam)
        logger.info(f"Number of train indices: {len(train_indices)}")
        logger.info(f"Train indices: {train_indices}")
        logger.info(f"Number of test indices: {len(test_indices)}")
        logger.info(f"Test indices: {test_indices}")

        # Again, training and testing indices are indices into the full dataset
        # train_indices are img indices, so the length is num_cams * num_train_timesteps
        # but train_timesteps are timesteps, so the length is num_train_timesteps (len(unique_train_timestamps))
        return train_timesteps, test_timesteps, train_indices, test_indices
    
    def project_lidar_pts_on_images(self, delete_out_of_view_points=True):
        """
        Project the lidar points on the images and attribute the color of the nearest pixel to the lidar point.
        
        Args:
            delete_out_of_view_points: bool
                If True, the lidar points that are not visible from the camera will be removed.
        """
        for cam in self.pixel_source.camera_data.values():
            lidar_depth_maps = []
            lidar_flows_maps = []
            lidar_flow_classes_maps = []
            lidar_camera_maps = []
            for frame_idx in tqdm(
                range(len(cam)), 
                desc="Projecting lidar pts on images for camera {}".format(cam.cam_name),
                dynamic_ncols=True
            ):
                normed_time = self.pixel_source.normalized_time[frame_idx]
                
                # get lidar depth on image plane
                closest_lidar_idx = self.lidar_source.find_closest_timestep(normed_time)
                lidar_infos = self.lidar_source.get_lidar_rays(closest_lidar_idx)
                lidar_points_xyz = (
                    lidar_infos["lidar_origins"]
                    + lidar_infos["lidar_viewdirs"] * lidar_infos["lidar_ranges"]
                ) # 
               
                # project lidar points to the image plane
                if cam.undistort:
                    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                                cam.intrinsics[frame_idx].cpu().numpy(),
                                cam.distortions[frame_idx].cpu().numpy(),
                                (cam.WIDTH, cam.HEIGHT),
                                alpha=1,
                            )
                    intrinsic_4x4 = torch.nn.functional.pad(
                            torch.from_numpy(new_camera_matrix), (0, 1, 0, 1)
                        ).to(self.device)
                else:
                    intrinsic_4x4 = torch.nn.functional.pad(
                        cam.intrinsics[frame_idx], (0, 1, 0, 1)
                    )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                lidar_points = (
                    lidar2img[:3, :3] @ lidar_points_xyz.T + lidar2img[:3, 3:4]
                ).T # (num_pts, 3)
                
                depth = lidar_points[:, 2]
                cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6) # (num_pts, 2)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                ) # (num_pts, )
                depth = depth[valid_mask]
                _cam_points = cam_points[valid_mask]
                depth_map = torch.zeros(
                    cam.HEIGHT, cam.WIDTH
                ).to(self.device)
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ] = depth.squeeze(-1)
                lidar_depth_maps.append(depth_map)

                # flow = lidar_infos["lidar_flows"][valid_mask] if "lidar_flows" in lidar_infos else None
                flow = lidar_infos["lidar_flows"] if "lidar_flows" in lidar_infos else None
                lidar_flows_maps.append(flow.to(self.device).float())

                flow_class = lidar_infos["lidar_flow_classes"] if "lidar_flow_classes" in lidar_infos else None
                lidar_flow_classes_maps.append(flow_class.to(self.device).float())

                lidar_cam_pts = lidar_points_xyz#[valid_mask]
                lidar_camera_maps.append(lidar_cam_pts)

                # visualize_point_cloud([lidar_points_xyz[valid_mask], lidar_infos["lidar_flows"][valid_mask]], [[0,0,1],[1,0,0]])

                
                # used to filter out the lidar points that are visible from the camera
                visible_indices = torch.arange(
                    self.lidar_source.num_points, device=self.device
                )[lidar_infos["lidar_mask"]][valid_mask]
                
                self.lidar_source.visible_masks[visible_indices] = True
                
                # attribute the color of the nearest pixel to the lidar point
                points_color = cam.images[frame_idx][
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ]
                self.lidar_source.colors[visible_indices] = points_color

            cam.load_depth(
                torch.stack(lidar_depth_maps, dim=0).to(self.device).float()
            )

            cam.load_lidar_flows(lidar_flows_maps)
            cam.load_lidar_flow_classes(lidar_flow_classes_maps)
            # self.lidar_source.lidar_to_worlds
            cam.load_lidar_points(lidar_camera_maps)
            
        if delete_out_of_view_points:
            self.lidar_source.delete_invisible_pts()
            
    def get_novel_render_traj(
        self,
        traj_types: List[str] = ["front_center_interp"],
        target_frames: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Get multiple novel trajectories of the scene for rendering.
        
        Args:
            traj_types: List[str]
                A list of trajectory types to generate. Options for each type include:
                - "front_center_interp": Interpolate key frames from the front center camera
                - "s_curve": S-shaped trajectory using the front three cameras
                - "three_key_poses": Creates a trajectory using three key poses from different cameras
            target_frames: int
                The total number of frames for each novel trajectory
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are trajectory types and values
            are the generated novel trajectories, each of shape (target_frames, 4, 4)
        """
        per_cam_poses = {}
        for cam_id in self.pixel_source.camera_list:
            per_cam_poses[cam_id] = self.pixel_source.camera_data[cam_id].cam_to_worlds
        
        novel_trajs = {}
        for traj_type in traj_types:
            novel_trajs[traj_type] = get_interp_novel_trajectories(
                self.type,
                self.scene_idx,
                per_cam_poses,
                traj_type,
                target_frames
            )
        
        return novel_trajs

    def prepare_novel_view_render_data(self, traj: torch.Tensor) -> list:
            """
            Prepare all necessary elements for novel view rendering.

            Args:
                traj (torch.Tensor): Novel view trajectory, shape (N, 4, 4)

            Returns:
                list: List of dicts, each containing elements required for rendering a single frame:
                    - cam_infos: Camera information (extrinsics, intrinsics, image dimensions)
                    - image_infos: Image-related information (indices, normalized time, viewdirs, etc.)
            """
            # Call the PixelSource's method
            return self.pixel_source.prepare_novel_view_render_data(self.type, traj, self.lidar_source)