from typing import Dict, List, Tuple
import logging
import os
import torch
from torch.nn import Parameter
from argparse import ArgumentParser, Namespace

from models.modules import NormalDeformNetwork
from models.gaussians.basics import *
from models.nodes.rigid import RigidNodes
from models.gaussians.vanilla import VanillaGaussians
from concurrent.futures import ThreadPoolExecutor
from utils.o3d_vis import visualize_point_cloud
from pytorch3d.ops import knn_points, knn_gather
from utils.lib import pointnet2_utils as pointutils
from utils.geometry import transform_points, QuaternionLossDot
# from datasets.voxel_utils import build_voxel_grid, build_grid
import time
from typing import Union, Sequence, Tuple
from pytorch3d.loss import chamfer_distance
from torch.utils.checkpoint import checkpoint
from utils.grid_c4 import quaternion_multiply, get_on_mesh_init_geo_values, compute_node_rotations, compute_body_to_cloth_angle, deformation_graph_pipeline_fast
import torch.nn.functional as F
from utils.misc import export_points_to_ply, import_str

from models.human_body import phalp_colors, SMPLTemplate, get_on_mesh_init_geo_values, batch_rigid_transform, quaternion_to_matrix, get_predefined_human_rest_pose, init_xyz_on_mesh, init_qso_on_mesh
import cv2
# from models.gaussian_renderer import render
# from arguments import PipelineParams

from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    quaternion_invert
)

logger = logging.getLogger()

class CPDSMPLDeformableNodes(RigidNodes):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ins_num = 7190
        self.train_mode = True
        self.use_voxel_deformer = self.ctrl_cfg.use_voxel_deformer
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)


        self.knn_num = 15
        self.n_nodes = 2500
        self.nn_ind = None
        self.rot_nn_ind = None
        self.reshape_means = None
        self.smpl_points_num = 6890
        
        smpl_vertices = torch.zeros(1, 3, device=self.device)
        smpl_original = torch.zeros(1, 3, device=self.device)
        smpl_pt_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        keypt_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        instances_fv = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        hinge_pt_ids = torch.ones(1, dtype=torch.int8, device=self.device)
        reshape_means_ind = torch.zeros(1, dtype=torch.long, device=self.device)
        # ref_smpl_fi = torch.zeros(1, dtype=torch.int8, device=self.device)

        self.register_buffer("reshape_means_ind", reshape_means_ind) #[N,6890,3]
        self.register_buffer("smpl_vertices", smpl_vertices) #[N,6890,3], smpl_can_means
        self.register_buffer("smpl_pt_masks", smpl_pt_masks) #[N,6890,3]
        self.register_buffer("keypt_masks", keypt_masks) #[N,6890,3]
        self.register_buffer("smpl_original", smpl_original) #[N,6890,3], template smpl points
        self.register_buffer("instances_fv", instances_fv) #[N,6890,3]
        self.register_buffer("hinge_pt_ids", hinge_pt_ids) #[N,6890,3]
        # self.register_buffer("ref_smpl_fi", ref_smpl_fi) #[N,6890,3]


    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        super().create_from_pcd(instance_pts_dict=instance_pts_dict)
        # init_embedding = torch.rand(self.num_instances, self.networks_cfg.embed_dim, device=self.device)
        # self.instances_embedding = Parameter(init_embedding) # overrided the previous one
        smpl_betas, smpl_quats, smpl_can_means, smpl_can_quats, smpl_can_trans = [], [], [], [], []
        instances_quats, instances_trans, instances_size = [], [], []
        instances_fv = []
        instances_pts, instances_pose = [], []
        self.smpl_ids, smpl_pt_masks, keypt_masks, hinge_pt_ids = [], [], [], []
        ref_smpl_fi = []
        # means_temps, smpl_faces = [], []
        smpl_point_ids, point_ids, init_colors = [], [], []
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            if v["smpl_quats"] is not None: #and v["ref_pts"][0].shape[0] > 70:
                smpl_quats.append(v["smpl_quats"][:, 1:, :].unsqueeze(1))
                # instances_quats.append(v["smpl_quats"].unsqueeze(1))
                instances_quats.append(v["smpl_quats"][:, 0, :].unsqueeze(1))
                instances_trans.append(v["smpl_trans"].unsqueeze(1))
                smpl_can_quats.append(v["can_smpl_quats"].unsqueeze(0))
                smpl_can_trans.append(v["can_smpl_trans"].unsqueeze(0))
                instances_fv.append(v["frame_info"].unsqueeze(1))
                smpl_betas.append(v["smpl_betas"].unsqueeze(0))
                instances_size.append(v["size"])
                self.smpl_ids.append(id_in_model)
                ref_smpl_fi.append(v["ref_fi"][0])
                instances_pts.append(v["pts"])
                smpl_can_means.append(v["ref_pts"][0].unsqueeze(0))
                init_colors.append(v["colors"])
                instances_pose.append(v["poses"].unsqueeze(1))
                point_ids.append(torch.full((v["pts"].shape[0], 1), id_in_model, dtype=torch.long))
                smpl_point_ids.append(torch.full((self.smpl_points_num, 1), id_in_model, dtype=torch.long))
                hinge_pt_ids.append(v["pt_ids"])
                if v["smpl_pt_masks"] is not None:
                    smpl_pt_masks.append(v["smpl_pt_masks"].to(self.device))
                else:
                    smpl_pt_masks.append(torch.zeros((v["pts"].shape[0], 1), dtype=torch.bool).to(self.device))
                if v["keypt_masks"] is not None:
                    keypt_masks.append(v["keypt_masks"].to(self.device))
                else:
                    keypt_masks.append(torch.zeros(v["pts"].shape[0], dtype=torch.bool).to(self.device))
                # if v["faces"] is not None:
                # smpl_faces.append(v["faces"].unsqueeze(0).to(self.device))
        
        # self.smpl_can_quats = torch.cat(smpl_can_quats, dim=0).to(self.device)   # (num_instances, 24, 4)
        # self.smpl_can_trans = torch.cat(smpl_can_trans, dim=0).to(self.device)   # (num_instances, 3)
        smpl_quats = torch.cat(smpl_quats, dim=1).to(self.device)                # (num_frame, num_instances, 23, 4)
        instances_quats = torch.cat(instances_quats, dim=1).to(self.device)      # (num_frame, num_instances, 1，4)
        instances_trans = torch.cat(instances_trans, dim=1).to(self.device)      # (num_frame, num_instances, 3)
        instances_fv = torch.cat(instances_fv, dim=1).to(self.device)            # (num_frame, num_instances)
        smpl_betas = torch.cat(smpl_betas, dim=0).to(self.device)                # (num_instances, 10)
        instances_size = torch.stack(instances_size).to(self.device)             # (num_instances, 3)
        smpl_can_means = torch.cat(smpl_can_means, dim=0).to(self.device) # (N, 3)
        instances_pts = torch.cat(instances_pts, dim=0).to(self.device)   # (N, 3)
        init_colors = torch.cat(init_colors, dim=0).to(self.device)       # (N, 3)
        instances_pose = torch.cat(instances_pose, dim=1).to(self.device) # (num_frame, num_instances, 4, 4)
        self.ref_smpl_fi = ref_smpl_fi


        self.smpl_pt_masks = torch.cat(smpl_pt_masks, dim=0)
        self.keypt_masks = torch.cat(keypt_masks, dim=0)

        point_ids = torch.cat(point_ids, dim=0).to(self.device)                  # (self.smpl_points_num*num_instances, 1)
        smpl_point_ids = torch.cat(smpl_point_ids, dim=0).to(self.device)        # (self.smpl_points_num*num_instances, 1)
        # np.save("smpl_point_ids.npy",smpl_point_ids.cpu().numpy())
        self._means = Parameter(instances_pts, requires_grad=True)
        if not torch.isnan(self.hinge_pt_ids).any():
            self.hinge_pt_ids = torch.cat(hinge_pt_ids, dim=0) .to(self.device) #(N)
            assert self.hinge_pt_ids.shape[0] == point_ids.shape[0], "self.hinge_pt_ids.shape[0] == point_ids.shape[0]"

        means_cur, sub_means_ind = [], []
        means_ind = torch.arange(self._means.shape[0], device=self.device, dtype=torch.int32)
        for i, ins_id in enumerate(self.smpl_ids):
            pc_mask = point_ids[:,0] == ins_id
            sub_means = self._means[pc_mask]
            sub_means_ind.append(means_ind[pc_mask])
            means_cur.append(sub_means.unsqueeze(0).contiguous())
        means_cur = torch.cat(means_cur, dim=0)
        sub_means_ind = torch.cat(sub_means_ind, dim=0)
        self.reshape_means = means_cur
        self.reshape_means_ind = sub_means_ind
        _, self.nn_ind, _ = knn_points(self.reshape_means, self.reshape_means, K=self.ctrl_cfg.knn_neighbors)
        means_rot_mat, means_normal_vecs = self.pca_frame_from_knn_batched(means_cur)
        init_Qr = matrix_to_quaternion(means_rot_mat.reshape(-1,3,3))
        # self.normal_vecs = means_normal_vecs.reshape(-1,3)
        
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        # self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        assert init_Qr.shape[0] == self.num_points, "The init_Qr.shape[0] must be equal to the self.num_points"
        self._quats = Parameter(init_Qr.to(self.device))

        dim_sh = num_sh_bases(self.sh_degree)
        # pose refinement
        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        # NOTE: init_colors actually is for visualization, we use random color here
        if self.sh_degree > 0:
            # shs[:, 0, :3] = fused_means_color
            shs[:, 0, :3]  = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            # shs[:, 0, :3] = torch.logit(init_means_colors, eps=1e-10)
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

        #######################################################
        self.template = SMPLTemplate(
            smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",
            num_human=smpl_betas.shape[0],
            init_beta=smpl_betas,
            cano_pose_type="da_pose",
            use_voxel_deformer=self.ctrl_cfg.use_voxel_deformer
        )
        if self.ctrl_cfg.use_voxel_deformer:
            self.template.voxel_deformer.enable_voxel_correction()
        
        opacity_init_value = torch.tensor(self.ctrl_cfg.opacity_init_value)
        x, q, s, o = get_on_mesh_init_geo_values(
            self.template,
            opacity_init_logit=torch.logit(opacity_init_value),
        )
        if self.ball_gaussians:
            s = s.mean(-1, keepdim=True)
        x = x.to(dtype=torch.float32, device=self.device)
        s = s.to(dtype=torch.float32, device=self.device)
        q = q.to(dtype=torch.float32, device=self.device)
        o = o.to(dtype=torch.float32, device=self.device)

        # if self.ctrl_cfg.constrain_xyz_offset:
        #     self.on_mesh_x = x.clone()

        # NOTE: In the future, we will also use colors of lidars to get the initialization of colors
        self.template = self.template.to(self.device)
        for fi in range(self.num_frames):
            instance_mask = instances_fv[fi]
            if instance_mask.sum() == 0:
                continue

            theta = torch.cat(
                (instances_quats[fi].unsqueeze(1), smpl_quats[fi]), dim=1
            )
            masked_theta = theta[instance_mask]
            masked_theta = masked_theta / masked_theta.norm(dim=-1, keepdim=True)
            W, A = self.template(
                masked_theta = masked_theta, 
                instances_mask = instance_mask
            ) # W: (num_instances, self.smpl_points_num, 24), A: (num_instances, self.smpl_points_num, 4, 4)
            T = torch.einsum("bnj, bjrc -> bnrc", W, A)
            R = T[:, :, :3, :3] # [N, 3, 3]
            t = T[:, :, :3, 3]  # [N, 3]
            
            reshaped_means = x.reshape(self.num_instances, self.smpl_points_num, 3)
            deformed_means = torch.einsum(
                "bnij,bnj->bni", R, reshaped_means[instance_mask]         
            ) + t  # [N, 6890, 3]
            bbox_min = deformed_means.min(dim=1)[0]
            bbox_max = deformed_means.max(dim=1)[0]
            local_shift = (bbox_min + bbox_max) / 2
            instances_trans[fi, instance_mask] = instances_trans[fi, instance_mask] - local_shift

        self.smpl_original  = x.detach().clone()
        self.update_knn(self.smpl_original) # x
        
        self.instances_quats   = Parameter(instances_quats.unsqueeze(2)) # (num_frame, num_instances, 1, 4)
        self.instances_trans   = Parameter(instances_trans)              # (num_frame, num_instances, 3)
        self.smpl_insts_quats  = Parameter(smpl_quats)                   # (num_frame, num_instances, 23, 4)
        self.instances_size    = instances_size                          # (num_instances, 3)
        self.point_ids         = point_ids                               # (self.smpl_points_num*num_instances, 1)
        # self.smpl_point_ids  = smpl_point_ids                          # (self.smpl_points_num*num_instances, 1)
        self.register_buffer("smpl_point_ids", smpl_point_ids) #[N,6890,3]
        self.register_buffer("smpl_betas", smpl_betas) #[N,10]
        # self.register_buffer("instances_fv", instances_fv) #[N,10]
        self.instances_fv = instances_fv
        self.smpl_vertices = smpl_can_means

    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"smpl_ins_rotation"]    = [self.instances_quats]
        param_groups[self.class_prefix+"smpl_ins_quats"]       = [self.smpl_insts_quats]
        param_groups[self.class_prefix+"smpl_ins_translation"] = [self.instances_trans]
        # param_groups[self.class_prefix+"embedding"] = [self.instances_embedding]
        # param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        return param_groups

    def pca_frame_from_knn_batched(self, X, eps=1e-8):
        """
        X: [B,N,3] float
        knn_idx: [B,N,k] long  (indices into dimension N)
        Returns:
        R: [B,N,3,3] rotation matrices, columns are [t1,t2,n] (right-handed)
        n: [B,N,3] normals
        """
        assert X.dim() == 3 and X.size(-1) == 3, f"X should be [B,N,3], got {X.shape}"
        
        B, N, _ = X.shape
        _, knn_idx, neigh = knn_points(X, X, K=10, return_nn=True)
        _, N2, k = knn_idx.shape
        assert N2 == N, "knn_idx second dim must match N"
        assert knn_idx.dtype == torch.long, "knn_idx must be torch.long"

        mu = neigh.mean(dim=2, keepdim=True)             # [B,N,1,3]
        Y = neigh - mu                                  # [B,N,k,3]

        # Covariance per point: [B,N,3,3]
        C = (Y.transpose(2, 3) @ Y) / (k + eps)

        # Eigh on last two dims, supports batched: evals [B,N,3], evecs [B,N,3,3]
        evals, evecs = torch.linalg.eigh(C)

        n = evecs[..., :, 0]   # [B,N,3] smallest eigenvector
        t1 = evecs[..., :, 1]
        t2 = evecs[..., :, 2]

        R = torch.stack((t1, t2, n), dim=-1)             # [B,N,3,3]
        det = torch.linalg.det(R)                        # [B,N]
        flip = det < 0
        # flip t2 to enforce right-handed
        t2 = torch.where(flip.unsqueeze(-1), -t2, t2)
        R = torch.stack((t1, t2, n), dim=-1)

        # normalize columns (numerical safety)
        R = torch.stack([F.normalize(R[..., :, i], dim=-1) for i in range(3)], dim=-1)

        n = R[..., :, 2]  # consistent with possibly flipped frame
        return R, n
    
    def update_knn(self, x: torch.Tensor) -> None:
        reshaped_smpl_original = x.reshape(self.num_instances, self.smpl_points_num, 3)
        self.fps_node_idx = pointutils.furthest_point_sample(reshaped_smpl_original, self.n_nodes)   # (G,)
        _, self.rot_nn_ind, _ = knn_points(reshaped_smpl_original, reshaped_smpl_original, K=self.knn_num, return_nn=False)
        _, self.nn_ind, _ = knn_points(self.reshape_means, self.reshape_means, K=self.ctrl_cfg.knn_neighbors)

    
    def get_deformation(self, local_means, use_prev=False) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """
        assert local_means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        nonrigid_embed = self.instances_embedding[self.point_ids[..., 0]]
        ins_height = self.instances_size[self.point_ids[..., 0]][..., 2]
        x = local_means.data / ins_height[:, None] * 2
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.point_ids.shape[0], 1)
        delta_xyz_norm = self.deform_network(x, t, nonrigid_embed)
        # delta_xyz, delta_quat, delta_scale = self.deform_network(x, t)
        return delta_xyz_norm  

    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
        depth: torch.Tensor
    ) -> None:
        self.after_train(radii, xys_grad, last_size, depth)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)


    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
        depth: torch.Tensor
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask]    = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )

    def deformation_graph_pipeline_preprocess(self, P1, P2, X1=None, quats_P1=None, quats_P2=None, n_nodes=500, K_node=16, use_theta=False):
        """
        Args:
            P1: (B,N,3) SMPL verts Pose1
            P2: (B,N,3) SMPL verts Pose2
            X1: (B,M,3) Pointcloud Pose1
            n_nodes: FPS sampling pts size
            K_node: neighbourhood pts size (for R)
            K_point: K 
        Returns:
            X2: (M,3) Pointcloud Pose2
        """
        assert P1.shape[0] == P2.shape[0], "P1.shape[0] must be equal to P2.shape[0]"
        B = P1.shape[0]
        node_quats_P1 = None
        node_quats_P2 = None
        
        # Step1: FPS method sample key points
        # node_idx = pointutils.furthest_point_sample(P1, n_nodes)   # (G,)
        # G1, G2 = P1[node_idx], P2[node_idx]  # (G,3)
        G1 = pointutils.gather_operation(P1.transpose(1,2).contiguous(), self.fps_node_idx)
        G2 = pointutils.gather_operation(P2.transpose(1,2).contiguous(), self.fps_node_idx)
        G1 = G1.permute(0,2,1).contiguous()
        G2 = G2.permute(0,2,1).contiguous()

        if quats_P1 is not None and quats_P2 is not None:
            node_quats_P1 = pointutils.gather_operation(quats_P1.transpose(1,2).contiguous(), self.fps_node_idx)
            node_quats_P2 = pointutils.gather_operation(quats_P2.transpose(1,2).contiguous(), self.fps_node_idx)
            node_quats_P1 = node_quats_P1.permute(0,2,1).contiguous()
            node_quats_P2 = node_quats_P2.permute(0,2,1).contiguous()

        if node_quats_P1 is not None and node_quats_P2 is not None:
            Qr = quaternion_multiply(node_quats_P2, quaternion_invert(node_quats_P1))  # (1,G,4)
            R = quaternion_to_matrix(Qr)  # (1,G,3,3)
        else:
            # _, knn_idx_nodes = pointutils.knn(K_node, G1, P1)
            node_idx_expanded = self.fps_node_idx.unsqueeze(-1).expand(-1, -1, K_node).to(torch.int64)  # (B, M, K)
            knn_idx_nodes = torch.gather(self.rot_nn_ind, dim=1, index=node_idx_expanded)   # (B, M, K)
            # R, G1_mean, G2_mean, max_R_eign = compute_node_rotations(P1, P2, knn_idx_nodes, None)  # (G,3,3)
            R, max_R_eign = compute_node_rotations(P1, P2, knn_idx_nodes)  # (G,3,3)
            # if not use_theta:
            Qr = matrix_to_quaternion(R.reshape(-1,3,3)).reshape(B, n_nodes, 4)

        # node_cloth_theta = None
        # if use_theta:
        #     node_cloth_theta = self.cloth_rots_theta.reshape(-1, self.smpl_points_num, 9).permute(0,2,1).contiguous()
        #     # node_cloth_theta = self.cloth_rots_theta.permute(0,2,1).contiguous()
        #     node_cloth_theta = pointutils.gather_operation(node_cloth_theta, node_idx)
        #     node_cloth_theta = node_cloth_theta.permute(0,2,1).contiguous()
        #     # R_delta = so3_exp(node_cloth_theta.reshape(-1,3))         # (M,3,3)  exp([ω]_x)
        #     # R_cloth = R_delta.reshape(B, n_nodes, 3, 3) @ R         # (M,N,3,3)
        #     # Qr = matrix_to_quaternion(R_cloth) 
        #     node_cloth_theta = node_cloth_theta.reshape(node_cloth_theta.shape[0], node_cloth_theta.shape[1], 3, 3)
        Qr = self.quat_act(Qr)

        return G1, G2, R, Qr#R_delta.reshape(B, n_nodes, 3, 3) 
    
    def get_ref_deformations(self, X1=None, K_node=11, instance_mask=None, use_theta=False, canical_mode=False):
        if canical_mode:
            quats = torch.cat([self.instances_quats[0], self.smpl_insts_quats[0]], dim=1)
            trans = self.instances_trans[0]
            smpl_can_quats = torch.zeros_like(quats)
            smpl_can_trans = torch.zeros_like(trans)
            for i, ins_id in enumerate(self.smpl_ids):
                quats = torch.cat([self.instances_quats[:,ins_id], self.smpl_insts_quats[:,ins_id]], dim=1)
                trans = self.instances_trans[:,ins_id]
                smpl_can_quats[ins_id] = quats[self.ref_smpl_fi[ins_id]]
                smpl_can_trans[ins_id] = trans[self.ref_smpl_fi[ins_id]] 
        
            rot_cur_frame = quat_to_rotmat(self.quat_act(smpl_can_quats[:,0,:].unsqueeze(1))).squeeze(1)          
            P = torch.tensor([[0.,1.,0.],
                    [0.,0.,1.],
                    [1.,0.,0.]]).to(self.device)

            num_instances = rot_cur_frame.shape[0]
            rot_cur_frame = torch.matmul(rot_cur_frame, P)  # (num_instances, 3, 3)

            o2w = torch.zeros((num_instances, 4, 4), device=rot_cur_frame.device, dtype=rot_cur_frame.dtype).to(self.device)
            o2w[:, :3, :3] = rot_cur_frame
            o2w[:, :3, 3]  = smpl_can_trans 
            o2w[:, 3, 3]   = 1.0
            w2o = torch.inverse(o2w)
            trans_cur_frame = smpl_can_trans

            W, A = self.template(
                masked_theta = smpl_can_quats, 
                instances_mask = instance_mask,
                xyz_canonical = self.smpl_original if self.use_voxel_deformer else None
            )
            T = torch.einsum("bnj, bjrc -> bnrc", W, A)
            R = T[:, :, :3, :3] # [N, 3, 3]
            t = T[:, :, :3, 3]  # [N, 3]
            
            reshaped_means = self.smpl_original.reshape(self.num_instances, self.smpl_points_num, 3)
            deformed_means = torch.einsum("bnij,bnj->bni", R, reshaped_means) + t  # [N, 6890, 3]

            # rot_per_pts = rot_cur_frame[self.smpl_point_ids[..., 0]]        # (num_points, 3, 3)
            trans_per_pts = trans_cur_frame[self.smpl_point_ids[..., 0]]
            # # # transform the means to world space
            deformed_means = deformed_means.reshape(-1, 3) + trans_per_pts
            deformed_means  = deformed_means.reshape(-1, self.smpl_points_num, 3)
        else:
            if self.in_test_set and (
                self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
            ):
                _prev_masked_theta = self.instances_quats[self.cur_frame - 1]#[instance_mask]
                _next_masked_theta = self.instances_quats[self.cur_frame + 1]#[instance_mask]
                _cur_masked_theta  = self.instances_quats[self.cur_frame]#[instance_mask]
                interpolated_theta = interpolate_quats(_prev_masked_theta, _next_masked_theta)
                
                inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
                quats_cur_frame = torch.where(inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta)

                _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
                _next_ins_trans = self.instances_trans[self.cur_frame + 1]
                _cur_ins_trans = self.instances_trans[self.cur_frame]
                interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
                
                trans_cur_frame = torch.where(inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans)#[instance_mask]
            else:
                quats_cur_frame = self.instances_quats[self.cur_frame]
                trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)

            rot_cur_frame = quat_to_rotmat(self.quat_act(quats_cur_frame)).squeeze(1)          
            P = torch.tensor([[0.,1.,0.],
                    [0.,0.,1.],
                    [1.,0.,0.]]).to(self.device)
        
            rot_cur_frame = torch.matmul(rot_cur_frame, P)  # (num_instances, 3, 3)

            num_instances = rot_cur_frame.shape[0]

            o2w = torch.zeros((num_instances, 4, 4), device=rot_cur_frame.device, dtype=rot_cur_frame.dtype).to(self.device)
            o2w[:, :3, :3] = rot_cur_frame
            o2w[:, :3, 3]  = trans_cur_frame
            o2w[:, 3, 3]   = 1.0
            w2o = torch.inverse(o2w)

            deformed_means  = self.transform_smpl_means(self.smpl_original) # Depend on the SMPL training stage 
            deformed_means  = deformed_means.reshape(-1, self.smpl_points_num, 3)

        smpl_verts_w2o2 = transform_points(deformed_means, w2o)
        G1, G2, R, Qr = self.deformation_graph_pipeline_preprocess(self.smpl_vertices, smpl_verts_w2o2, X1=X1, n_nodes=self.n_nodes, K_node=K_node, use_theta=use_theta) #K_node=15,the largest the better

        return G1, G2, R, Qr, rot_cur_frame, smpl_verts_w2o2

    
    def preprocess_per_frame(self, local_means, canical_mode=False):
        # frame_id = self.cur_frame.item()
        K = self.ctrl_cfg.knn_neighbors
        self.instance_mask    = torch.full([local_means.shape[0]], False, dtype=torch.bool, device=self.device)
        # full_feet_mask     = torch.full([local_means.shape[0]], False, dtype=torch.bool, device=self.device)
        full_offsets          = torch.zeros_like(self._means)
        # full_delta_g2_offsets = torch.zeros_like(self._means)
        full_quats_cur        = torch.zeros_like(self.get_quats)
        
        self.nn_list = []
        if canical_mode:
            instance_mask = torch.full([self.num_instances], True, dtype=torch.bool, device=self.device)
        else:
            instance_mask = self.instances_fv[self.cur_frame]
        
        G1, G2, R, Qr, rot_cur_frame, smpl_verts_w2o2 = self.get_ref_deformations(X1=self.reshape_means, K_node=11, instance_mask=instance_mask, use_theta=False, canical_mode=canical_mode)

        # local_means_cur  = []
        # sub_means_ind    = []
        # means_ind = torch.arange(local_means.shape[0], device=self.device, dtype=torch.int32)
        # G_ind = 0
        for i, ins_id in enumerate(self.smpl_ids):
            valid_mask = self.point_ids[:,0] == ins_id
            sub_means  = local_means[valid_mask]
            sub_quats  = self._quats[valid_mask]
            # sub_means_ind.append(means_ind[valid_mask])
            sub_hinge_pt_ids = self.hinge_pt_ids[valid_mask]
            sub_hinge_pt_mask = sub_hinge_pt_ids == 1
            if instance_mask[ins_id]:
                self.instance_mask[valid_mask] = True
                # new_sub_means, new_sub_quats, sub_feet_means, sub_feet_mask = deformation_graph_pipeline_fast(G1[ins_id], G2[ins_id], R[ins_id], Qr[ins_id], X1=sub_means, X1_quats=sub_quats, S1=None, P1=self.smpl_vertices[ins_id], theta=G_cloth_theta[ins_id], G2_feet_mask=G2_feet_mask[ins_id], K_point=1, X_theta=X_theta) 
                new_sub_means, new_sub_quats, _ = deformation_graph_pipeline_fast(G1[ins_id], G2[ins_id], R[ins_id], Qr[ins_id], X1=sub_means, X1_quats=sub_quats, S1=None, P1=self.smpl_vertices[ins_id], K_point=1) 
                if sub_hinge_pt_ids.sum() > 0:
                    sub_hinge_pt = sub_means[sub_hinge_pt_mask]
                    hinge_trans = G2[ins_id].mean(0) - G1[ins_id].mean(0)
                    sub_hinge_pt = sub_hinge_pt @ rot_cur_frame[ins_id].transpose(0,1) + hinge_trans
                    new_sub_means[sub_hinge_pt_mask] = sub_hinge_pt
                    hinge_Qr = matrix_to_quaternion(rot_cur_frame[ins_id].reshape(-1,3,3))
                    hinge_Qr = self.quat_act(hinge_Qr)
                    hinge_Qr = quat_mult(hinge_Qr, sub_quats[sub_hinge_pt_mask])
                    new_sub_quats[sub_hinge_pt_mask] = self.quat_act(hinge_Qr)

                full_offsets[valid_mask]   = new_sub_means - sub_means
                full_quats_cur[valid_mask] = new_sub_quats
            else:
                full_quats_cur[valid_mask] = sub_quats

        # delta_xyz_norm = None
        # if self.ctrl_cfg.use_deformgs_for_nonrigid and self.step > self.ctrl_cfg.use_deformgs_after: #
        #     delta_xyz_norm = self.get_deformation(local_means=self._means+full_offsets)
        #     full_offsets = full_offsets + delta_xyz_norm * self.normal_vecs
            
        return full_offsets, full_quats_cur, smpl_verts_w2o2
    
    def transform_smpl_means(self, means: torch.Tensor, smpl_mode=True) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        if smpl_mode:
            assert means.shape[0] == self.smpl_point_ids.shape[0], \
                "its a bug here, we need to pass the mask for smpl_points_ids"
        else:
            assert means.shape[0] == self.point_ids.shape[0], \
                "its a bug here, we need to pass the mask for points_ids"
        instance_mask = self.instances_fv[self.cur_frame]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_masked_theta = torch.cat((self.instances_quats[self.cur_frame - 1], self.smpl_insts_quats[self.cur_frame - 1]), dim=1)[instance_mask]
            _next_masked_theta = torch.cat((self.instances_quats[self.cur_frame + 1], self.smpl_insts_quats[self.cur_frame + 1]), dim=1)[instance_mask]
            _cur_masked_theta = torch.cat((self.instances_quats[self.cur_frame], self.smpl_insts_quats[self.cur_frame]), dim=1)[instance_mask]
            interpolated_theta = interpolate_quats(_prev_masked_theta, _next_masked_theta)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1, instance_mask] & self.instances_fv[self.cur_frame + 1, instance_mask]
            masked_theta = torch.where(
                inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta
            )

            # ############################
            _quats_prev_frame = self.instances_quats[self.cur_frame - 1]
            _quats_next_frame = self.instances_quats[self.cur_frame + 1]
            _quats_cur_frame  = self.instances_quats[self.cur_frame]
            interpolated_quats = interpolate_quats(_quats_prev_frame, _quats_next_frame)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            quats_cur_frame = torch.where(
                inter_valid_mask[:, None, None], interpolated_quats, _quats_cur_frame
            )
        else:
            theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_insts_quats[self.cur_frame]), dim=1
            )
            masked_theta = theta[instance_mask]
            quats_cur_frame = self.instances_quats[self.cur_frame]
        
        if smpl_mode:
            masked_theta = self.quat_act(masked_theta)
            W, A = self.template(
                masked_theta = masked_theta, 
                instances_mask = instance_mask,
                xyz_canonical = means.reshape(self.num_instances, self.smpl_points_num, 3) if self.use_voxel_deformer else None
            )
            T = torch.einsum("bnj, bjrc -> bnrc", W, A)
            R = T[:, :, :3, :3] # [N, 3, 3]
            t = T[:, :, :3, 3]  # [N, 3]
            
            reshaped_means = means.reshape(self.num_instances, self.smpl_points_num, 3)
            deformed_means = torch.einsum(
                "bnij,bnj->bni", R, reshaped_means[instance_mask]         
            ) + t  # [N, 6890, 3]
            
            means_container = torch.zeros_like(reshaped_means)
            means_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_means)
            means_container = means_container.reshape(-1, 3)
        else:
            # quats_cur_frame = self.instances_quats[self.cur_frame] #这边需要更改为masked_theta
            
            quats_cur_frame = quats_cur_frame.squeeze(1)  
            rot_cur_frame = quat_to_rotmat(
                            self.quat_act(quats_cur_frame)
                            )#.squeeze(1)          
            P = torch.tensor([[0.,1.,0.],
                            [0.,0.,1.],
                            [1.,0.,0.]]).to(self.device)
        
            rot_cur_frame = torch.matmul(rot_cur_frame, P)  # (num_instances, 3, 3)
            rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]  # (num_points, 3, 3)

            means_container = torch.bmm(
                rot_per_pts, means.unsqueeze(-1)
            ).squeeze(-1)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)

        if smpl_mode:
            trans_per_pts = trans_cur_frame[self.smpl_point_ids[..., 0]]
        else:
            trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means_container += trans_per_pts
        return means_container
    
    def transform_smpl_means_and_quats(self, means: torch.Tensor, quats: torch.Tensor, smpl_mode=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        transform the means and quats of gaussians to world space
        according to the pose at the current frame
        """
        if smpl_mode:
            assert means.shape[0] == self.smpl_point_ids.shape[0], \
                "its a bug here, we need to pass the mask for smpl_points_ids"
        else:
            assert means.shape[0] == self.point_ids.shape[0], \
                "its a bug here, we need to pass the mask for points_ids"
        instance_mask = self.instances_fv[self.cur_frame]
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_masked_theta = torch.cat((self.instances_quats[self.cur_frame - 1], self.smpl_insts_quats[self.cur_frame - 1]), dim=1)[instance_mask]
            _next_masked_theta = torch.cat((self.instances_quats[self.cur_frame + 1], self.smpl_insts_quats[self.cur_frame + 1]), dim=1)[instance_mask]
            _cur_masked_theta = torch.cat((self.instances_quats[self.cur_frame], self.smpl_insts_quats[self.cur_frame]), dim=1)[instance_mask]
            interpolated_theta = interpolate_quats(_prev_masked_theta, _next_masked_theta)
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1, instance_mask] & self.instances_fv[self.cur_frame + 1, instance_mask]
            masked_theta = torch.where(
                inter_valid_mask[:, None, None], interpolated_theta, _cur_masked_theta
            )
        else:
            theta = torch.cat(
                (self.instances_quats[self.cur_frame], self.smpl_insts_quats[self.cur_frame]), dim=1
            )
            masked_theta = theta[instance_mask]

        if smpl_mode:
            masked_theta = self.quat_act(masked_theta)
            W, A = self.template(
                masked_theta = masked_theta, 
                instances_mask = instance_mask,
                xyz_canonical = means.reshape(self.num_instances, self.smpl_points_num, 3) if self.use_voxel_deformer else None
            )
            T = torch.einsum("bnj, bjrc -> bnrc", W, A)
            R = T[:, :, :3, :3] # [N, 3, 3]
            t = T[:, :, :3, 3]  # [N, 3]
            
            reshaped_means = means.reshape(self.num_instances, self.smpl_points_num, 3)
            deformed_means = torch.einsum(
                "bnij,bnj->bni", R, reshaped_means[instance_mask]         
            ) + t  # [N, 6890, 3]
            
            means_container = torch.zeros_like(reshaped_means)
            means_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_means)
            means_container = means_container.reshape(-1, 3)
        else:
            # quats_cur_frame = self.instances_quats[self.cur_frame] 
            quats_cur_frame = self.instances_quats[self.cur_frame] 
            rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
            ).squeeze(1)          
            P = torch.tensor([[0.,1.,0.],
                            [0.,0.,1.],
                            [1.,0.,0.]]).to(self.device)
        
            rot_cur_frame = torch.matmul(rot_cur_frame, P)  # (num_instances, 3, 3)
            rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]        # (num_points, 3, 3)

            means_container = torch.bmm(
                rot_per_pts, means.unsqueeze(-1)
            ).squeeze(-1)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5
            
            inter_valid_mask = self.instances_fv[self.cur_frame - 1] & self.instances_fv[self.cur_frame + 1]
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame] # (num_instances, 3)

        if smpl_mode:
            trans_per_pts = trans_cur_frame[self.smpl_point_ids[..., 0]]
        else:
            trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means_container += trans_per_pts
        
        reshaped_quats = quats.reshape(self.num_instances, self.smpl_points_num, 4)
        R_quats = matrix_to_quaternion(R)
        deformed_quats = quat_mult(
            self.quat_act(R_quats),
            self.quat_act(reshaped_quats[instance_mask])
        )
        quats_container = torch.zeros_like(reshaped_quats)
        quats_container.index_add_(0, instance_mask.nonzero().squeeze(), deformed_quats)
        # fill other with [1, 0, 0, 0]
        quats_container.index_add_(0, (~instance_mask).nonzero().squeeze(), torch.tensor([[[1., 0., 0., 0.]]], device=self.device).repeat((~instance_mask).sum(), self.smpl_points_num, 1))
        quats_container = quats_container.reshape(-1, 4)
        return means_container, quats_container
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        # smpl_gs_dict = None
        # smpl_gs_dict = self.get_smpl_gaussians(cam, render_fn) 

        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask

        delta_xyz, delta_quat, delta_scale, delta_xyz_norm = None, None, None, None

        delta_xyz, delta_quat, _ = self.preprocess_per_frame(local_means=self._means)
        
        means = self._means + delta_xyz
        world_means = self.transform_smpl_means(means, False)

        if delta_quat is not None:
            # quats = self.get_quats + delta_quat
            world_quats = self.transform_quats(delta_quat)
        else:
            world_quats = self.transform_quats(self._quats)
        world_quats = world_quats.squeeze(1)
        
        if delta_scale is not None:
            activated_scales = self.get_scaling + delta_scale
        else:
            activated_scales = self.get_scaling

        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        valid_mask = self.get_pts_valid_mask()
            
        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_rotations = self.quat_act(world_quats)
        activated_colors = rgbs

        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=activated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )

        # check nan in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
            "local_xyz_deformed": means[filter_mask] if delta_xyz is not None else None,
        }
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = super().compute_reg_loss()
        out_of_bound_losscfg = self.reg_cfg.get("out_of_bound_loss", None)
        if out_of_bound_losscfg is not None:
            w = out_of_bound_losscfg.w_pos
            local_xyz_deformed = self._gs_cache["local_xyz_deformed"]
            if w > 0 and local_xyz_deformed is not None:
                local_xyz_deformed = self._gs_cache["local_xyz_deformed"]
                per_pts_size = self.instances_size[self.point_ids[..., 0]]
                loss_dict["out_of_bound_loss"] = torch.relu(local_xyz_deformed.abs() - per_pts_size / 2).mean() * w

        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            joint_smooth_reg = temporal_smooth_reg.get("joint_smooth", None)
            if joint_smooth_reg is not None:
                if self.cur_frame >= 1 and self.cur_frame < self.num_frames - 1:
                    valid_mask = (
                        self.instances_fv[self.cur_frame - 1] & \
                        self.instances_fv[self.cur_frame + 1] & \
                        self.instances_fv[self.cur_frame]
                    )
                    cur_theta = torch.cat(
                        (self.instances_quats[self.cur_frame], self.smpl_qauts[self.cur_frame]), dim=1
                    )[valid_mask]
                    next_theta = torch.cat(
                        (self.instances_quats[self.cur_frame + 1], self.smpl_qauts[self.cur_frame + 1]), dim=1
                    )[valid_mask]
                    prev_theta = torch.cat(
                        (self.instances_quats[self.cur_frame - 1], self.smpl_qauts[self.cur_frame - 1]), dim=1
                    )[valid_mask]
                    thetas = torch.vstack([prev_theta, cur_theta, next_theta])
                    thetas = self.quat_act(thetas)
                    J_transformed, _ = batch_rigid_transform(
                        quaternion_to_matrix(thetas),
                        self.template.J_canonical[valid_mask].repeat(3, 1, 1),
                        self.template._template_layer.parents,
                    )
                    
                    cur_trans = self.instances_trans[self.cur_frame, valid_mask]
                    next_trans = self.instances_trans[self.cur_frame + 1, valid_mask]
                    prev_trans = self.instances_trans[self.cur_frame - 1, valid_mask]
                    trans = torch.vstack([prev_trans, cur_trans, next_trans])
                    J_transformed += trans.unsqueeze(-2)
                    J_transformed = J_transformed.reshape(3, -1, 24, 3)
                    
                    velocity_prev = (J_transformed[1] - J_transformed[0])
                    velocity_next = (J_transformed[2] - J_transformed[1])
                    # l2 loss
                    loss_dict["smpl_temporal_smooth"] = (velocity_next - velocity_prev).abs().mean() \
                        * joint_smooth_reg.w
        

        knn_reg = self.reg_cfg.get("knn_reg", None)
        if knn_reg is not None and False:
            K = self.ctrl_cfg.knn_neighbors
            instances_mask = self.instances_fv[self.cur_frame]
            nn_ind = self.nn_ind[instances_mask] # (num_instances, smpl_points_num, knn_neighbors)
            
            if not self.ctrl_cfg.freeze_shs_dc:
                valid_shs_dc = self._features_dc[self.reshape_means_ind].reshape(self.num_instances, self.ins_num, 3)[instances_mask] # (num_instances, smpl_points_num, 3)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 3)
                knn_shs_dc = torch.gather(valid_shs_dc.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded) # (num_instances, smpl_points_num, knn_neighbors, 3)
                shs_dc_std = knn_shs_dc.std(dim=2).mean()
                loss_dict["knn_reg_dc"] = shs_dc_std * knn_reg.lambda_std_shs_dc
            
            if not self.ctrl_cfg.freeze_shs_rest and self.sh_degree > 0:
                dim_sh = num_sh_bases(self.sh_degree)
                valid_shs_rest = self._features_rest[self.reshape_means_ind].reshape(self.num_instances, self.ins_num, -1)[instances_mask] # (num_instances, smpl_points_num, (dim_sh-1)*3)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, (dim_sh-1)*3)
                knn_shs_rest = torch.gather(valid_shs_rest.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded) # (num_instances, smpl_points_num, knn_neighbors, (dim_sh-1)*3)
                shs_rest_std = knn_shs_rest.std(dim=2).mean()
                loss_dict["knn_reg_rest"] = shs_rest_std * knn_reg.lambda_std_shs_rest
            
            if not self.ctrl_cfg.freeze_o:
                valid_o = self.get_opacity[self.reshape_means_ind].reshape(self.num_instances, self.ins_num, 1)[instances_mask] # (num_instances, smpl_points_num, 1)
                nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 1)
                knn_o = torch.gather(valid_o.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
                o_std = knn_o.std(dim=2).mean()
                loss_dict["knn_reg_o"] = o_std * knn_reg.lambda_std_o

            # if not self.ctrl_cfg.freeze_s:
            #     scale_dim = 1 if self.ball_gaussians else 3
            #     valid_s = self.get_scaling[self.reshape_means_ind].reshape(self.num_instances, self.ins_num, scale_dim)[instances_mask] # (num_instances, smpl_points_num, 1)
            #     nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, scale_dim)
            #     knn_s = torch.gather(valid_s.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
            #     s_std = knn_s.std(dim=2).mean()
            #     loss_dict["knn_reg_s"] = s_std * knn_reg.lambda_std_s
            
            # if not self.ctrl_cfg.freeze_q:
            #     valid_q = self._quats[self.reshape_means_ind].reshape(self.num_instances, self.ins_num, 4)[instances_mask] # (num_instances, smpl_points_num, 4)
            #     nn_ind_expanded = nn_ind.unsqueeze(-1).expand(-1, -1, -1, 4)
            #     knn_q = torch.gather(valid_q.unsqueeze(2).expand(-1, -1, K, -1), 1, nn_ind_expanded)
            #     q_std = knn_q.std(dim=2).mean()
            #     loss_dict["knn_reg_q"] = q_std * knn_reg.lambda_std_q
        
  
        x_offset_reg = self.reg_cfg.get("x_offset", None)
        if x_offset_reg is not None and self.ctrl_cfg.constrain_xyz_offset and not self.ctrl_cfg.freeze_x:
            instances_mask = self.instances_fv[self.cur_frame]
            valid_x = self.smpl_means.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask] # (num_instances, smpl_points_num, 3)
            valid_x_on_mesh = self.on_mesh_x.reshape(self.num_instances, self.smpl_points_num, 3)[instances_mask]
            x_offset = (valid_x - valid_x_on_mesh).norm(dim=-1).mean()
            
            loss_dict["x_offset"] = x_offset * x_offset_reg.w

        return loss_dict
    
    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")

            new_avg_grad_norm, avg_grad_norm = None, None
            dense_flag = False
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                dense_flag = True
                avg_grad_norm = self.xys_grad_norm / self.vis_counts #(self.vis_counts + 1e-6)**0.5 #
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                    new_smpl_masks,
                    new_keypt_masks,
                    new_hinge_pt_ids,
                    # split_normal_vecs
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                    dup_smpl_masks,
                    dup_keypt_masks,
                    dup_hinge_pt_ids,
                    # dup_normal_vecs
                ) = self.dup_gaussians(dups)

                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)
                # self.normal_vecs = torch.cat([self.normal_vecs, split_normal_vecs, dup_normal_vecs], dim=0)
                if self.smpl_pt_masks is not None:
                    self.smpl_pt_masks  = torch.cat([self.smpl_pt_masks, new_smpl_masks, dup_smpl_masks], dim=0)
                    # split_avg_grad_norm = torch.zeros(new_smpl_masks.shape[0], dtype=avg_grad_norm.dtype).to(self._means.device)
                    # dup_avg_grad_norm = torch.zeros(dup_smpl_masks.shape[0], dtype=avg_grad_norm.dtype).to(self._means.device)
                    low, high = 1e-7, 1e-4
                    split_avg_grad_norm = (
                        torch.empty(new_smpl_masks.shape[0], dtype=avg_grad_norm.dtype, device=self._means.device)
                        .uniform_(low, high)
                    )
                    dup_avg_grad_norm = (
                        torch.empty(dup_smpl_masks.shape[0], dtype=avg_grad_norm.dtype, device=self._means.device)
                        .uniform_(low, high)
                    )
                    new_avg_grad_norm = torch.cat([avg_grad_norm, split_avg_grad_norm, dup_avg_grad_norm], dim=0)
                else:
                    # split_avg_grad_norm = torch.zeros(new_smpl_masks.shape[0], dtype=avg_grad_norm.dtype).to(self._means.device)
                    # dup_avg_grad_norm = torch.zeros(dup_smpl_masks.shape[0], dtype=avg_grad_norm.dtype).to(self._means.device)
                    new_avg_grad_norm = avg_grad_norm
                    print("WARN: self.smpl_pt_masks is None.")
                
                if self.keypt_masks is not None:
                    self.keypt_masks = torch.cat([self.keypt_masks, new_keypt_masks, dup_keypt_masks], dim=0)
                else:
                    print("WARN: self.keypt_masks is None.")

                if self.hinge_pt_ids is not None:
                    self.hinge_pt_ids = torch.cat([self.hinge_pt_ids, new_hinge_pt_ids, dup_hinge_pt_ids], dim=0)
                else:
                    print("WARN: self.hinge_pt_ids is None.")

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

                # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                    # save checkpoints right when the opacity is reset (saves every 2k)
                culls_flag = False
                if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                    culls_flag = True
                    deleted_mask = self.cpd_cull_gaussians_V2(new_avg_grad_norm)
                    param_groups = self.get_gaussian_param_groups()
                    remove_from_optim(optimizer, deleted_mask, param_groups)
                    if self.smpl_pt_masks is not None and deleted_mask.numel() > 0:
                        if new_avg_grad_norm is not None:
                            new_avg_grad_norm = new_avg_grad_norm[~deleted_mask]
                        elif avg_grad_norm is not None:
                            new_avg_grad_norm = avg_grad_norm[~deleted_mask]
                        else:
                            new_avg_grad_norm = self.xys_grad_norm / self.vis_counts
                            new_avg_grad_norm = new_avg_grad_norm[~deleted_mask]
                    else:
                        if new_avg_grad_norm is not None:
                            new_avg_grad_norm = new_avg_grad_norm
                        elif avg_grad_norm is not None:
                            new_avg_grad_norm = avg_grad_norm
                        else:
                            new_avg_grad_norm = self.xys_grad_norm / self.vis_counts

            print(f"Class {self.class_prefix} left points: {self.num_points}")

            means_cur, sub_means_ind = [], []
            means_ind = torch.arange(self._means.shape[0], device=self.device, dtype=torch.int32)
            for i, ins_id in enumerate(self.smpl_ids):
                pc_mask = self.point_ids[:,0] == ins_id
                sub_means = self._means[pc_mask]
                sub_means_ind.append(means_ind[pc_mask])
                means_cur.append(sub_means.unsqueeze(0).contiguous())
            means_cur = torch.cat(means_cur, dim=0)
            sub_means_ind = torch.cat(sub_means_ind, dim=0)
            self.reshape_means = means_cur
            self.reshape_means_ind = sub_means_ind

            _, self.nn_ind, _ = knn_points(self.reshape_means, self.reshape_means, K=self.ctrl_cfg.knn_neighbors)
            # instance_mask = torch.full([self.num_instances], True, dtype=torch.bool, device=self.device)
            # _, _, _, _, _, remap_smpl_can_means = self.get_ref_deformations(X1=self.reshape_means, K_node=11, instance_mask=instance_mask, use_theta=False, canical_mode=True)
            # self.smpl_vertices = remap_smpl_can_means.detach()

            # if self.smpl_pt_masks is not None and (dense_flag or culls_flag):
            #     self.update_parameters(new_avg_grad_norm, optimizer)

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
                
        culls = culls & (~self.keypt_masks)
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        if self.smpl_pt_masks is not None:
            self.smpl_pt_masks = self.smpl_pt_masks[~culls]


        if self.keypt_masks is not None:
            self.keypt_masks = self.keypt_masks[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls
    
    def cpd_cull_gaussians(self, new_avg_grad_norm):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        is_keypt = self.keypt_masks            # Highest priority, last to be removed
        is_smpl = self.smpl_pt_masks[:,0]      # Second highest priority
        # Priority masks
        lowest_mask = (~is_keypt) & (~is_smpl)
        mid_mask = (~is_keypt) & (is_smpl)
        high_mask = is_keypt

        culls = torch.zeros(self._means.shape[0], dtype=torch.bool, device=self.device)

        # Build opacity_culls mask
        opacity_culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            opacity_culls = opacity_culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            opacity_culls = opacity_culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                opacity_culls = opacity_culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()

        for i, ins_id in enumerate(self.smpl_ids):
            pc_mask = self.point_ids[:,0] == ins_id
            sub_means = self._means[pc_mask].data
            sub_avg_grad_norm = new_avg_grad_norm[pc_mask]
            sub_opacity_culls = opacity_culls[pc_mask]  
            
            total_to_cull = sub_means.shape[0] - self.ins_num
            if total_to_cull <= 0:
                continue

            def cull_level(level_mask, remaining_to_cull):
                if remaining_to_cull <= 0:
                    return 0
                
                sub_level_mask = level_mask[pc_mask]
                n_level = int(sub_level_mask.sum())
                if n_level == 0:
                    return remaining_to_cull
                
                # High priority to cull the points with low opacities
                sub_level_opacity = sub_level_mask & sub_opacity_culls
                n_opacity = int(sub_level_opacity.sum())
                
                # Low priority to cull the points without low opacities
                sub_level_non_opacity = sub_level_mask & (~sub_opacity_culls)
                n_non_opacity = int(sub_level_non_opacity.sum())
                
                culled_in_level = 0
                
                # Step 1
                if n_opacity > 0 and remaining_to_cull > 0:
                    to_cull_opacity = min(n_opacity, remaining_to_cull)
                    if to_cull_opacity < n_opacity:
                        grad_opacity = sub_avg_grad_norm[sub_level_opacity]
                        _, inds = torch.topk(grad_opacity, to_cull_opacity, largest=False)
                        local_inds = sub_level_opacity.nonzero(as_tuple=True)[0][inds]
                    else:
                        local_inds = sub_level_opacity.nonzero(as_tuple=True)[0]
                    
                    global_inds = pc_mask.nonzero(as_tuple=True)[0][local_inds]
                    culls[global_inds] = True
                    culled_in_level += to_cull_opacity
                    remaining_to_cull -= to_cull_opacity
                
                # Step 2
                if n_non_opacity > 0 and remaining_to_cull > 0:
                    to_cull_non_opacity = min(n_non_opacity, remaining_to_cull)
                    grad_non_opacity = sub_avg_grad_norm[sub_level_non_opacity]
                    _, inds = torch.topk(grad_non_opacity, to_cull_non_opacity, largest=False)
                    local_inds = sub_level_non_opacity.nonzero(as_tuple=True)[0][inds]
                    global_inds = pc_mask.nonzero(as_tuple=True)[0][local_inds]
                    culls[global_inds] = True
                    culled_in_level += to_cull_non_opacity
                    remaining_to_cull -= to_cull_non_opacity
                
                return remaining_to_cull


            total_to_cull = cull_level(lowest_mask, total_to_cull)  # Level 1: Lowest
            total_to_cull = cull_level(mid_mask, total_to_cull)     # Level 2: Mid
            total_to_cull = cull_level(high_mask, total_to_cull)    # Level 3: Highest (keypoints)
            
            if total_to_cull > 0:
                print(f"Warning: instance {ins_id} need to be removed {total_to_cull} points. However, there are no points left to eliminate.")

        # Apply culling
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]
        # self.normal_vecs = self.normal_vecs[~culls]

        if self.smpl_pt_masks is not None:
            self.smpl_pt_masks = self.smpl_pt_masks[~culls]
        if self.keypt_masks is not None:
            self.keypt_masks = self.keypt_masks[~culls]
        if self.hinge_pt_ids is not None:
            self.hinge_pt_ids = self.hinge_pt_ids[~culls]
        print(f"     Cull: {n_bef - self.num_points}")
        return culls
    
    def cpd_cull_gaussians_V2(self, new_avg_grad_norm):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        is_keypt = self.keypt_masks            # Highest priority, last to be removed
        is_smpl = self.smpl_pt_masks[:,0]      # Second highest priority
        # Priority masks
        lowest_mask = (~is_keypt) & (~is_smpl)
        mid_mask = (~is_keypt) & (is_smpl)
        high_mask = is_keypt

        culls = torch.zeros(self._means.shape[0], dtype=torch.bool, device=self.device)

        # Build opacity_culls mask
        opacity_culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            opacity_culls = opacity_culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale  #31.64
            ).squeeze()
            opacity_culls = opacity_culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                opacity_culls = opacity_culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()

        for i, ins_id in enumerate(self.smpl_ids):
            pc_mask = self.point_ids[:,0] == ins_id
            sub_means = self._means[pc_mask].data
            sub_opacity_culls = opacity_culls[pc_mask]
            sub_avg_grad_norm = new_avg_grad_norm[pc_mask]
            
            total_to_cull = sub_means.shape[0] - self.ins_num
            if total_to_cull <= 0:
                continue

            culled_local = torch.zeros(sub_means.shape[0], dtype=torch.bool, device=self.device)

            def cull_level(level_mask, remaining_to_cull):
                if remaining_to_cull <= 0:
                    return 0
                
                sub_level_mask = level_mask[pc_mask]

                sub_level_mask = sub_level_mask & (~culled_local)
                n_level = int(sub_level_mask.sum())
                if n_level == 0:
                    return remaining_to_cull
                
                # Step 1
                sub_level_opacity = sub_level_mask & sub_opacity_culls
                n_opacity = int(sub_level_opacity.sum())
                
                if n_opacity > 0 and remaining_to_cull > 0:
                    to_cull_opacity = min(n_opacity, remaining_to_cull)
                    local_inds = sub_level_opacity.nonzero(as_tuple=True)[0][:to_cull_opacity]
                    global_inds = pc_mask.nonzero(as_tuple=True)[0][local_inds]
                    culls[global_inds] = True
                    culled_local[local_inds] = True
                    remaining_to_cull -= to_cull_opacity
                
                # Step 2
                if remaining_to_cull > 0:
                    sub_level_remaining = sub_level_mask & (~culled_local)
                    n_remaining = int(sub_level_remaining.sum())
                    if n_remaining > 0:
                        to_cull = min(n_remaining, remaining_to_cull)
                        grad_remaining = sub_avg_grad_norm[sub_level_remaining]
                        _, inds = torch.topk(grad_remaining, to_cull, largest=False)
                        local_inds = sub_level_remaining.nonzero(as_tuple=True)[0][inds]
                        global_inds = pc_mask.nonzero(as_tuple=True)[0][local_inds]
                        culls[global_inds] = True
                        culled_local[local_inds] = True
                        remaining_to_cull -= to_cull
                
                return remaining_to_cull

            # levle-by-level
            total_to_cull = cull_level(lowest_mask, total_to_cull)
            total_to_cull = cull_level(mid_mask, total_to_cull)
            total_to_cull = cull_level(high_mask, total_to_cull)
            
            if total_to_cull > 0:
                print(f"Warning: instance {ins_id} need to be removed {total_to_cull} points. However, there are no points left to eliminate.")

        # Apply culling
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]
        # self.normal_vecs = self.normal_vecs[~culls]
        if self.smpl_pt_masks is not None:
            self.smpl_pt_masks = self.smpl_pt_masks[~culls]
        if self.keypt_masks is not None:
            self.keypt_masks = self.keypt_masks[~culls]
        if self.hinge_pt_ids is not None:
            self.hinge_pt_ids = self.hinge_pt_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new ids
        new_ids = self.point_ids[split_mask].repeat(samps, 1)


        new_smpl_masks = None
        if self.smpl_pt_masks is not None:
            new_smpl_masks = self.smpl_pt_masks[split_mask].repeat(samps, 1)

        new_keypt_masks = None
        if self.keypt_masks is not None:
            new_keypt_masks = torch.full_like(new_smpl_masks, False)[:, 0]
        
        new_hinge_pt_ids = None
        if self.hinge_pt_ids is not None:
            new_hinge_pt_ids = self.hinge_pt_ids[split_mask].repeat(samps)
        
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_ids, new_smpl_masks, new_keypt_masks, new_hinge_pt_ids

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_ids = self.point_ids[dup_mask]
        # dup_normal_vecs = self.normal_vecs[dup_mask]
        dup_smpl_masks = None
        if self.smpl_pt_masks is not None:
            dup_smpl_masks = self.smpl_pt_masks[dup_mask]
        dup_keypt_masks = None
        if self.keypt_masks is not None:
            dup_keypt_masks = torch.full_like(dup_smpl_masks, False)[:,0]

        dup_hinge_pt_ids = None
        if self.hinge_pt_ids is not None:
            dup_hinge_pt_ids = self.hinge_pt_ids[dup_mask]

        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_ids, dup_smpl_masks, dup_keypt_masks, dup_hinge_pt_ids
    
    def deform_gaussian_points(
        self, gaussian_dict: Dict[str, torch.Tensor], cur_normalized_time: float,
    ) -> Dict[str, torch.Tensor]:
        """
        deform the points
        """
        means = gaussian_dict["means"]
        nonrigid_embed = self.instances_embedding[gaussian_dict["ids"].squeeze()]
        cur_normalized_time = torch.tensor(cur_normalized_time, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(means.shape[0], 1)
        delta_xyz, delta_quat, delta_scale = self.deform_network(means, cur_normalized_time, nonrigid_embed)
        gaussian_dict["means"] = means + delta_xyz
        if delta_scale is not None:
            gaussian_dict["scales"] = gaussian_dict["scales"] + delta_scale
        if delta_quat is not None:
            gaussian_dict["quats"] = gaussian_dict["quats"] + delta_quat
        return gaussian_dict

    def load_state_dict(self, dict: Dict, **kwargs) -> str:
        instances_num = dict["instances_fv"].shape[1]
        frame_num = dict["instances_fv"].shape[0]
        self.instances_quats = torch.zeros(frame_num, instances_num, 1, 4, device=self.device)
        self.instances_trans = torch.zeros(frame_num, instances_num, 3, device=self.device)
        self.smpl_betas = torch.zeros(instances_num, 10, device=self.device)
        self.smpl_insts_quats = torch.zeros(frame_num, instances_num, 23, 4, device=self.device)
        self.smpl_point_ids = torch.zeros(instances_num*self.smpl_points_num, 1, dtype=torch.long, device=self.device)
        #############################
        # self.smpl_can_quats    = torch.zeros(instances_num, 24, 4, device=self.device)
        # self.smpl_can_trans    = torch.zeros(instances_num, 3, device=self.device)
        self.smpl_vertices     = torch.zeros(instances_num, self.smpl_points_num, 3, device=self.device)
        self.smpl_pt_masks     = torch.zeros(instances_num*self.ins_num, 1, dtype=torch.bool, device=self.device)
        self.keypt_masks       = torch.zeros(instances_num*self.ins_num, dtype=torch.bool, device=self.device)
        self.smpl_original     = torch.zeros(instances_num*self.smpl_points_num, 3, device=self.device)
        # self.smpl_faces        = torch.zeros(13776, 3, device=self.device)
        # self.cloth_rots_theta  = torch.zeros(instances_num, self.smpl_points_num, 3, 3, device=self.device)
        self.instances_fv = torch.zeros(frame_num, instances_num, dtype=torch.bool, device=self.device)
        self.hinge_pt_ids = torch.zeros(instances_num*self.ins_num, dtype=torch.bool, device=self.device)
        self.reshape_means_ind = torch.zeros(instances_num*self.ins_num, dtype=torch.long, device=self.device)
        #############################
        self.template = SMPLTemplate(
            smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",
            num_human=instances_num,
            init_beta=torch.zeros(instances_num, 10, device=self.device),
            cano_pose_type="da_pose",
            use_voxel_deformer=self.use_voxel_deformer,
            is_resume=True
        ).to(self.device)
        # if self.use_voxel_deformer:
        #     self.template.voxel_deformer.enable_voxel_correction()
        ####################
        point_ids      = dict['points_ids']
        smpl_point_ids = dict['smpl_point_ids']
        instances_fv   = dict['instances_fv']
        instances_size = dict['instances_size']
        smpl_pt_masks  = dict['smpl_pt_masks']
        keypt_masks    = dict['keypt_masks']
        msg = super().load_state_dict(dict, **kwargs)
        
        # msg = VanillaGaussians.load_state_dict(self, dict, **kwargs)
        self.smpl_betas     = dict['smpl_betas']
        self.reshape_means_ind = dict['reshape_means_ind']
        self.smpl_insts_quats = dict['smpl_insts_quats']
        # self.smpl_faces     = dict['smpl_faces']
        # self.cloth_rots_theta   = dict['cloth_rots_theta']
        self.point_ids      = point_ids
        self.smpl_point_ids = smpl_point_ids
        self.smpl_ids = torch.unique(smpl_point_ids)
        self.instances_fv   = instances_fv
        self.instances_size = instances_size
        self.smpl_pt_masks  = smpl_pt_masks
        self.keypt_masks    = keypt_masks
        self.instances_quats = torch.nn.Parameter(self.instances_quats.unsqueeze(2).contiguous())
        means_cur, sub_means_ind = [], []
        means_ind = torch.arange(self._means.shape[0], device=self.device, dtype=torch.int32)
        for i, ins_id in enumerate(self.smpl_ids):
            pc_mask = self.point_ids[:,0] == ins_id
            sub_means = self._means[pc_mask]
            sub_means_ind.append(means_ind[pc_mask])
            means_cur.append(sub_means.unsqueeze(0).contiguous())
        means_cur = torch.cat(means_cur, dim=0)
        sub_means_ind = torch.cat(sub_means_ind, dim=0)
        self.reshape_means = means_cur
        self.reshape_means_ind = sub_means_ind
        self.update_knn(self.smpl_original)
        self.train_mode = False
        return msg
    
    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = super().collect_gaussians_from_ids(ids)
        # collect embeddings
        for id in ids:
            instance_embedding = self.instances_embedding[id]
            gaussian_dict[id]["embedding"] = instance_embedding
        return gaussian_dict

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        replace instances from the model
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            self._means = Parameter(torch.cat([self._means, new_gaussian["_means"]], dim=0))
            self._scales = Parameter(torch.cat([self._scales, new_gaussian["_scales"]], dim=0))
            self._quats = Parameter(torch.cat([self._quats, new_gaussian["_quats"]], dim=0))
            self._features_dc = Parameter(torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0)) #[N，3]
            self._features_rest = Parameter(torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0)) # [N, 15, 3]
            self._opacities = Parameter(torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0))
            # keeps original point ids
            self.point_ids = torch.cat([self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)], dim=0)
            self.point_types = torch.cat([self.point_types, new_gaussian["point_types"]], dim=0)
            # replace embeddings
            # NOTE: modify data in nn.Parameter directly
            self.instances_embedding.data[ins_id] = new_gaussian["embedding"]

    def export_gaussians_to_ply(
        self, alpha_thresh: float, instance_id: List[int] = None, specific_frame: int = 0,
    ) -> Dict[str, torch.Tensor]:
        self.cur_frame = specific_frame
        pts_mask = self.point_ids[..., 0] == instance_id

        if self.ctrl_cfg.use_deformgs_for_nonrigid and self.step > self.ctrl_cfg.use_deformgs_after:
            delta_xyz, _, _ = self.get_deformation(local_means=self._means)
            means = self._means + delta_xyz
        else:
            means = self._means
        means = means[pts_mask]
        direct_color = self.colors[pts_mask]
        
        activated_opacities = self.get_opacity[pts_mask]
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }
