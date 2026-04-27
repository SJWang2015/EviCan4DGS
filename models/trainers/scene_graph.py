from typing import Dict, List
import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere
from models.gaussians.basics import *
from utils.o3d_vis import save_gaussians_as_ply

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
        if "CPDDeformableNodes" in self.model_config:
            self.gaussian_classes["CPDDeformableNodes"] = GSModelType.CPDDeformableNodes
        if "CPDSMPLDeformableNodes" in self.model_config:
            self.gaussian_classes["CPDSMPLDeformableNodes"] = GSModelType.CPDSMPLDeformableNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict, cpddeformnode_pts_dict, smpl_cpddeformnode_pts_dict = {}, {}, {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )
        
        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config and "CPDDeformableNodes" not in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )

        if "CPDDeformableNodes" in self.model_config:
            _pts_dict = dataset.get_init_objects(
                cur_node_type='CPDDeformableNodes',        
                # exclude_smpl="SMPLNodes" in self.model_config,
                exclude_smpl= False,
                **self.model_config["CPDDeformableNodes"]["init"]
            )

            if "CPDSMPLDeformableNodes" in self.model_config:
                smpl_cpddeformnode_pts_dict = {k: v for k, v in _pts_dict.items() if v["smpl_mask"]}
                cpddeformnode_pts_dict      = {k: v for k, v in _pts_dict.items() if not v["smpl_mask"]}
                # smplnode_pts_dict           = {k: v for k, v in _pts_dict.items() if not v["smpl_mask"] and not v['cpddeformable_mask']}
                # cpddeformnode_pts_dict      = {k: v for k, v in _pts_dict.items() if not v["smpl_mask"] and v['cpddeformable_mask']}
            else:
                cpddeformnode_pts_dict = _pts_dict
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict, **cpddeformnode_pts_dict, **smpl_cpddeformnode_pts_dict}
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    if init_cfg.from_lidar.get("return_flow", None) is None:
                        sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device)
                    else:
                        sampled_pts, sampled_color, sampled_flow, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device)
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )
                
                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )
                
            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
                
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )

            if class_name == 'CPDDeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=cpddeformnode_pts_dict
                )

            if class_name == 'CPDSMPLDeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smpl_cpddeformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")
                
        logger.info(f"Initialized gaussians from pcd")
    
    # def forward(
    #     self, 
    #     image_infos: Dict[str, torch.Tensor],
    #     camera_infos: Dict[str, torch.Tensor],
    #     src_pcs: torch.Tensor = None,
    #     tgt_pcs: torch.Tensor = None,
    #     flow_classes: torch.Tensor = None,
    #     novel_view: bool = False
    # ) -> Dict[str, torch.Tensor]:
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        # src_pcs = None,
        # tgt_pcs = None,
        # cam_id = None,
        # processed_cam_list = None,
        # human_masks_list = None,
        novel_view = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        # if processed_cam_list is not None:
        #     processed_cam = processed_cam_list[cam_id]
        # else:
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        

        # human_masks = image_infos['human_masks']

        gs_render_fn = lambda gs_cls, cam: self.render_gaussians(
                gs=gs_cls,
                cam=cam,
                near_plane=self.render_cfg.near_plane,
                far_plane=self.render_cfg.far_plane,
                render_mode="RGB+ED",
                radius_clip=self.render_cfg.get('radius_clip', 0.)
            )
        
        gs = self.collect_gaussians(
            cam=processed_cam,
            # image_ids=image_infos["img_idx"].flatten()[0],
            # src_pcs=src_pcs,
            # tgt_pcs=tgt_pcs,
            # processed_cam_list = processed_cam_list,
            # human_masks_list = human_masks_list,
            # render_fn=gs_render_fn
        )
        # gs = self.collect_gaussians(
        #     cam=processed_cam,
        #     image_ids=image_infos_list["img_idx"].flatten()[0],
        #     src_pcs=src_pcs,
        #     tgt_pcs=tgt_pcs,
        #     flow_classes=flow_classes
        # )
        # gs_dict = {
        #     "_means": [],
        #     "_scales": [],
        #     "_quats": [],
        #     "_rgbs": [],
        #     "_opacities": [],
        #     "class_labels": [],
        # }
        # ply_name = f"output_{self.cur_frame}.ply"
        # save_gaussians_as_ply(
        #     ply_name,
        #     gs._means.detach().cpu().numpy(),
        #     gs._quats.detach().cpu().numpy(),
        #     gs._scales.detach().cpu().numpy(),
        #     gs._opacities.detach().cpu().numpy(),
        #     gs._rgbs.detach().cpu().numpy()
        #     )

        smpl_outputs, outputs, smpl_render_fn = None, None, None
        if gs.extras is not None:
            cpd_gs = gs.extras["cpd_gs"]
            smpl_gs = gs.extras["smpl_gs"]
            cpd_gs_dict = {
                "_means": [],
                "_scales": [],
                "_quats": [],
                "_rgbs": [],
                "_opacities": [],
                "class_labels": [],
            }
            cpd_gs_dict["_means"] = torch.cat([gs.means, cpd_gs["_means"]], dim=0)
            cpd_gs_dict["_scales"] = torch.cat([gs.scales, cpd_gs["_scales"]], dim=0)
            cpd_gs_dict["_quats"] = torch.cat([gs.quats, cpd_gs["_quats"]], dim=0)
            cpd_gs_dict["_rgbs"] = torch.cat([gs.rgbs, cpd_gs["_rgbs"]], dim=0)
            cpd_gs_dict["_opacities"] = torch.cat([gs.opacities, cpd_gs["_opacities"]], dim=0)


            comp_smpl_pts_lable = torch.full((smpl_gs["_means"].shape[0],), self.gaussian_classes["CPDSMPLDeformableNodes"], device=self.device)
            smpl_pts_labels = torch.cat([self.pts_labels, comp_smpl_pts_lable], dim=0)
            comp_pts_lable = torch.full((cpd_gs["_means"].shape[0],), self.gaussian_classes["CPDSMPLDeformableNodes"], device=self.device)
            self.pts_labels = torch.cat([self.pts_labels, comp_pts_lable], dim=0)
            if self.render_dynamic_mask:
                self.dynamic_pts_mask = (self.pts_labels != 0).float()
                dynamic_smpl_pts_mask = (smpl_pts_labels != 0).float()
            
            smpl_gs_dict = {
                "_means": [],
                "_scales": [],
                "_quats": [],
                "_rgbs": [],
                "_opacities": [],
                "class_labels": [],
            }
            smpl_gs_dict["_means"] = torch.cat([gs.means, smpl_gs["_means"]], dim=0)
            smpl_gs_dict["_scales"] = torch.cat([gs.scales, smpl_gs["_scales"]], dim=0)
            smpl_gs_dict["_quats"] = torch.cat([gs.quats, smpl_gs["_quats"]], dim=0)
            smpl_gs_dict["_rgbs"] = torch.cat([gs.rgbs, smpl_gs["_rgbs"]], dim=0)
            smpl_gs_dict["_opacities"] = torch.cat([gs.opacities, smpl_gs["_opacities"]], dim=0)

            cpd_gs = dataclass_gs(
                _means=cpd_gs_dict["_means"],
                _scales=cpd_gs_dict["_scales"],
                _quats=cpd_gs_dict["_quats"],
                _rgbs=cpd_gs_dict["_rgbs"],
                _opacities=cpd_gs_dict["_opacities"],
                detach_keys=[],    # if "means" in detach_keys, then the means will be detached
                extras=None# to save some extra information (TODO) more flexible way
            )
            
            smpl_gs = dataclass_gs(
                _means=smpl_gs_dict["_means"],
                _scales=smpl_gs_dict["_scales"],
                _quats=smpl_gs_dict["_quats"],
                _rgbs=smpl_gs_dict["_rgbs"],
                _opacities=smpl_gs_dict["_opacities"],
                detach_keys=[],    # if "means" in detach_keys, then the means will be detached
                extras=None# to save some extra information (TODO) more flexible way
            )
            # smpl_outputs, smpl_render_fn = gs_render_fn(smpl_gs, processed_cam)
            outputs, render_fn = gs_render_fn(cpd_gs, processed_cam)
        else:
            outputs, render_fn = gs_render_fn(gs, processed_cam)
        # render gaussians
        # outputs, render_fn = self.render_gaussians(
        #     gs=gs,
        #     cam=processed_cam,
        #     near_plane=self.render_cfg.near_plane,
        #     far_plane=self.render_cfg.far_plane,
        #     render_mode="RGB+ED",
        #     radius_clip=self.render_cfg.get('radius_clip', 0.)
        # )
        # outputs, render_fn = gs_render_fn(gs, processed_cam)
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )

        if smpl_outputs is not None:
            smpl_outputs["rgb_sky"] = sky_model(image_infos)
            smpl_outputs["rgb_sky_blend"] = smpl_outputs["rgb_sky"] * (1.0 - smpl_outputs["opacity"])
            
            # affine transformation
            smpl_outputs["rgb"] = self.affine_transformation(
                smpl_outputs["rgb_gaussians"] + smpl_outputs["rgb_sky"] * (1.0 - smpl_outputs["opacity"]), image_infos
            )
        
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth

                if smpl_outputs is not None:
                    smpl_gaussian_mask = smpl_pts_labels != self.gaussian_classes["Background"]
                    smpl_sep_rgb, smpl_sep_depth, smpl_sep_opacity = smpl_render_fn(smpl_gaussian_mask)
                    smpl_outputs["Dynamic_rgb"] = self.affine_transformation(smpl_sep_rgb, image_infos)
                    smpl_outputs["Dynamic_opacity"] = smpl_sep_opacity
                    smpl_outputs["Dynamic_depth"] = smpl_sep_depth

        if smpl_outputs is not None:        
            new_smpl_outputs = {f"smpl_{k}": v for k, v in smpl_outputs.items()}
            outputs = {**outputs, **new_smpl_outputs}
        
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict