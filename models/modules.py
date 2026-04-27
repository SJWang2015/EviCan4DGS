import torch
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr
from utils.geometry import rotation_6d_to_matrix
from utils.lib import pointnet2_utils as pointutils
# from torch_scatter import scatter_add
import numpy as np
from pytorch3d.ops import knn_points, knn_gather

logger = logging.getLogger()

class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError

class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded

class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x
    
class SkyModel(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        head_mlp_layer_width: int = 64,
        enable_appearance_embedding: bool = True,
        appearance_embedding_dim: int = 16,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=6
        )
        self.direction_encoding.requires_grad_(False)
        
        self.enable_appearance_embedding = enable_appearance_embedding
        if self.enable_appearance_embedding:
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = nn.Embedding(n, appearance_embedding_dim, dtype=torch.float32)
            
        in_dims = self.direction_encoding.n_output_dims + appearance_embedding_dim \
            if self.enable_appearance_embedding else self.direction_encoding.n_output_dims
        self.sky_head = MLP(
            in_dims=in_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )
        self.in_test_set = False
    
    def forward(self, image_infos):
        directions = image_infos["viewdirs"]
        self.device = directions.device
        prefix = directions.shape[:-1]
        
        dd = self.direction_encoding(directions.reshape(-1, 3)).to(self.device)
        if self.enable_appearance_embedding:
            # optionally add appearance embedding
            if "img_idx" in image_infos and not self.in_test_set:
                appearance_embedding = self.appearance_embedding(image_infos["img_idx"]).reshape(-1, self.appearance_embedding_dim)
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*dd.shape[:-1], self.appearance_embedding_dim),
                    device=dd.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
        rgb_sky = self.sky_head(dd).to(self.device)
        rgb_sky = F.sigmoid(rgb_sky)
        return rgb_sky.reshape(prefix + (3,))
    
    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class EnvLight(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
    def forward(self, image_infos):
        l = image_infos["viewdirs"]
        
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class AffineTransform(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        embedding_dim: int = 4,
        pixel_affine: bool = False,
        base_mlp_layer_width: int = 64,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.embedding_dim = embedding_dim
        self.pixel_affine = pixel_affine
        self.embedding = nn.Embedding(n, embedding_dim, dtype=torch.float32)
        
        input_dim = (embedding_dim + 2)if self.pixel_affine else embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 12),
        )
        self.in_test_set = False
        
        self.zero_init()
        
    def zero_init(self):
        torch.nn.init.zeros_(self.embedding.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, image_infos):
        if "img_idx" in image_infos and not self.in_test_set:
            embedding = self.embedding(image_infos["img_idx"])
        else:
            # use mean appearance embedding
            embedding = torch.ones(
                (*image_infos["viewdirs"].shape[:-1], self.embedding_dim),
                device=image_infos["viewdirs"].device,
            ) * self.embedding.weight.mean(dim=0)
        if self.pixel_affine:
            embedding = torch.cat([embedding, image_infos["pixel_coords"]], dim=-1)
        affine = self.decoder(embedding)
        affine = affine.reshape(*embedding.shape[:-1], 3, 4)
        
        affine[..., :3, :3] = affine[..., :3, :3] + torch.eye(3, device=affine.device).reshape(1, 3, 3)
        return affine

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(
        self,
        class_name: str,
        n: int,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        
        self.zero_init() # important for initialization !!

    # def zero_init(self):
    #     torch.nn.init.zeros_(self.embeds.weight)
    def zero_init(self):
        torch.nn.init.zeros_(self.embedding.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.decoder[2].weight)
        torch.nn.init.zeros_(self.decoder[2].bias)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, x_multires=10, t_multires=10):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.x_multires = x_multires
        self.t_multires = t_multires
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(self.x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
    
    
class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10, 
                 deform_quat=True, deform_scale=True):
        super(ConditionalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale: 
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
    

class NormalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10):
        super(NormalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 1)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz_norm = self.gaussian_warp(h)
        return d_xyz_norm


class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
        is_resume=False
    ) -> None:
        super().__init__()
        # vtx B,N,3, vtx_features: B,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y
        B = vtx.shape[0]
        assert vtx.shape[0] == vtx_features.shape[0], "Batch size mismatch"

        # * Prepare Grid
        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )
        grid = grid.expand(B, -1, -1)

        gt_bbox_min = (vtx.min(dim=1).values).to(device)
        gt_bbox_max = (vtx.max(dim=1).values).to(device)
        offset = (gt_bbox_min + gt_bbox_max) * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (
            (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * self.global_scale
        ).unsqueeze(-1) #类似于一个半径

        corner = torch.ones_like(offset) * scale
        corner[:, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(-1, 1, 3)  
        max_vert = (offset + corner).reshape(-1, 1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=1)

        self.register_buffer("scale", scale.unsqueeze(1)) # [B, 1, 1]
        self.register_buffer("offset", offset.unsqueeze(1)) # [B, 1, 3]

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        if not is_resume:
            weights = (
                self._query_weights_smpl(
                    grid_denorm,
                    smpl_verts=vtx.detach().clone(),
                    smpl_weights=vtx_features.detach().clone(),
                )
                .detach()
                .clone()
            )  #b,c,d,h,w
        else:
            # random initialization
            weights = torch.randn(
                B, vtx_features.shape[-1], *resolution_dhw
            ).to(device)

        self.register_buffer("lbs_voxel_base", weights.detach())
        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]

        # # debug
        # import numpy as np
        # np.savetxt("./debug/dbg.xyz", grid_denorm[0].detach().cpu())
        # np.savetxt("./debug/vtx.xyz", vtx[0].detach().cpu())
        return

    def enable_voxel_correction(self):
        voxel_w_correction = torch.zeros_like(self.lbs_voxel_base)
        self.voxel_w_correction = nn.Parameter(voxel_w_correction)

    def enable_additional_correction(self, additional_channels, std=1e-4):
        additional_correction = (
            torch.ones(
                self.lbs_voxel_base.shape[0],
                additional_channels,
                *self.lbs_voxel_base.shape[2:]
            )
            * std
        )
        self.additional_correction = nn.Parameter(additional_correction)

    @property
    def get_voxel_weight(self):
        w = self.lbs_voxel_base
        if hasattr(self, "voxel_w_correction"):
            w = w + self.voxel_w_correction
        if hasattr(self, "additional_correction"):
            w = torch.cat([w, self.additional_correction], dim=1)
        return w

    def get_tv(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).mean()
        return (tv_x + tv_y + tv_z) / 3.0

    def get_mag(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        return torch.norm(d, dim=1).mean()

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        # xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight, #[23,24,16,64,64]
            self.normalize(xc)[:, :, None, None], #[23,6890,1,1,3]
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(3, 4).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        dist, idx, _ = knn_points(x, smpl_verts.detach(), K=30) # [B, N, 30]
        dist = dist.sqrt().clamp_(0.0001, 1.0)
        expanded_smpl_weights = smpl_weights.unsqueeze(2).expand(-1, -1, idx.shape[2], -1) # [B, N, 30, J]
        weights = expanded_smpl_weights.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, expanded_smpl_weights.shape[-1])) # [B, N, 30, J]

        ws = 1.0 / dist
        ws = ws / ws.sum(-1, keepdim=True)
        weights = (ws[..., None] * weights).sum(-2)

        b = x.shape[0]
        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w) # 23, 24, 16, 64, 64, 64
        for _ in range(30):
            mean = (
                weights[:, :, 2:, 1:-1, 1:-1]
                + weights[:, :, :-2, 1:-1, 1:-1]
                + weights[:, :, 1:-1, 2:, 1:-1]
                + weights[:, :, 1:-1, :-2, 1:-1]
                + weights[:, :, 1:-1, 1:-1, 2:]
                + weights[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
            weights[:, :, 1:-1, 1:-1, 1:-1] = (
                weights[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * 0.7 + mean
            sums = weights.sum(1, keepdim=True)
            weights = weights / sums
        return weights.detach() #[23,24,16,64,64]
 
class ConditionalModulationNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10, 
                 deform_quat=True, deform_scale=True):
        super(ConditionalModulationNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, residuals, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz = self.gaussian_warp(h)
        # d_xyz = x - residuals * lamda_xyz
        # d_xyz = self.gaussian_warp(h) 
        scaling, rotation = None, None
        if self.deform_scale: 
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
    

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.freqs = 2 ** torch.arange(num_freqs).float() * torch.pi

    def forward(self, t):
        enc = [t]
        for f in self.freqs:
            enc.append(torch.sin(f * t))
            enc.append(torch.cos(f * t))
        return torch.cat(enc, dim=-1)
    

# ===============================================================
# Mixed Factorized + Joint Encoder for 3DGS point features
# ===============================================================
class ISAB(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads, num_inducing):
        super().__init__()
        self.num_inducing = num_inducing
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, dim_input))

        self.mha1 = nn.MultiheadAttention(dim_input, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(dim_input, num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.GELU(),
            nn.Linear(dim_output, dim_output)
        )
        self.norm1 = nn.LayerNorm(dim_output)
        self.norm2 = nn.LayerNorm(dim_output)

    def forward(self, X, mask=None):
        # X: [B, N, D]
        B = X.shape[0]
        Z = self.inducing_points.unsqueeze(0).expand(B, -1, -1)  # [B, m, D]

        H, _ = self.mha1(query=Z, key=X, value=X, key_padding_mask=~mask if mask is not None else None)
        Y, _ = self.mha2(query=X, key=H, value=H)

        # Residual Connection + MLP
        out = self.norm1(X + Y)
        out = self.norm2(out + self.fc(out))
        return out  # [B, N, D]
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, dim))
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, X, mask=None):
        # X: [B, N, D]
        B = X.size(0)
        S = self.seed_vectors.unsqueeze(0).expand(B, -1, -1)  # [B, k, D]
        H, _ = self.mha(query=S, key=X, value=X, key_padding_mask=~mask if mask is not None else None)
        out = self.norm1(S + H)
        out = self.norm2(out + self.fc(out))
        return out  # [B, k, D]

class FactorizedJointEncoder(nn.Module):
    def __init__(self, geo_dim, rad_dim, latent_dim=128):
        super().__init__()

        # --- Geometry branch（position + covriance）---
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # # --- Radiance branch（color + opacity）---
        # self.rad_encoder = nn.Sequential(
        #     nn.Linear(rad_dim, 64),
        #     nn.GELU(),
        #     nn.Linear(64, 64),
        #     nn.GELU(),
        #     nn.LayerNorm(64)
        # )

        # # --- Feature Fusion ---
        # fused_in_dim = 128 + 64 
        # self.joint_fusion = nn.Sequential(
        #     nn.Linear(fused_in_dim, latent_dim),
        #     nn.GELU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.LayerNorm(latent_dim)
        # )

    def quat_to_rotmat(self, q):
        # q normalized [B,N,4]
        q = F.normalize(q, dim=-1)
        qw, qx, qy, qz = q.unbind(-1)
        R = torch.stack([
            torch.stack([1-2*(qy**2+qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], dim=-1),
            torch.stack([2*(qx*qy + qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz - qx*qw)], dim=-1),
            torch.stack([2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx**2+qy**2)], dim=-1)
        ], dim=-2)
        return R

    def quat_scale_to_sigma(self, quat, scale):
        R = self.quat_to_rotmat(quat)
        D = torch.diag_embed(scale**2)
        Sigma = R @ D @ R.transpose(-1, -2)
        return Sigma

    def forward(self, mu_A, quat_A, scale_A, color_opacity_A):
        '''
        mu_A:      [num_instances, num_pts, 3]
        quat_A:    [num_instances, num_pts, 4]
        scale_A:   [num_instances, num_pts, 3]
        color_A:   [num_instances, num_pts, 3]
        opacity_A: [num_instances, num_pts, 1]        
        '''
        # Sigma = self.quat_scale_to_sigma(quat_A, scale_A)
        # xx, yy, zz, xy, xz, yz = Sigma[...,0,0], Sigma[...,1,1], Sigma[...,2,2], Sigma[...,0,1], Sigma[...,0,2], Sigma[...,1,2]
        # cov6 = torch.stack([xx, yy, zz, xy, xz, yz], dim=-1)
        # geom_attr = torch.cat([mu_A, cov6], dim=-1)     # [B,N,9]
        geom_attr = torch.cat([mu_A, quat_A, scale_A], dim=-1)     # [B,N,10]
        # rad_attr  = torch.cat([color_A, opacity_A], dim=-1)  # [B,N,4]
    
        joint_feat = self.geo_encoder(geom_attr)
        # rad_feat = self.rad_encoder(color_opacity_A)

        # fused = torch.cat([geo_feat, rad_feat], dim=-1)

        # # # Fusion
        # joint_feat = self.joint_fusion(fused)
        # return joint_feat, Sigma  
        return joint_feat
    
class ScaleEmbedding(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, s):
        return self.fc(torch.log1p(s)) 

def gmm_responsibility(B_points, A_mu, A_Sigma, k=3, eps=1e-8):
        """
        B_points: [B, N, 3]
        A_mu: [B, M, 3]
        A_Sigma: [B, M, 3, 3]
        Returns:
            values: [B, N, k]
            indices: [B, N, k]
        """
        B, N, C = B_points.shape
        _, M, _ = A_mu.shape

        # 1. KNN: for each B_points[b, n], find K nearest in A_mu[b]
        # Compute squared distances: [B, N, M]
        # B_points_exp = B_points.unsqueeze(2)           # [B, N, 1, 3]
        # A_mu_exp = A_mu.unsqueeze(1)                   # [B, 1, M, 3]
        # sq_dists = ((B_points_exp - A_mu_exp) ** 2).sum(dim=3)  # [B, N, M]
        # knn_dists, knn_indices = torch.topk(sq_dists, k, dim=2, largest=False, sorted=True)  # [B, N, k]
        _, knn_indices, _ = knn_points(B_points, A_mu, K=5)

        # 2. Gather the K nearest A_mu and corresponding A_Sigma
        # Gathered A_mu_knn: [B, N, k, 3]
        # batch_indices = torch.arange(B, device=B_points.device).view(B, 1, 1).expand(-1, N, k)
        A_mu_knn = knn_gather(A_mu, knn_indices)
        A_sigma_flatten = A_Sigma.view(B, M, -1)
        A_Sigma_flatten_knn = knn_gather(A_sigma_flatten, knn_indices)
        A_Sigma_knn = A_Sigma_flatten_knn.reshape(B, N, -1, 3, 3)

        # 3. For each B_points, compute diff to KNN means
        B_points_expanded = B_points.unsqueeze(2)  # [B, N, 1, 3]
        diff = B_points_expanded - A_mu_knn        # [B, N, k, 3]

        # 4. Mahalanobis
        # For each [B, N, k], compute Mahalanobis distance
        # Sigma_inv: [B, N, k, 3, 3]
        Sigma_inv = torch.linalg.inv(A_Sigma_knn)  # [B, N, k, 3, 3]
        det_S = torch.linalg.det(A_Sigma_knn).clamp(min=eps)  # [B, N, k]
        norm_const = (1.0 / ((2 * torch.pi) ** (C / 2) * torch.sqrt(det_S)))  # [B, N, k]

        # Mahalanobis: einsum over last dim
        mahal = torch.einsum('bnkc,bnkcd,bnkd->bnk', diff, Sigma_inv, diff)  # [B, N, k]

        log_lk = torch.log(norm_const + eps) - 0.5 * mahal  # [B, N, k]
        probs = torch.exp(log_lk - log_lk.max(dim=2, keepdim=True)[0])  # [B, N, k]
        responsibilities = probs / (probs.sum(dim=2, keepdim=True) + eps)  # [B, N, k]

        # values/indices: already k neighbors, so just sort if needed
        # sorted_responsibilities, sort_indices = responsibilities.sort(dim=2, descending=True)
        # sorted_knn_indices = torch.gather(knn_indices, 2, sort_indices)
        values, indices = torch.topk(responsibilities, k, dim=-1, largest=True, sorted=True)

        return values, indices


class Conditional3DGSNetwork(nn.Module):
    def __init__(self, D=8, W=256, embed_dim=16, x_multires=10, t_multires=10):
        super().__init__()
        self.skips = [D // 2]
        self.embed_dim = embed_dim
        # self.scale_embed = ScaleEmbedding(embed_dim=scale_cond_dim)
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        # self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        # self.embed_diff_fn, diff_input_ch = get_embedder(x_multires, 3)
        in_dim = time_input_ch + 3 + embed_dim + 3 # xB, t, scaleCond, deta_μ
 
        
        # === Feature Extraction from SMPL-based 3DGS ===
        hidden_dim = W//2
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.backbone2 = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, W),
            nn.GELU(),
            nn.LayerNorm(W),
            nn.Linear(W, W),
            nn.GELU(),
            nn.LayerNorm(W),
            nn.Linear(W, 3),
            nn.LayerNorm(3),
        )
        

        # === Branch-wise Property  ===
        self.head_mu = nn.Linear(W, 3)
  
    
    # def forward(self, x_B, t_B, mu_A, scale_A, Sigma_A, feats_A, condition):
    def forward(self, x_B, t_B, pos_diff, condition):
        """
            Input:
                x_B: (B,N,3)
                t_B: (B,N,1)
                diff: (B,N,3)
                condition: (B,N,C), L >= N && L >= M
            Output:
                Δμ_B
        """
        t_emb = self.embed_time_fn(t_B)
        # x_emb = self.embed_fn(x_B)
        # pos_diff_emb = self.embed_diff_fn(pos_diff)
        feat = torch.cat([x_B, pos_diff, t_emb, condition], dim=-1)
        new_feat = self.backbone(feat)
        # normalize before concat
        feat = torch.cat([feat, new_feat], dim=-1)
        feat = self.backbone2(feat)
        delta_mu = F.sigmoid(feat)
        return delta_mu


if __name__ == "__main__":
    # B, N = 2,1000000
    # pts = torch.rand( N, 3).cuda()  # 假设点云
    # net = LocalVoxelLayer(input_ch=3, embed_dim=1,voxel_size=4, res=0.02).cuda()
    # t = 0.2 * torch.ones((N, 1), device="cuda")   # (B,N,1)
    # out = net(pts, t,t)
    # print(out[0].shape)  # (B,N,5,5,5,16)
    import os, glob
    from utils.o3d_vis import visualize_point_cloud
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "/home/ubuntu/Repositories/drivestudio/01/"
    
    # scene_rot_files = os.listdir(os.path.join(data_path,'flow/rot/'))
    data_files = glob.glob(os.path.join(data_path, '*.npz'))
    data_files.sort()

    pattern = r"/(\d+)_(\d+)\.npz$"

    device = "cuda:0"
    smpl_vtx, smpl_means, smpl_means_cur = [], [], [] 
    # np.savez("./01/"+f"{ins_id:03d}.npz",smpl_means=smpl_verts_w2o2.detach().cpu().numpy(), cur_smpl=new_sub_means.detach().cpu().numpy(), smpl_verts=self.smpl_vertices[i].detach().cpu().numpy())
    for ins_i, _ in enumerate(data_files):
        item = np.load(data_files[ins_i], allow_pickle=True)
        smpl_vtx.append(torch.from_numpy(item["smpl_verts"]).unsqueeze(0).to(device))
        smpl_means.append(torch.from_numpy(item["smpl_means"]).unsqueeze(0).to(device))
        smpl_means_cur.append(torch.from_numpy(item["cur_smpl"]).unsqueeze(0).to(device))

    
    smpl_means  = torch.cat(smpl_means, dim=0) 
    B, N, C = smpl_means.shape
    smpl_quats  = torch.rand((B, N, 4), device=device)
    smpl_scale  = torch.rand((B, N, 3), device=device)
    smpl_colors = torch.rand((B, N, 3), device=device)
    smpl_alpha  = torch.rand((B, N, 1), device=device)
  
    # smpl_encoder = FactorizedJointEncoder(geo_dim=9, rad_dim=4, latent_dim=128).to(device)
    deform_nn = Conditional3DGSNetwork(D=8, W=256, embed_dim=16, x_multires=10, t_multires=10).to(device)

    # smpl_feats, smpl_sigma = smpl_encoder(smpl_means, smpl_quats, smpl_scale, smpl_colors, smpl_alpha)

    total_num = 7000

    for i, item in enumerate(smpl_means_cur):
        if item[0].shape[0] > total_num:
            sampled_idx = (torch.randperm(item[0].shape[0])[:total_num]).to(device)
            smpl_means_cur[i] = (item[0][sampled_idx]).unsqueeze(0)
        else:
            res_num = total_num - item[0].shape[0]
            sampled_idx = (torch.randperm(6890)[:res_num]).to(device)
            res_points = smpl_means[i][sampled_idx]
            smpl_means_cur[i] = torch.cat([item[0], res_points], dim=0).unsqueeze(0)
    smpl_means_cur  = torch.cat(smpl_means_cur, dim=0) 

    init_embedding = torch.rand(B, total_num, 16, device=device)
    for i, item in enumerate(smpl_means_cur):
        # item = torch.from_numpy(item).to(device)  
        t = torch.tensor(i).unsqueeze(0).repeat(B, item.shape[0], 1).to(device)
        # detal_xyz, detal_quats, detal_scale, detal_colors, detal_alpha = deform_nn(item, t, smpl_means[i], smpl_scale[i], smpl_sigma[i], smpl_feats[i])
        detal_xyz = deform_nn(smpl_means_cur, t, smpl_means_cur, init_embedding)
        print(detal_xyz.shape)

  
