#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import trimesh
from utils.vis_utils import save_points
from scene.appearance_network import AppearanceNetwork
import open3d as o3d
from scipy.ndimage import gaussian_filter

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        #plane_gaussian_need
        self.plane_mask = torch.empty(0)
         
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.iso = 2.0
        self.setup_functions()
        # appearance network and appearance embedding
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(6000, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    def get_scaling_with_3D_filter_and_plane(self,iso):
        scales = self.get_scaling#n * 3
        #find plane gaussian
        sorted_scales,sort_indexs = torch.sort(scales,dim=1)
        bs = sorted_scales[...,1] / sorted_scales[...,0]
        plane_mask = (bs > iso)
        
        # mask = torch.ones_like(scales)
        index = torch.repeat_interleave(torch.arange(0,3,device="cuda").reshape(1,3),dim=0,repeats=torch.sum(plane_mask))
        small_dire_repeat = torch.repeat_interleave(sort_indexs[plane_mask][...,0].reshape(-1,1),dim=-1,repeats=3)
        small_direct_mask = (index != small_dire_repeat)
        
        scales = torch.square(scales) + torch.square(self.filter_3D) 
         
        scales[plane_mask] = scales[plane_mask] * small_direct_mask
        
        scales = torch.sqrt(scales)
        return scales
    
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    #camera转换到gaussian坐标系的矩阵
    def get_view2gaussian(self, viewmatrix):
        r = self._rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]
        
        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
        rots = R
        xyz = self.get_xyz
        N = xyz.shape[0] 
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0
        
        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W#高斯坐标向camera坐标转换
        
        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]
        
        #相当于求逆
        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0
        
        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()
        return V2G

    @torch.no_grad()
    def compute_3D_filter(self, cameras):#mip-spalating
        # print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"},
            {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    #save_ply in origin_extend
    def save_ply_origin(self,path,bbox,block_index,x_block_num,z_block_num):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filter_3D = self.filter_3D.detach().cpu().numpy()
        
        x_index = block_index % x_block_num
        z_index = block_index // x_block_num
        if x_index == 0:
            bbox[0] = -10000
        if x_index == (x_block_num - 1):
            bbox[2] = 10000
        if z_index == 0:
            bbox[1] = -10000
        if z_index == (z_block_num - 1):
            bbox[3] = 10000
        
        out_of_bbox_idx = np.where((xyz[:,0] < bbox[0]) | (xyz[:,2] < bbox[1]) | (xyz[:,0] > bbox[2]) | (xyz[:,2] > bbox[3]))[0]
        filter_xyz = np.delete(xyz, out_of_bbox_idx, axis=0)
        filter_normals = np.delete(normals, out_of_bbox_idx, axis=0)
        filter_f_dc = np.delete(f_dc, out_of_bbox_idx, axis=0)
        filter_f_rest = np.delete(f_rest, out_of_bbox_idx, axis=0)
        filter_opacities = np.delete(opacities, out_of_bbox_idx, axis=0)
        filter_scale = np.delete(scale, out_of_bbox_idx, axis=0)
        filter_rotation = np.delete(rotation, out_of_bbox_idx, axis=0)
        filter_filter_3D = np.delete(filter_3D, out_of_bbox_idx, axis=0)
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        filter_elements = np.empty(filter_xyz.shape[0], dtype=dtype_full)
        filter_attributes = np.concatenate((filter_xyz, filter_normals, filter_f_dc, filter_f_rest, filter_opacities, filter_scale, filter_rotation,filter_filter_3D), axis=1)
        filter_elements[:] = list(map(tuple, filter_attributes))
        filter_el = PlyElement.describe(filter_elements, 'vertex')
        PlyData([filter_el]).write(path)

    def save_fused_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities = self.inverse_opacity_activation(current_opacity_with_filter).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    @torch.no_grad()
    def get_tetra_points(self,extend=None,statistical_filter = True,plane_iso=2):
        M = trimesh.creation.box()#顶点坐标范围为-0.5-0.5
        M.vertices *= 2#顶点坐标范围为-1-1
        
        rots = build_rotation(self._rotation)#四元数转旋转矩阵
        xyz = self.get_xyz
        
        # scale = self.get_scaling_with_3D_filter_and_plane(iso=16) * 3. # TODO test
        if plane_iso == -1:
            print(-1)
            scale = self.get_scaling_with_3D_filter * 3. # TODO test
        else:
            scale = self.get_scaling_with_3D_filter_and_plane(iso=plane_iso) * 3. # TODO test
        
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)#n * 3 * 8
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)#n * 3 * 8
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()#n*8,3
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        #直通滤波:filter points not in block bbox
        if extend:
            direct_filtering_mask = ((vertices[...,0] > extend[0]) * (vertices[...,0] < extend[2]) * (vertices[...,2] > extend[1]) * (vertices[...,2] < extend[3]))
            vertices = vertices[direct_filtering_mask]
            vertices_scale = vertices_scale[direct_filtering_mask]
        #统计滤波去除离群点
        if statistical_filter:
            print("统计滤波......")
            
            vertices_numpy = vertices.cpu().numpy()
            pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(vertices_numpy))
            
            # pcd = pcd.remove_duplicated_points()
            
            std_ratio = 8
            nb_neighbors = 50
            filter_pcd,save_index= pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio,print_progress = True) 
            
            vertices_np = np.asarray(filter_pcd.points)
            
            vertices = torch.tensor(vertices_np,dtype=torch.float32).to(vertices.device)
            vertices_scale = vertices_scale[save_index]
            
        # o3d.io.write_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station2/block0/temp/gs_points.ply", filter_pcd)
        return vertices, vertices_scale
    def get_lod_tetra_points(self,extend=None,statistical_filter = True,plane_iso=2,levels=1,simplify = "uniform"):
        M = trimesh.creation.box()#顶点坐标范围为-0.5-0.5
        M.vertices *= 2#顶点坐标范围为-1-1
        
        rots = build_rotation(self._rotation)#四元数转旋转矩阵
        xyz = self.get_xyz
        
        # scale = self.get_scaling_with_3D_filter_and_plane(iso=16) * 3. # TODO test
        if plane_iso == -1:
            print(-1)
            scale = self.get_scaling_with_3D_filter * 3. # TODO test
        else:
            scale = self.get_scaling_with_3D_filter_and_plane(iso=plane_iso) * 3. # TODO test
        
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)#n * 3 * 8
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)#n * 3 * 8
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()#n*8,3
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        #直通滤波:filter points not in block bbox
        if extend:
            #对extend做10%的扩张
            xmin,zmin,xmax,zmax = extend[0],extend[1],extend[2],extend[3]
            extend = [xmin-0.1*(xmax-xmin),zmin-0.1*(zmax-zmin),xmax+0.1*(xmax-xmin),zmax+0.1*(zmax-zmin)]
            
            direct_filtering_mask = ((vertices[...,0] > extend[0]) * (vertices[...,0] < extend[2]) * (vertices[...,2] > extend[1]) * (vertices[...,2] < extend[3]))
            vertices = vertices[direct_filtering_mask]
            vertices_scale = vertices_scale[direct_filtering_mask]
        #统计滤波去除离群点
        if statistical_filter:
            print("统计滤波......")
            
            vertices_numpy = vertices.cpu().numpy()
            pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(vertices_numpy))
            
            # pcd = pcd.remove_duplicated_points()
            
            std_ratio = 8
            nb_neighbors = 50
            filter_pcd,save_index= pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio,print_progress = True) 
            
            # vertices_np = np.asarray(filter_pcd.points)
            # vertices = torch.tensor(vertices_np,dtype=torch.float32).to(vertices.device)
            vertices_scale = vertices_scale[save_index]
            
        '''
        点云简化
        '''
        #均匀滤波
        from tqdm import tqdm
        if simplify == "simplify":
            #Uniform sampling
            vertices_array = []
            vertices_scale_array = []
            def non_uniform_sampling(start, end, levels):
                # 使用指数曲线生成非均匀采样
                x = np.linspace(0, 1, levels)  # 从0到1线性分布
                samples = start + (end - start) * (x**2)  # 使用指数2进行加速
                
                return samples
            k_s = non_uniform_sampling(1,100,levels)
            for level in tqdm(range(levels)):
                # K = level + 1
                # K = 100
                K = int(k_s[level])
                uni_down_pcd = filter_pcd.uniform_down_sample(every_k_points=K)
                
                #TODO:保存pcd
                # o3d.io.write_point_cloud(f"/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod/block0/point_cloud/iteration_30000/lod/lod_{level}.ply", uni_down_pcd)
                
                level_index = range(0,len(vertices_scale),K)
                
                vertices_np = np.asarray(uni_down_pcd.points)
                vertices_uni = torch.tensor(vertices_np,dtype=torch.float32).to(vertices.device)
                vertices_array.append(vertices_uni)
                vertices_scale_array.append(vertices_scale[level_index])
        #体素滤波
        elif simplify == "voxel":
            vertices_array = []
            vertices_scale_array = []
            filter_points = np.asarray(filter_pcd.points)
            
            points_min = np.min(filter_points, axis=0)
            points_max = np.max(filter_points, axis=0)
            
            #计算包围盒的距离
            distance = np.sqrt(np.sum((points_max - points_min) ** 2))
            
            max_resolution = 8192
            min_resolution = 512
            
            def uniform_sampling(start, end, levels):
                # 使用指数曲线生成非均匀采样
                for j in range(levels):
                    if j == 0:
                        samples = [start]
                    else:
                        samples.append(start * (2 ** j))
                
                return samples

            resolutions = uniform_sampling(min_resolution, max_resolution, levels)
            voxel_sizes = [distance / resolution for resolution in resolutions]
            voxel_sizes.reverse()
            for level,voxel_size in enumerate(voxel_sizes):
               rt = filter_pcd.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound = points_min, max_bound = points_max)
               voxel_filter_pcd = rt[0]
               voxel_filter_scales = []
               vertices_scale_cpu = vertices_scale.cpu()
               
            #    o3d.io.write_point_cloud(f"/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod/block0/point_cloud/iteration_30000/lod/lod_{level}.ply", voxel_filter_pcd)
               
               for j,point_indexs in tqdm(enumerate(rt[2])):
                   if len(point_indexs) == 1:
                        voxel_filter_scales.append(vertices_scale_cpu[point_indexs[0]].item())
                   else:
                       temp_scales = []
                       for point_index in point_indexs:
                           temp_scales.append(vertices_scale_cpu[point_index].squeeze().item())
                       voxel_filter_scales.append(np.mean(temp_scales))
               vertices_np = np.asarray(voxel_filter_pcd.points)
               vertices_vox = torch.tensor(vertices_np,dtype=torch.float32).to(vertices.device)
               vertices_array.append(vertices_vox)
               
               vertices_scale_array.append(torch.tensor(voxel_filter_scales,dtype=torch.float32).to(vertices.device).unsqueeze(1))
        elif simplify ==  "curvature":#基于曲率的滤波
            vertices_array = []
            vertices_scale_array = []
            import copy
            def calculate_surface_curvature(pcd, radius=0.1, max_nn=30):
                pcd_n = copy.deepcopy(pcd)
                pcd_n.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(max_nn=max_nn,radius = radius))
                covs = np.asarray(pcd_n.covariances)
                vals, vecs = np.linalg.eig(covs)
                curvature = np.min(vals, axis=1)/np.sum(vals, axis=1)
                return curvature#n,1
            curvature = calculate_surface_curvature(filter_pcd,radius = 0.1)
            thsolds = []
            start_cur = 0.05
            end_cur = 0.25
            def non_uniform_sampling(start, end, levels):
                # 使用指数曲线生成非均匀采样
                x = np.linspace(0, 1, levels)  # 从0到1线性分布
                samples = start + (end - start) * (x**2)  # 使用指数2进行加速
                
                return samples
            # thsolds = non_uniform_sampling(start_cur, end_cur, levels)
            for i in range(levels):
                thsolds.append(start_cur + i * (end_cur - start_cur) / (levels - 1))
            for level,th in enumerate(thsolds):
                cur_mask = curvature > th
                cur_filter_pcd = np.asarray(filter_pcd.points)
                cur_filter_pcd = cur_filter_pcd[cur_mask]
                cur_filter_scale = vertices_scale[cur_mask]
                
                #o3d保存
                voxel_filter_pcd_o3d = o3d.geometry.PointCloud()
                voxel_filter_pcd_o3d.points = o3d.utility.Vector3dVector(cur_filter_pcd)
                o3d.io.write_point_cloud(f"/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod/block0/point_cloud/iteration_30000/lod/lod_{level}.ply", voxel_filter_pcd_o3d)
                
                vertices_cur = torch.tensor(cur_filter_pcd,dtype=torch.float32).to(vertices.device)
                vertices_array.append(vertices_cur)
                vertices_scale_array.append(cur_filter_scale)
        elif simplify == "feature_points":#特征点：体素结合曲率
            vertices_array = []
            vertices_scale_array = []
            #计算每个点的曲率权重
            import copy
            def calculate_surface_curvature(pcd, radius=0.1, max_nn=30):
                pcd_n = copy.deepcopy(pcd)
                pcd_n.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(max_nn=max_nn,radius = radius))
                covs = np.asarray(pcd_n.covariances)
                vals, vecs = np.linalg.eig(covs)
                curvature = np.min(vals, axis=1)/np.sum(vals, axis=1)
                return np.real(curvature)#n,1
            curvature = calculate_surface_curvature(filter_pcd,radius = 0.1)
            
            filter_points = np.asarray(filter_pcd.points)
            
            points_min = np.min(filter_points, axis=0)
            points_max = np.max(filter_points, axis=0)
            
            #计算包围盒的距离
            distance = np.sqrt(np.sum((points_max - points_min) ** 2))
            
            max_resolution = 8192
            min_resolution = 512
            
            def uniform_sampling(start, end, levels):
                # 使用指数曲线生成非均匀采样
                for j in range(levels):
                    if j == 0:
                        samples = [start]
                    else:
                        samples.append(start * (2 ** j))
                
                return samples

            resolutions = uniform_sampling(min_resolution, max_resolution, levels)
            voxel_sizes = [distance / resolution for resolution in resolutions]
            voxel_sizes.reverse()
            for level,voxel_size in enumerate(voxel_sizes):
               rt = filter_pcd.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound = points_min, max_bound = points_max)
               voxel_filter_pcd = rt[0]
               voxel_filter_scales = []
               vertices_scale_cpu = vertices_scale.cpu()
            #    o3d.io.write_point_cloud(f"/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod/block0/point_cloud/iteration_30000/lod/lod_{level}.ply", voxel_filter_pcd)
               voxel_filter_points = np.asarray(voxel_filter_pcd.points)
               for j,point_indexs in tqdm(enumerate(rt[2])):
                   if len(point_indexs) == 1:
                        voxel_filter_scales.append(vertices_scale_cpu[point_indexs[0]].item())
                   else:
                       temp_curs = []
                       for point_index in point_indexs:
                           one_cur = curvature[point_index]
                           temp_curs.append(one_cur)
                        #根据曲率计算权重
                       weights = np.array(temp_curs) / (np.sum(np.array(temp_curs)) + 1e-5)
                       temp_scales = []
                       temp_xyz = []
                       for point_index in point_indexs:
                           temp_scales.append(vertices_scale_cpu[point_index].squeeze().item())
                           temp_xyz.append(filter_points[point_index])
                        
                       voxel_filter_scales.append(np.sum(np.array(temp_scales)*weights))
                       voxel_filter_points[j] =np.sum(np.array(temp_xyz)*weights[:,None],axis = 0)

            #    vertices_np = np.asarray(voxel_filter_pcd.points)
               voxel_filter_pcd_o3d = o3d.geometry.PointCloud()
               voxel_filter_pcd_o3d.points = o3d.utility.Vector3dVector(voxel_filter_points)
               o3d.io.write_point_cloud(f"/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_lod/block0/point_cloud/iteration_30000/lod/lod_{level}.ply", voxel_filter_pcd)
            
            
               vertices_vox = torch.tensor(voxel_filter_points,dtype=torch.float32).to(vertices.device)
               vertices_array.append(vertices_vox)
               
               vertices_scale_array.append(torch.tensor(voxel_filter_scales,dtype=torch.float32).to(vertices.device).unsqueeze(1))
        return vertices_array, vertices_scale_array
# @torch.no_grad()

    def voxel_filter(self,points,scales,voxel_size):

        # 创建一个字典来存储体素内的点
        voxel_dict = {}
        
        # 遍历点云中的每个点
        for point, scale in zip(points, scales):
            # 计算体素的位置
            voxel_coord = np.floor(point / voxel_size)
            voxel_coord_tuple = tuple(voxel_coord.astype(int))

            if voxel_coord_tuple not in voxel_dict:
                voxel_dict[voxel_coord_tuple] = {'points': [], 'scales': []}
            
            voxel_dict[voxel_coord_tuple]['points'].append(point)
            voxel_dict[voxel_coord_tuple]['scales'].append(scale)

        # 计算每个体素的代表点和颜色
        new_points = []
        new_scales = []

        for voxel in voxel_dict.values():
            if len(voxel['points']) > 0:
                # 计算平均位置和平均颜色
                avg_point = np.mean(voxel['points'], axis=0)
                avg_scale = np.mean(voxel['scales'], axis=0)
                new_points.append(avg_point)
                new_scales.append(avg_scale)

        return new_points,new_scales
        
    def get_tetra_points_from_depth(self,depths,views,extend):
    #渲染深度
        from utils.depth_utils import depths_to_points
        import open3d as o3d
        from tqdm import tqdm
        pcd = o3d.geometry.PointCloud()
        device = depths[0].device
        
        for i,depth in tqdm(enumerate(depths)):
            view = views[i]
            # render_pkg = render(view, gaussians, pipe, background, kernel_size=kernel_size)
            # mode_id, mode, point_list, depth, means2D, conic_opacity = render_pkg["mode_id"], render_pkg["modes"], render_pkg["point_list"], render_pkg["alpha_depth"], render_pkg["means2D"], render_pkg["conic_opacity"] 
            points = depths_to_points(view, depth).reshape(-1, 3)
            print("raw_point_num",points.shape)
            
            #筛选在extend中的点
            direct_filtering_mask = ((points[...,0] > extend[0]) * (points[...,0] < extend[2]) * (points[...,2] > extend[1]) * (points[...,2] < extend[3]))
            filter_points = points[direct_filtering_mask]
            
            print("filter_point_num",filter_points.shape)
            
            if i == 0:
                pcd.points = o3d.utility.Vector3dVector(filter_points.cpu().numpy())
            else:
                origin_points = np.asarray(pcd.points)
                filter_points_np = filter_points.cpu().numpy()
                
                hb_points = np.concatenate([origin_points,filter_points_np],axis=0)
                pcd.points = o3d.utility.Vector3dVector(hb_points)
        pcd = pcd.remove_duplicated_points()
        pcd = pcd.uniform_down_sample(every_k_points = 32)
        o3d.io.write_point_cloud("/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station2/block0/temp/depth_points.ply", pcd)

        all_points = torch.tensor(np.asarray(pcd.points)).to(device)
        return all_points,torch.ones_like(all_points,device=device) * 10
        
    
    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # self.filter_3D = self.filter_3D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        #TODO:change 2 3d gaussian
        # padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        # padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        # selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        #TODO:chang to 3d gaussian
        # selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune
    def densify_and_scale_split(self, grad_threshold, min_opacity, scene_extent, max_screen_size, scale_factor, scene_mask=None, N=2, no_grad=True):
        assert scale_factor > 0
        n_init_points = self.get_xyz.shape[0]
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent * scale_factor
        if max_screen_size:
            # scale_mask = torch.logical_or(
            #     scale_mask,
            #     self.max_radii2D > max_screen_size
            # )
            pass
        if scene_mask:
            scale_mask = torch.logical_and(scene_mask, scale_mask)
        if no_grad:
            selected_pts_mask = scale_mask
        else:
            # Extract points that satisfy the gradient condition
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, scale_mask)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # prune_filter = torch.logical_or(prune_filter, (self.get_opacity < min_opacity).squeeze())
        
        self.prune_points(prune_filter)

        torch.cuda.empty_cache()
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
    def setPlaneIso(self,iso):
        self.iso = iso
    def updatePlaneMask(self):
        sorted_scaling,sorted_index = torch.sort(self.get_scaling,dim=1)
        bs = (sorted_scaling[...,1] / sorted_scaling[...,0])
        new_mask = (bs > self.iso)#n,
        # print(torch.sum(new_mask) / self.get_xyz.shape[0])
        #plane gaussian can not to be not plane gaussian
        # self.plane_mask = (self.plane_mask | new_mask)
        
        self.plane_mask = new_mask
        self.smallest_dir = sorted_index[new_mask][...,0].unsqueeze(-1).to(torch.int64)
    
    #根据
    def filter_mask(self,mask):
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._opacity = self._opacity[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self.filter_3D = self.filter_3D[mask]
    
    def to_cpu(self):
        self._xyz = self._xyz.cpu()
        self._features_dc = self._features_dc.cpu()
        self._features_rest = self._features_rest.cpu()
        self._opacity = self._opacity.cpu()
        self._scaling = self._scaling.cpu()
        self._rotation = self._rotation.cpu()
        self.filter_3D = self.filter_3D.cpu()