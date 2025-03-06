'''
提取多个尺度点云,再根据点云提取Mesh
'''
import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
from tqdm import tqdm 
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
import open3d as o3d
import pymeshlab

@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
        
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            if return_color:
                color_integrated = ret["color_integrated"]    
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
            final_alpha = torch.min(final_alpha, alpha_integrated)
            
        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, filter_mesh : bool, texture_mesh : bool,extend = None,par_idx = None,plane_iso=2,levels=1,simplify = "uniform"):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")#mesh保存路径

    makedirs(render_path, exist_ok=True)
    
    # generate tetra points here
    '''
    points[n*9,3]:前n*8为一个标准bbox(-1,1)被施加对应3d gs的旋转和缩放,最后的n为3d gaussian的中心xyz
    points_scale[n*9,3]:所有bbox的scale
    '''
    points_array, points_scale_array = gaussians.get_lod_tetra_points(extend=extend,plane_iso=plane_iso,levels=levels,simplify = simplify)
    
    for level_new in range(levels):
        points_scale = points_scale_array[level_new]
        points = points_array[level_new]
        if os.path.exists(os.path.join(render_path, f"lod_cells_{level_new}.pt")):
            print("load existing cells")
            cells = torch.load(os.path.join(render_path, f"lod_cells_{level_new}.pt"))
            
        else:
            # create cell and save cells
            print("create cells and save")
            cells = cpp.triangulate(points)#Delaunay三角剖分生成四面体
            # we should filter the cell if it is larger than the gaussians
            torch.save(cells, os.path.join(render_path, f"lod_cells_{level_new}.pt"))
        # evaluate alpha
        alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)#透明度

        vertices = points.cuda()[None]
        tets = cells.cuda().long()

        print(vertices.shape, tets.shape, alpha.shape)
        level = 0.5
        def alpha_to_sdf(alpha):    
            sdf = alpha - level
            sdf = sdf[None]
            return sdf
        
        sdf = alpha_to_sdf(alpha)
        
        torch.cuda.empty_cache()
        # batch_size = 1024 * 4
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None],batch_size=None)#行进四面体
        
        torch.cuda.empty_cache()
        
        end_points, end_sdf = verts_list[0]
        end_scales = scale_list[0]
        
        faces=faces_list[0].cpu().numpy()
        points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
            
        left_points = end_points[:, 0, :]
        right_points = end_points[:, 1, :]
        left_sdf = end_sdf[:, 0, :]
        right_sdf = end_sdf[:, 1, :]
        left_scale = end_scales[:, 0, 0]
        right_scale = end_scales[:, 1, 0]
        distance = torch.norm(left_points - right_points, dim=-1)
        scale = left_scale + right_scale
        
        n_binary_steps = 8
        # n_binary_steps = 1

        for step in range(n_binary_steps):#二分搜索
            print("binary search in step {}".format(step))
            mid_points = (left_points + right_points) / 2
            alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size)
            mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
            
            ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

            left_sdf[ind_low] = mid_sdf[ind_low]
            right_sdf[~ind_low] = mid_sdf[~ind_low]
            left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
            right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        
            points = (left_points + right_points) / 2
            if step not in [n_binary_steps - 1]:
                continue
            
            if texture_mesh:
                _, color = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
                vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
            else:
                vertex_colors=None
            mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
            
            # filter
            if filter_mesh:
                print("filter_mesh")
                # mask = (distance <= scale).cpu().numpy()
                
                #for power\rural
                mask = (distance <= 3 * scale).cpu().numpy()
                
                face_mask = mask[faces].all(axis=1)
                mesh.update_vertices(mask)
                mesh.update_faces(face_mask)
            
            if par_idx is None:
                #file hole 
                ms = pymeshlab.MeshSet()
                m = pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices),
                face_matrix=np.asarray(mesh.faces),
                v_color_matrix=np.asarray(mesh.visual.vertex_colors) / 255)
                ms.add_mesh(m, "mesh")
                ms.meshing_close_holes()#补洞
                # ms.meshing_remove_connected_component_by_face_number(mincomponentsize = 300,removeunref  = True)
                ms.save_current_mesh(os.path.join(render_path, f"lod_mesh_{level_new}.ply"))
            else:
                return mesh
    print("Done!")
    
    

def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, filter_mesh : bool, texture_mesh : bool,partitioning = False,partitioning_num = 2,plane_iso=2,levels = 1,simplify = "uniform"):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    
    origin_extend = scene.origin_extend
    if not partitioning:
        with torch.no_grad():
            cams = scene.getTrainCameras()
            marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, filter_mesh, texture_mesh,extend=origin_extend,plane_iso=plane_iso,levels = levels,simplify = simplify)
    else:
        x_span = origin_extend[2] - origin_extend[0]
        z_span = origin_extend[3] - origin_extend[1]
        extends = []
        if x_span > z_span:
            x_add = x_span / partitioning_num
            for i in range(partitioning_num):
                extends.append([origin_extend[0] + i * x_add, origin_extend[1],origin_extend[0] + (i+1) * x_add , origin_extend[3]])
        else:
            z_add = z_span / partitioning_num
            for i in range(partitioning_num):
                extends.append([origin_extend[0], origin_extend[1] + i * z_add,origin_extend[2], origin_extend[1] + (i+1) * z_add])
        meshes = []
        for idx,extend in enumerate(extends):
            with torch.no_grad():
                cams = scene.getTrainCameras()
                one_mesh = marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, filter_mesh, texture_mesh,extend=extend,par_idx=idx,plane_iso=plane_iso)
                meshes.append(one_mesh)
        vertices = []
        faces = []
        vertex_colors = []
        vertex_nums = 0
        for mesh in meshes:
            one_vertexs = np.asarray(mesh.vertices)
            vertices.append(one_vertexs)
            faces.append(np.asarray(mesh.faces) + vertex_nums)
            vertex_colors.append(np.asarray(mesh.visual.vertex_colors) / 255)
            
            vertex_nums = (vertex_nums + len(one_vertexs))
            
        #concat
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        vertex_colors = np.concatenate(vertex_colors, axis=0)
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(
        vertex_matrix=vertices,
        face_matrix=faces,
        v_color_matrix=vertex_colors)
        ms.add_mesh(m, "mesh")
        ms.meshing_close_holes()#补洞
        # ms.meshing_remove_connected_component_by_face_number(mincomponentsize = 300,removeunref  = True)
        render_path = os.path.join(dataset.model_path, "test", "ours_{}".format(iteration), "fusion")
        ms.save_current_mesh(os.path.join(render_path, f"mesh_binary_search_30000.ply"))
        print("Done!")
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("-block_index", type=int, default = 0)
    parser.add_argument("-resolution", type=int, default = 4)
    parser.add_argument("-con", type=bool, default = True)
    
    parser.add_argument("-partitioning", type=bool, default = False)
    parser.add_argument("-partitioning_num", type=int, default = 2)
    
    # parser.add_argument("--no_block", type=bool, default = False)
    
    parser.add_argument("--plane_iso", type=int, default = 2)
    parser.add_argument("--levels", type=int, default = 5)
    
    parser.add_argument("--simplify", type=str, default="feature_points", help = "point cloud simplify methods")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.filter_mesh, args.texture_mesh,partitioning=args.partitioning,partitioning_num=args.partitioning_num,plane_iso=args.plane_iso,levels=args.levels,simplify = args.simplify)