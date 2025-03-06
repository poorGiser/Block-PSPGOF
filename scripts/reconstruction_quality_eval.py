import torch
from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
def evaluate(raw_pcd_path, reconstructed_pcd_path):
    batch_size = 1
    
    cham3D = dist_chamfer_3D.chamfer_3DDist()
    
    raw_pcd = o3d.io.read_point_cloud(raw_pcd_path)#get form mesh
    raw_pcd_points = np.asarray(raw_pcd.points)
    raw_pcd_points_cuda = torch.tensor(raw_pcd_points).cuda().unsqueeze(0).to(torch.float32)
    
    reconstructed_pcd = o3d.io.read_point_cloud(reconstructed_pcd_path)#mesh
    reconstructed_pcd_points = np.asarray(reconstructed_pcd.points)
    reconstructed_pcd_points_cuda = torch.tensor(reconstructed_pcd_points).cuda().unsqueeze(0).to(torch.float32)
    
    dist1, dist2, idx1, idx2= cham3D(raw_pcd_points_cuda, reconstructed_pcd_points_cuda)
    
    print("Chamfer distance: ", (dist1.mean().item() + dist2.mean().item()) / 2)
    print(f"fscore :", fscore(dist1, dist2))
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--raw_pcd_path', '-s', type=str, default="/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/rural/point_cloud/raw_transform.ply")
    parser.add_argument('--raw_pcd_path', '-s', type=str, default="/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/data/power_station/point_cloud/raw_transform.ply")

    parser.add_argument('--reconstructed_pcd_path','-m', type=str, default="/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/output/power_station_ablation_3/merge/30000_merge_bbox.ply")
    
    args = parser.parse_args()
    evaluate(args.raw_pcd_path, args.reconstructed_pcd_path)