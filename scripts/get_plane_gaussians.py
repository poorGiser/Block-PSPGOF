'''
从给定的gaussians scale中获取平面高斯
平面高斯的定义:两个比较大的scale带着一个小的scale
'''
import torch
def get_plane_gaussians(scaling):
    sorted_scaling,_ = torch.sort(scaling,dim=1)
    
    multiple1 = 5.0
    multiple2 = 5.0
    
    plane_mask = ((sorted_scaling[...,2] / sorted_scaling[...,1]) < multiple1) *  ((sorted_scaling[...,1] / sorted_scaling[...,0]) > multiple2)
    # plane_mask = ((sorted_scaling[...,1] / sorted_scaling[...,0]) > 10.0)
    
    
    return plane_mask
    
    