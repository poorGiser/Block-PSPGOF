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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim

import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
'''
img:h * w * 3
'''
def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
  if img.shape[-1] != ref.shape[-1]:
    raise ValueError(
        f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
    )
  num_channels = img.shape[-1]
  img_mat = img.reshape([-1, num_channels])
  ref_mat = ref.reshape([-1, num_channels])
  is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
  mask0 = is_unclipped(img_mat)
  # Because the set of saturated pixels may change after solving for a
  # transformation, we repeatedly solve a system `num_iters` times and update
  # our estimate of which pixels are saturated.
  for _ in range(num_iters):
    # Construct the left hand side of a linear system that contains a quadratic
    # expansion of each pixel of `img`.
    a_mat = []
    for c in range(num_channels):
      a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
    a_mat.append(img_mat)  # Linear term.
    a_mat.append(np.ones_like(img_mat[:, :1]))  # Bias term.
    a_mat = np.concatenate(a_mat, axis=-1)
    warp = []
    for c in range(num_channels):
      # Construct the right hand side of a linear system containing each color
      # of `ref`.
      b = ref_mat[:, c]
      # Ignore rows of the linear system that were saturated in the input or are
      # saturated in the current corrected color estimate.
      mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
      ma_mat = np.where(mask[:, None], a_mat, 0)
      mb = np.where(mask, b, 0)
      # Solve the linear system. We're using the np.lstsq instead of np because
      # it's significantly more stable in this case, for some reason.
      w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
      assert np.all(np.isfinite(w))
      warp.append(w)
    warp = np.stack(warp, axis=-1)
    # Apply the warp to update img_mat.
    img_mat = np.clip(
        np.matmul(a_mat, warp), 0, 1)
  corrected_img = np.reshape(img_mat, img.shape)
  return corrected_img

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)

        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, scale):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ f"gt_{scale}"
                renders_dir = method_dir / f"test_preds_{scale}"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                
                ssims_cc = []
                psnrs_cc = []
                lpipss_cc = []
                device = renders[0].device
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # renders[idx].shape:1 3 859 1148 torch.tensor
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
                for idx in tqdm(range(len(renders)), desc="Color correct Metric evaluation progress"):
                    gt_image = gts[idx].squeeze().permute(1,2,0).cpu().numpy()#3 859 1148
                    predict_image = renders[idx].squeeze().permute(1,2,0).cpu().numpy()
                    pred_image_cc = color_correct(predict_image,gt_image)
                    gt_image,pred_image_cc = torch.tensor(pred_image_cc.transpose(2,0,1)).unsqueeze(0), torch.tensor(gt_image.transpose(2,0,1)).unsqueeze(0)
                    
                    ssims_cc.append(ssim(pred_image_cc,gt_image))
                    psnrs_cc.append(psnr(pred_image_cc,gt_image))
                    lpipss_cc.append(lpips_fn(pred_image_cc.to(device),gt_image.to(device)).detach())
                    
                    print(image_names[idx],psnrs_cc[len(psnrs_cc)-1])
                    
                    

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                
                print("  SSIM_CC : {:>12.7f}".format(torch.tensor(ssims_cc).mean(), ".5"))
                print("  PSNR_CC : {:>12.7f}".format(torch.tensor(psnrs_cc).mean(), ".5"))
                print("  LPIPS_CC: {:>12.7f}".format(torch.tensor(lpipss_cc).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims_cc).mean().item(),
                                                        "PSNR": torch.tensor(psnrs_cc).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss_cc).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims_cc).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs_cc).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss_cc).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=-1)
    
    args = parser.parse_args()
    evaluate(args.model_paths, args.resolution)
