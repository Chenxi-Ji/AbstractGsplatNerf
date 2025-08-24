import os 
import json 

import time
import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy.spatial.transform import Rotation 
from PIL import Image 

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from utils import dir_to_rpy_and_rot, generate_samples, alpha_blending,  regulate# ,transfer_c2w_to_w2c, convert_input_to_pose, compute_depth_and_radius, sort_gauss
from utils import generate_fixed_poses,generate_bound
from utils import alpha_blending_interval, alpha_blending_interval_2
from render_models import GsplatRGB, TransferModel, AlphaBlending

#from collections import defaultdict

# from simple_model2_alphatest5_2 import AlphaModel, DepthModel, MeanModel
# from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
# from generate_poses import generate_poses

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

bound_opts = {
    'conv_mode': 'matrix',
    'optimize_bound_args': {
        'iteration': 100, 
        # 'lr_alpha':0.02, 
        'early_stop_patience':5},
}, 

def alpha_blending_ref(net, input_ref):
    
    N = net.get_num()
    triu_mask = torch.triu(torch.ones(N+2, N+2), diagonal=1)
    bg_color=(net.get_bg_color_tile()).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2)

    else:
        N=min(N,2000)
        net.update_model_param(0,N,"middle")
        model = BoundedModule(net, input_ref, device=DEVICE)
        colors_alpha = model.forward(input_ref)  #[1, TH, TW, N, 4]

        # net.update_model_param(0,N,"fast")
        # colors_alpha = net.forward(input_ref)  #[1, TH, TW, N, 4]

        colors, alpha = colors_alpha.split([3,1], dim=-1)

        ones = torch.ones_like(alpha[:, :, :, 0:1, :])
        alpha = torch.cat([alpha,ones], dim=-2) # [1, TH, TW, 2, 1]
        colors = torch.cat([colors,bg_color], dim=-2) # [1, TH, TW, 2, 3]

        colors_alpha_out = alpha_blending(alpha, colors, "fast", triu_mask)
        color_out, alpha_out = colors_alpha_out.split([3,1], dim=-1)

        color_out = color_out.squeeze(-2)
        return color_out

def alpha_blending_ptb_linear(net, input_ref, input_ptb, bound_method):

    N = net.get_num()
    gs_batch = net.get_gs_batch()
    epoch = max(gs_batch,N//gs_batch)+3
    triu_mask = torch.triu(torch.ones(epoch+2, epoch+2), diagonal=1)
    bg_color=(net.get_bg_color_tile()).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2), bg_color.squeeze(-2)
    else:
        colors_lb_list = []
        colors_ub_list = []
        alpha_lb_list = []
        alpha_ub_list = []

        for i, idx_start in enumerate(range(0, N, gs_batch)):
            idx_end = min(idx_start + gs_batch, N)
            #print("epoch:", i)

            if i==0:
                net.update_model_param(idx_start,idx_end,"middle")
            else:
                net.update_model_param(idx_start,idx_end,"middle")


            model = BoundedModule(net, input_ref, bound_opts=bound_opts, device=DEVICE)

            # colors_alpha_lb_ibp, colors_alpha_ub_ibp = model.compute_bounds(x=(input_ptb,), method="ibp")
            # reference_interm_bounds = {}
            # for node in model.nodes():
            #     if (node.perturbed
            #         and isinstance(node.lower, torch.Tensor)
            #         and isinstance(node.upper, torch.Tensor)):
            #         reference_interm_bounds[node.name] = (node.lower, node.upper)

            # colors_alpha_lb, colors_alpha_ub = model.compute_bounds(x= (input_ptb, ), method=bound_method, reference_bounds=reference_interm_bounds)  #[1, TH, TW, N, 4]
            colors_alpha_lb, colors_alpha_ub = model.compute_bounds(x= (input_ptb, ), method=bound_method)  #[1, TH, TW, N, 4]
            colors_alpha_lb, colors_alpha_ub = regulate(colors_alpha_lb), regulate(colors_alpha_ub)

            color_lb, alpha_lb = colors_alpha_lb.split([3,1],dim=-1)
            color_ub, alpha_ub = colors_alpha_ub.split([3,1],dim=-1)

            print(f"color_lb min and color_ub max: {torch.min(color_lb).item():.4} {torch.max(color_ub).item():.4}")
            # print(f"alpha_lb min and alpha_ub max: {torch.min(alpha_lb).item():.4} {torch.max(alpha_ub).item():.4}")

            alpha_lb_list.append(alpha_lb)
            alpha_ub_list.append(alpha_ub)
            colors_lb_list.append(color_lb)
            colors_ub_list.append(color_ub)

        del model,colors_alpha_lb, colors_alpha_ub
        torch.cuda.empty_cache()

        # Add background
        ones = torch.ones_like(alpha_lb)
        alpha_lb_list.append(ones)
        alpha_ub_list.append(ones)

        colors_lb_list.append(bg_color)
        colors_ub_list.append(bg_color)

        alphas_lb = torch.cat(alpha_lb_list, dim = -2)
        alphas_ub = torch.cat(alpha_ub_list, dim = -2)
        colors_lb = torch.cat(colors_lb_list, dim = -2)
        colors_ub = torch.cat(colors_ub_list, dim = -2)

        colors_alphas_lb = torch.cat([colors_lb,alphas_lb], dim = -1) #[1, TH, TW, E ,4]
        colors_alphas_ub = torch.cat([colors_ub,alphas_ub], dim = -1)
        #colors_alphas_ref = (colors_alphas_lb+colors_alphas_ub)/2

        # blending_net = AlphaBlending("slow", triu_mask)
        # blending_model = BoundedModule(blending_net, colors_alphas_ref, device=DEVICE)

        # ptb = PerturbationLpNorm(x_L=colors_alphas_lb,x_U=colors_alphas_ub)
        # colors_alphas_ptb = BoundedTensor(colors_alphas_ref, ptb)
        # color_alpha_out_lb, color_alpha_out_ub = blending_model.compute_bounds(x= (colors_alphas_ptb, ), method="ibp")

        color_alpha_out_lb, color_alpha_out_ub = alpha_blending_interval(colors_alphas_lb, colors_alphas_ub)

        color_out_lb,alpha_out_lb = color_alpha_out_lb.split([3,1],dim=-1)
        color_out_ub,alpha_out_ub = color_alpha_out_ub.split([3,1],dim=-1)

    return color_out_lb.squeeze(-2), color_out_ub.squeeze(-2)

def alpha_blending_ptb_interval(net, input_ref, input_ptb, bound_method):

    N = net.get_num()
    gs_batch = net.get_gs_batch()
    bg_color=(net.get_bg_color_tile()).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2), bg_color.squeeze(-2)
    else:
        alpha_lb_list = []
        alpha_ub_list = []

        for i, idx_start in enumerate(range(0, N, gs_batch)):
            idx_end = min(idx_start + gs_batch, N)
            #print("epoch:", i)

            net.update_model_param(idx_start,idx_end,"middle")
            model = BoundedModule(net, input_ref, bound_opts=bound_opts, device=DEVICE)

            alpha_lb, alpha_ub = model.compute_bounds(x= (input_ptb, ), method=bound_method)  #[1, TH, TW, N, 4]
            alpha_lb, alpha_ub = regulate(alpha_lb), regulate(alpha_ub)

            #print(f"alpha_lb min and alpha_ub max: {torch.min(alpha_lb).item():.4} {torch.max(alpha_ub).item():.4}")

            alpha_lb_list.append(alpha_lb)
            alpha_ub_list.append(alpha_ub)

        del model
        torch.cuda.empty_cache()

        alpha_lb = torch.cat(alpha_lb_list, dim=-2)
        alpha_ub = torch.cat(alpha_ub_list, dim=-2)

        # Load Colors within Tile
        colors = net.get_color_tile()
        #print(alpha_lb.shape, colors.shape)
        colors = colors.view(1, 1, 1, alpha_lb.size(-2), 3).repeat(1, alpha_lb.size(1), alpha_lb.size(2), 1, 1)
        colors = torch.cat([colors, bg_color], dim = -2)

        # Add background
        ones = torch.ones_like(alpha_lb[:, :, :, 0:1, :])
        alpha_lb = torch.cat([alpha_lb, ones], dim=-2)
        alpha_ub = torch.cat([alpha_ub, ones], dim=-2)

        color_alpha_out_lb, color_alpha_out_ub = alpha_blending_interval_2(alpha_lb, alpha_ub, colors)

        color_out_lb,alpha_out_lb = color_alpha_out_lb.split([3,1],dim=-1)
        color_out_ub,alpha_out_ub = color_alpha_out_ub.split([3,1],dim=-1)

    return color_out_lb.squeeze(-2), color_out_ub.squeeze(-2)
    
    
def main(setup_dict):
    key_list = ["bound_method", "render_method", "width", "height", "f", "tile_size", "scene_path", "checkpoint_filename", "save_folder", "save_ref", "save_bound", "domain_type"]
    bound_method, render_method, width, height, f, tile_size, scene_path, checkpoint_filename, save_folder, save_ref, save_bound, domain_type = (setup_dict[key] for key in key_list)

    # Load Already Trained Scene Files
    script_dir = os.path.dirname(os.path.realpath(__file__))
    scene_folder = os.path.join(script_dir, 'nerfstudio/', scene_path)
    transform_file = os.path.join(scene_folder, 'dataparser_transforms.json')
    checkpoint_file = os.path.join(scene_folder, 'nerfstudio_models/', checkpoint_filename)

    # Load Transformation Matrix and Scale
    with open (transform_file, 'r') as fp:
        data_transform = json.load(fp)
        transform = np.array (data_transform['transform'])
        transform_hom = np.vstack((transform, np.array([0,0,0,1])))
        scale = data_transform['scale']


    transform_hom = torch.from_numpy(transform_hom).to(dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(scale,dtype=DTYPE, device=DEVICE)

    # Make Folder to Save Abstract Images
    save_folder_full = os.path.join(script_dir, save_folder)
    if not os.path.exists(save_folder_full):
        os.makedirs(save_folder_full)

    # Load Trained 3DGS 
    scene_parameters = torch.load(checkpoint_file)
    means = scene_parameters['pipeline']['_model.gauss_params.means'].to(DEVICE)
    quats = scene_parameters['pipeline']['_model.gauss_params.quats'].to(DEVICE)
    opacities = torch.sigmoid(scene_parameters['pipeline']['_model.gauss_params.opacities']).to(DEVICE)
    scales = torch.exp(scene_parameters['pipeline']['_model.gauss_params.scales']).to(DEVICE)
    colors = torch.sigmoid(scene_parameters['pipeline']['_model.gauss_params.features_dc']).to(DEVICE)
    gauss_num = means.size(0)
    print(f"Number of Total Gaussians in the Scene: {gauss_num}")

    assert torch.all((opacities>=0) & (opacities<=1))

    # Define camera_dict and scene_dict
    camera_dict = {
        "fx": f,
        "fy": f,
        "width": width,
        "height": height,
    }
    scene_dict_all = {
        "means": means,
        "quats": quats,
        "opacities": opacities,
        "scales": scales,
        "colors": colors
    }

    # Define Background Image
    bg_pure_color = torch.tensor([123/255, 139/255, 196/255])
    bg_color = bg_pure_color.view(1, 1, 3).repeat(height, width,  1).to(DEVICE)
    
    # Generate Rotation Matrix
    start_arr = np.array([-np.cos(np.deg2rad(20))*2.5, np.sin(np.deg2rad(20))*2.5, 0.0])*4
    end_arr = np.array([0.0, 0.0, 0.0])
    rot = dir_to_rpy_and_rot(start_arr, end_arr)
    rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)
    trans = np.array([-np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])*6
    # print("rot:",rot)

    # Define Operational Domain
    # input_min = torch.tensor([6, np.deg2rad(13), np.deg2rad(-1)]).to(DEVICE)
    # input_max = torch.tensor([7, np.deg2rad(15), np.deg2rad(1)]).to(DEVICE)
    input_min = torch.tensor([0.9]).to(DEVICE)
    input_max = torch.tensor([1]).to(DEVICE)
    # input_max = input_min
    partition_per_dim = 500

    inputs_lb, inputs_ub, inputs_ref = generate_bound(input_min, input_max, partition_per_dim) # [partition_per_dim^N, N]
    inputs_lb, inputs_ub, inputs_ref = inputs_lb.to(DEVICE), inputs_ub.to(DEVICE), inputs_ref.to(DEVICE)
    partition_num = len(inputs_ref)

    inputs_queue = list(zip(inputs_lb, inputs_ub, inputs_ref))

    absimg_num = 0

    # initialize tqdm without a fixed total
    pbar = tqdm(total=len(inputs_queue),desc="Processing inputs", unit="item")

    while inputs_queue:
        input_lb, input_ub, input_ref = inputs_queue.pop(0) # [N, ]
        input_lb, input_ub, input_ref = input_lb.unsqueeze(0), input_ub.unsqueeze(0), input_ref.unsqueeze(0) #[1, N]

        ptb = PerturbationLpNorm(x_L=input_lb,x_U=input_ub)
        input_ptb = BoundedTensor(input_ref, ptb)

        img_ref = np.zeros((height, width,3))
        img_lb = np.zeros((height, width,3))
        img_ub = np.zeros((height, width,3))

        render_net = GsplatRGB(camera_dict, scene_dict_all, bg_color).to(DEVICE)
        verf_net = TransferModel(render_net, rot, trans, transform_hom, scale, domain_type).to(DEVICE)
        verf_net.sort_gauss(input_ref)
        
        tiles_queue = [
            (h,w,min(h+tile_size, height),min(w+tile_size, width)) \
            for h in range(0, height, tile_size) for w in range(0, width, tile_size) 
        ] 

        while tiles_queue!=[]:
            hl,wl,hu,wu = tiles_queue.pop(0)
            # hl,wl,hu,wu = 12, 42, 18, 48
            tile_dict = {
                "hl": hl,
                "wl": wl,
                "hu": hu,
                "wu": wu,
            }

            input_samples = generate_samples(input_lb, input_ub, input_ref)
            verf_net.crop_gauss(input_samples, tile_dict)

            if save_ref:
                ref_tile = alpha_blending_ref(verf_net, input_ref)
                # print(f"ref_tile min and max: {torch.min(ref_tile).item():.4} {torch.max(ref_tile).item():.4}")
                
                

            if save_bound:
                lb_tile, ub_tile = alpha_blending_ptb_interval(verf_net, input_ref, input_ptb, bound_method)
                # print(f"lb_tile min and ub_tile max: {torch.min(lb_tile).item():.4} {torch.max(ub_tile).item():.4}")
                lb_tile_np = lb_tile.squeeze(0).detach().cpu().numpy() # [TH, TW, 3]
                ub_tile_np = ub_tile.squeeze(0).detach().cpu().numpy()
                img_lb[hl:hu, wl:wu, :] = lb_tile_np
                img_ub[hl:hu, wl:wu, :] = ub_tile_np

            
            if save_ref:
                ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                img_ref[hl:hu, wl:wu, :] = ref_tile_np
            if save_bound:
                lb_tile_np = lb_tile.squeeze(0).detach().cpu().numpy() # [TH, TW, 3]
                ub_tile_np = ub_tile.squeeze(0).detach().cpu().numpy()
                img_lb[hl:hu, wl:wu, :] = lb_tile_np
                img_ub[hl:hu, wl:wu, :] = ub_tile_np
            
        if save_ref:
            img_ref= (img_ref.clip(min=0.0, max=1.0)*255).astype(np.uint8)
            res_ref = Image.fromarray(img_ref)
            res_ref.save(f'{save_folder_full}/ref_{absimg_num}.png')

        if save_bound:
            img_lb = (img_lb.clip(min=0.0, max=1.0)*255).astype(np.uint8)
            img_ub = (img_ub.clip(min=0.0, max=1.0)*255).astype(np.uint8)
            res_lb = Image.fromarray(img_lb)
            res_ub = Image.fromarray(img_ub)
            res_lb.save(f'{save_folder_full}/lb_{absimg_num}.png')
            res_ub.save(f'{save_folder_full}/ub_{absimg_num}.png')

            absimg_num+=1

        pbar.update(1)
        break
    pbar.close()

            
        

    return 0

if __name__=='__main__':

    # Setup Parameters
    bound_method = 'backward'
    render_method = 'gsplat_rgb'
    
    width = 80#64*1
    height = 80#64*1
    f = 100#80*1
    tile_size = 6 #4

    scene_path = 'outputs/airplane_grey/splatfacto/2025-08-02_025446'
    checkpoint_filename = "step-000299999.ckpt"

    save_folder = "AbstractImages/output"
    save_ref = False
    save_bound = True

    setup_dict = {
        "bound_method": bound_method,
        "render_method": render_method,
        "width": width,
        "height": height,
        "f": f,
        "tile_size": tile_size,
        "scene_path": scene_path,
        "checkpoint_filename": checkpoint_filename,
        "save_folder": save_folder,
        "save_ref": save_ref,
        "save_bound": save_bound,
        "domain_type": "z",
    }

    start_time=time.time()
    main(setup_dict)
    end_time = time.time()

    print(f"Running Time:{(end_time-start_time)/60:.4f} min")


### Next Work
### 1 Use better method to drop off GS (maybe keep all involved GS but not perturbed)
### 2 Reduce Computation time of Cumulative Product
### 3 Reduce Computation Complexity of Matrix Multiplication