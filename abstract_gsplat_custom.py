import os 
import json 
import shutil

import time
import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy.spatial.transform import Rotation 
from PIL import Image 

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

from utils import dir_to_rpy_and_rot, convert_input_to_rot
from utils import generate_bound, generate_samples
from utils import alpha_blending, alpha_blending_interval
from render_models import GsplatRGB, TransferModel

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
    
    N = net.call_model("get_num")
    triu_mask = torch.triu(torch.ones(N+2, N+2), diagonal=1)
    bg_color=(net.call_model("get_bg_color_tile")).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2)

    else:
        # N=min(N,2000)
        # net.call_model("update_model_param", 0,N,"middle")
        # model = BoundedModule(net, input_ref, device=DEVICE)
        # colors_alpha = model.forward(input_ref)  #[1, TH, TW, N, 4]

        net.call_model("update_model_param", 0,N,"fast")
        # print("intpu_ref:", input_ref.shape)
        colors_alpha = net.call_model_preprocess("render_color_alpha", input_ref)  #[1, TH, TW, N, 4]

        colors, alpha = colors_alpha.split([3,1], dim=-1)

        ones = torch.ones_like(alpha[:, :, :, 0:1, :])
        alpha = torch.cat([alpha,ones], dim=-2) # [1, TH, TW, 2, 1]
        colors = torch.cat([colors,bg_color], dim=-2) # [1, TH, TW, 2, 3]

        colors_alpha_out = alpha_blending(alpha, colors, "fast", triu_mask)
        color_out, alpha_out = colors_alpha_out.split([3,1], dim=-1)

        color_out = color_out.squeeze(-2)
        return color_out


def alpha_blending_ptb(net, input_ref, input_lb, input_ub, bound_method):

    N = net.call_model("get_num")
    gs_batch = net.call_model("get_gs_batch")
    bg_color=(net.call_model("get_bg_color_tile")).unsqueeze(0).unsqueeze(-2) #[1, TH, TW, N, 3]

    if N==0:
        return bg_color.squeeze(-2), bg_color.squeeze(-2)
    else:
        alphas_int_lb = []
        alphas_int_ub = []

        hl,wl,hu,wu = (net.call_model("get_tile_dict")[key] for key in ["hl", "wl", "hu", "wu"])

        ptb = PerturbationLpNorm(x_L=input_lb,x_U=input_ub)
        input_ptb = BoundedTensor(input_ref, ptb)

        with torch.no_grad():
            for i, idx_start in enumerate(range(0, N, gs_batch)):
                idx_end = min(idx_start + gs_batch, N)
                # print("epoch:", i)

                net.call_model("update_model_param",idx_start,idx_end,"middle")
                model = BoundedModule(net, input_ref, bound_opts=bound_opts, device=DEVICE)

                alpha_ibp_lb, alpha_ibp_ub = model.compute_bounds(x=(input_ptb, ), method="ibp")
                reference_interm_bounds = {}
                for node in model.nodes():
                    if (node.perturbed
                        and isinstance(node.lower, torch.Tensor)
                        and isinstance(node.upper, torch.Tensor)):
                        reference_interm_bounds[node.name] = (node.lower, node.upper)

                alpha_int_lb, alpha_int_ub = model.compute_bounds(x= (input_ptb, ), method="forward", reference_bounds=reference_interm_bounds)  #[1, TH, TW, N, 4]
                
                alpha_int_lb = alpha_int_lb.reshape(1, hu-hl, wu-wl, idx_end-idx_start, 1)
                alpha_int_ub = alpha_int_ub.reshape(1, hu-hl, wu-wl, idx_end-idx_start, 1)

                alphas_int_lb.append(alpha_int_lb.detach())
                alphas_int_ub.append(alpha_int_ub.detach())

            del model
            torch.cuda.empty_cache()

            alphas_int_lb = torch.cat(alphas_int_lb, dim=-2)
            alphas_int_ub = torch.cat(alphas_int_ub, dim=-2)

        # Load Colors within Tile and Add background
        colors = net.call_model("get_color_tile")
        colors = colors.view(1, 1, 1, alphas_int_lb.size(-2), 3).repeat(1, alphas_int_lb.size(1), alphas_int_lb.size(2), 1, 1)
        colors = torch.cat([colors, bg_color], dim = -2)

        ones = torch.ones_like(alphas_int_lb[:, :, :, 0:1, :])
        alphas_int_lb = torch.cat([alphas_int_lb, ones], dim=-2)
        alphas_int_ub = torch.cat([alphas_int_ub, ones], dim=-2)        

        color_alpha_out_lb, color_alpha_out_ub = alpha_blending_interval(alphas_int_lb, alphas_int_ub, colors)

        color_out_lb,alpha_out_lb = color_alpha_out_lb.split([3,1],dim=-1)
        color_out_ub,alpha_out_ub = color_alpha_out_ub.split([3,1],dim=-1)

    return color_out_lb.squeeze(-2), color_out_ub.squeeze(-2)

    
# Helper to build synthetic scene 
def polar_to_cartesian(angle_deg, distance):
    rad = np.deg2rad(angle_deg)
    x = distance * np.cos(rad)
    y = distance * np.sin(rad)
    return x, y

def make_scene_from_splats(splats, device, dtype=torch.float32, sigma_min=1e-2, assume_scales_logspace=False):
    """
    Build scene_dict_all compatible with renderer from a list of splat dicts.
    Each splat: { 'angle' or 'pos', 'distance', 'sigma', 'color', 'opacity', optional 'z' }
    """
    means_list, quats_list, opac_list, scales_list, colors_list = [], [], [], [], []
    for s in splats:
        if 'pos' in s:
            x, y = s['pos'][:2]
        else:
            x, y = polar_to_cartesian(s['angle'], s.get('distance', 1.0)) #convert from angle / distance to cartesian coords
        z = s.get('z', 0.0) # set the z to zero for right now
        means_list.append([x, y, z])
        quats_list.append([0.0, 0.0, 0.0, 1.0])  # identity quaternion
        opac_list.append(float(s.get('opacity', 1.0)))
        sigma = s.get('sigma', 0.5)
        if isinstance(sigma, (float, int)):
            sx = sy = sz = max(sigma, sigma_min)
            # sz = sigma_min
        else:
            sx, sy, sz = [max(float(v), sigma_min) for v in sigma]
        scales_list.append([sx, sy, sz])
        colors_list.append(tuple(s.get('color', (1.0, 1.0, 1.0))))

    means = torch.tensor(means_list, dtype=dtype, device=device)           # (N,3)
    quats = torch.tensor(quats_list, dtype=dtype, device=device)           # (N,4)
    opacities = torch.tensor(opac_list, dtype=dtype, device=device).unsqueeze(-1)  # (N, 1)

    scales = torch.tensor(scales_list, dtype=dtype, device=device)        # (N,3)
    colors = torch.tensor(colors_list, dtype=dtype, device=device)        # (N,3)

    if assume_scales_logspace:
        scales = torch.log(torch.clamp(scales, min=1e-6))

    scene_dict_all = {
        "means": means,
        "quats": quats,
        "opacities": opacities,
        "scales": scales,
        "colors": colors
    }
    return scene_dict_all

def main(setup_dict):
    key_list = ["bound_method", "render_method", "width", "height", "f", "tile_size", "partition_per_dim", "selection_per_dim", "bg_img_path", "save_folder", "save_ref", "save_bound", "domain_type", "N_samples", "input_min", "input_max", "splats"]
    bound_method, render_method, width, height, f, tile_size, partition_per_dim, selection_per_dim, bg_img_path, save_folder, save_ref, save_bound, domain_type, N_samples, input_min, input_max, splats = (setup_dict[key] for key in key_list)
    
    # Get camera distance with default
    camera_z_distance = setup_dict.get("camera_z_distance", 10.0)
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Make Folder to Save Abstract Images
    save_folder_full = os.path.join(script_dir, save_folder)
    # Clear directory if it exists

    if not os.path.exists(save_folder_full):
        os.makedirs(save_folder_full)
    # if os.path.exists(save_folder_full):
    #     shutil.rmtree(save_folder_full)
    #     print(f"Cleared existing output directory: {save_folder_full}")
    # os.makedirs(save_folder_full)


    #TODO make this autoclear feature work for partitions and non partitions
    # outputs_folder = os.path.join(script_dir, "Outputs")
    # if os.path.exists(outputs_folder):
    #     shutil.rmtree(outputs_folder) 
    #     # os.removedirs(outputs_folder)


    # # if not os.path.exists(save_folder_full):
    # os.makedirs(save_folder_full)

    # Generate synthetic scene from splats
    scene_dict_all = make_scene_from_splats(splats, device=DEVICE, assume_scales_logspace=False)
    means = scene_dict_all['means']
    quats = scene_dict_all['quats']
    opacities = scene_dict_all['opacities']
    scales = scene_dict_all['scales']
    colors = scene_dict_all['colors']
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

    # Define Background Image
    if bg_img_path is None:
        bg_pure_color = torch.tensor([123/255, 139/255, 196/255], dtype=DTYPE, device=DEVICE)
        bg_color = bg_pure_color.view(1, 1, 3).repeat(height, width, 1).to(DEVICE)
    else:
        bg_img = Image.open(bg_img_path).convert("RGB")  # ensure 3 channels
        bg_img = bg_img.resize((width, height), Image.LANCZOS) 
        bg_img = np.array(bg_img, dtype=np.float32)  # shape: (H, W, 3)
        bg_color = torch.from_numpy(bg_img/255).to(DEVICE)  # shape: (H, W, 3)
    

    #TODO make this dynamic.
    #Initial gaussian positoin
    pos_start = np.array([0.0, 0.0, 0.0])
    pos_end = np.array([2.0, 0.0, 0.0])

    gs_rot = dir_to_rpy_and_rot(pos_start, pos_end)
    # gs_rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)
    gs_trans = np.array([0.0, 0.0, 0.0])

    gs_transform_matrix = np.identity(4)
    # gs_transform_matrix[:3, :3] = gs_rot
    gs_transform_matrix[:3, 3] = gs_trans

    print(f"gs transformation matrix: {gs_transform_matrix}")






    # Generate Rotation Matrix
    start_arr = np.array([0.0, 0.0, camera_z_distance])  
    end_arr = np.array([0.0, 0.0, camera_z_distance])  
    rot = dir_to_rpy_and_rot(start_arr, end_arr)
    # rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)
    # Base translation: 
    # trans[0] = base x position (0 for x-axis movement)
    # trans[1] = base y position (0 for y-axis movement)  
    # trans[2] = fixed z distance from origin (camera looks along z-axis)
    trans = np.array([0.0, 0.0, camera_z_distance])  
    # trans = torch.from_numpy(trans).to(device=DEVICE, dtype=DTYPE)

    camera_transformation = np.identity(4)
    camera_transformation[:3, :3] = rot
    camera_transformation[:3, 3] = trans

    # camera_transformation = camera_transformation @ np.linalg.inv(gs_transform_matrix)
    # camera_transformation = np.linalg.inv(gs_transform_matrix)
    camera_transformation = gs_transform_matrix


    rot = camera_transformation[:3, :3]
    trans = camera_transformation[:3, 3]
    rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)
    trans = torch.from_numpy(trans).to(device=DEVICE, dtype=DTYPE)



    
    # Identity transform and scale for synthetic scenes
    transform_hom = torch.eye(4, dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)

    inputs_lb, inputs_ub, inputs_ref = generate_bound(input_min, input_max, partition_per_dim, selection_per_dim) # [partition_per_dim^N, N]
    inputs_lb, inputs_ub, inputs_ref = inputs_lb.to(DEVICE), inputs_ub.to(DEVICE), inputs_ref.to(DEVICE)
    
    inputs_queue = list(zip(inputs_lb, inputs_ub, inputs_ref))

    absimg_num = 0

    # initialize tqdm without a fixed total
    pbar = tqdm(total=len(inputs_queue),desc="Processing inputs", unit="item")

    while inputs_queue:
        input_lb, input_ub, input_ref = inputs_queue.pop(0) # [N, ]
        input_lb, input_ub, input_ref = input_lb.unsqueeze(0), input_ub.unsqueeze(0), input_ref.unsqueeze(0) #[1, N]

        if save_ref:
            img_ref = np.zeros((height, width,3))
        if save_bound:
            img_lb = np.zeros((height, width,3))
            img_ub = np.zeros((height, width,3))

        rot = convert_input_to_rot(input_ref, trans, domain_type)
        rot = torch.from_numpy(rot).to(dtype=DTYPE, device=DEVICE)

        render_net = GsplatRGB(camera_dict, scene_dict_all, bg_color).to(DEVICE)
        verf_net = TransferModel(render_net, rot, trans, transform_hom, scale, domain_type).to(DEVICE)
        verf_net.call_model_preprocess("sort_gauss", input_ref)
        
        tiles_queue = [
            (h,w,min(h+tile_size, height),min(w+tile_size, width)) \
            for h in range(0, height, tile_size) for w in range(0, width, tile_size) 
        ] 

        while tiles_queue!=[]:
            hl,wl,hu,wu = tiles_queue.pop(0)
            tile_dict = {
                "hl": hl,
                "wl": wl,
                "hu": hu,
                "wu": wu,
            }

            input_samples = generate_samples(input_lb, input_ub, input_ref, N_samples)
            verf_net.call_model_preprocess("crop_gauss",input_samples, tile_dict)

            if save_ref:
                ref_tile = alpha_blending_ref(verf_net, input_ref)
                # print(f"ref_tile min and max: {torch.min(ref_tile).item():.4} {torch.max(ref_tile).item():.4}")
                ref_tile_np = ref_tile.squeeze(0).detach().cpu().numpy()
                img_ref[hl:hu, wl:wu, :] = ref_tile_np

            if save_bound:
                lb_tile, ub_tile = alpha_blending_ptb(verf_net, input_ref, input_lb, input_ub, bound_method)
                # print(f"lb_tile min and ub_tile max: {torch.min(lb_tile).item():.4} {torch.max(ub_tile).item():.4}")
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
        # if absimg_num>=1:
        #     break
    pbar.close()

    return 0



def generate_partitions(distance, distance_partitions, fov, fov_partitions):
        '''
        Inputs: 
        distance - the maximum distance the camera can see out to
        fov - the angular field of view of the camera (radians).
        distance_partitions - the number of partitions along the distance
        fov_partitions - the number of partitions along the angular dimension
        
        Returns:
        partitions: (N, 2) array of [distance, angle] partition centers
        distance_bounds: size of each distance partition
        angle_bounds: size of each angle partition
        partition_bounds: list of tuples (distance_lb, distance_ub, angle_lb, angle_ub) for each partition
        '''

        distances = np.linspace(0, distance, num=distance_partitions)
        angles = np.linspace(-fov/2, fov/2, num=fov_partitions)

        distance_bounds = distance / (distance_partitions - 1) if distance_partitions > 1 else distance
        angle_bounds = fov / (fov_partitions - 1) if fov_partitions > 1 else fov

        distance_grid, angle_grid = np.meshgrid(distances, angles, indexing='ij')
        partitions = np.column_stack((distance_grid.ravel(), angle_grid.ravel()))

        # Compute bounds for each partition
        partition_bounds = []
        for dist_center, angle_center in partitions:
            dist_lb = max(0, dist_center - distance_bounds / 2)
            dist_ub = min(distance, dist_center + distance_bounds / 2)
            angle_lb = angle_center - angle_bounds / 2
            angle_ub = angle_center + angle_bounds / 2
            partition_bounds.append((dist_lb, dist_ub, angle_lb, angle_ub))

        print(f"partitions{partitions}")
        print(f"partition shape: {partitions.shape}")

        return partitions, distance_bounds, angle_bounds, partition_bounds


def generate_splats(partitions):

    my_splats = []

    A = np.random.rand()
    B = np.random.rand()
    C = np.random.rand()


    #TODO #Should be ok to index the first dim (double check if things go wrong)
    for partition in partitions:

        my_splats.append({
            'distance': partition[0],  # Directly specify position at origin
            'angle': partition[1],
            'sigma': 1,
            'color': (A, B, C),  # Red color for visibility
            'opacity': 0.9
        })
    return my_splats


def render_partitions(setup_dict_base, distance, distance_partitions, fov, fov_partitions, 
                      partition_by_angle_only=True):
    """
    Generate abstract renders for each partition defined by generate_partitions.
    
    Args:
        setup_dict_base: Base setup dictionary (will be modified for each partition)
        distance: Maximum distance for partitions
        distance_partitions: Number of distance partitions
        fov: Field of view in radians
        fov_partitions: Number of angular partitions
        partition_by_angle_only: If True, only partition by angle (distance fixed at max)
    
    Returns:
        List of rendered partition information
    """
    # Generate partitions
    partitions, distance_bounds, angle_bounds, partition_bounds = generate_partitions(
        distance, distance_partitions, fov, fov_partitions
    )
    my_splats = generate_splats(partitions=partitions)
    
    print(f"Generated {len(partitions)} partitions")
    print(f"Distance bounds: {distance_bounds:.4f}, Angle bounds: {angle_bounds:.4f} rad ({np.degrees(angle_bounds):.2f} deg)")
    
    # For each partition, render abstract images
    for idx, (partition_center, bounds) in enumerate(zip(partitions, partition_bounds)):
        dist_lb, dist_ub, angle_lb, angle_ub = bounds
        
        print(f"\nPartition {idx+1}/{len(partitions)}: "
              f"Distance [{dist_lb:.2f}, {dist_ub:.2f}], "
              f"Angle [{np.degrees(angle_lb):.2f}°, {np.degrees(angle_ub):.2f}°]")
        
        # Create setup dict for this partition
        setup_dict = setup_dict_base.copy()
        
        if partition_by_angle_only:
            # Only partition by angle, use single distance
            # Set input bounds to angle bounds
            setup_dict["input_min"] = torch.tensor([angle_lb]).to(DEVICE)
            setup_dict["input_max"] = torch.tensor([angle_ub]).to(DEVICE)
            setup_dict["domain_type"] = "round" 
            # Update save folder to include partition info
            setup_dict["save_folder"] = f"{setup_dict_base['save_folder']}/partition_{idx:03d}_angle_{np.degrees(angle_lb):.1f}to{np.degrees(angle_ub):.1f}"

            setup_dict["splats"] = my_splats # redefine the splats based on the partitions
        else:
            # Partition by both distance and angle - would need 2D input support
            # This would require a custom domain_type or using "3" type
            raise NotImplementedError("2D partitioning (distance + angle) requires custom domain type")
        
        # Render abstract images for this partition
        main(setup_dict)
    
    return partitions, partition_bounds


if __name__=='__main__':

    # Setup Parameters
    bound_method = 'forward'
    render_method = 'gsplat_rgb'
    object_name = "synthetic_splats"
    
    width = 128
    height = 128
    # Focal length: smaller = wider FOV (zoomed out), larger = narrower FOV (zoomed in)
    # For 128x128 image: f=80 gives ~90° FOV, f=160 gives ~45° FOV, f=40 gives ~120° FOV
    f = 80  # Reduced from 160 to zoom out (wider field of view)
    tile_size = 64

    partition_per_dim = 100
    selection_per_dim = 5

    bg_img_path = None

    # Choose "x" to move along x-axis, or "y" to move along y-axis
    # Camera will be at fixed z distance (trans[2]) and move perpendicular to z-axis
    domain_type = "z"  # Camera moves along x-axis (perpendicular to z-axis)

    save_folder = "Outputs/AbstractImages/"+object_name+"/"+domain_type
    save_ref = True
    save_bound = True

    N_samples = 5

    # Camera moves along x-axis (perpendicular to z-axis)
    # Range: from -2 to +2 units along x-axis
    input_min = torch.tensor([0.0]).to(DEVICE)
    input_max = torch.tensor([7.0]).to(DEVICE)

    # Camera distance from origin (larger = further back, sees more)
    camera_z_distance = 10.0  

    # Create synthetic splats - centered at origin
    my_splats = []
    # Place gaussian at origin (0, 0, 0)
    my_splats.append({
        'pos': [0.0, 2.0, 3.0],  
        'sigma': 1,
        'color': (1.0, 0.0, 0.0), 
        'opacity': 0.9
    })

    my_splats.append({
        'distance': 1,  
        'angle': 180,
        'sigma': 1,
        'color': (0.0, 1.0, 0.0),  
        'opacity': 0.9
    })

    my_splats.append({
        'distance': 1,  
        'angle': 0,
        'sigma': 1,
        'color': (0.0, 0.0, 1.0),  
        'opacity': 0.9
    })

    # my_splats.append({
    # 'distance': 2.0,
    # 'angle': 90,
    # 'z': -1.0,  # Below the xy-plane
    # 'sigma': [1.5, 1.5, 0.3],  # Wide in x/y, thin in z
    # 'color': (1.0, 0.5, 0.0),  # Orange
    # 'opacity': 0.8
    # })

    # my_splats.append({
    #     'pos': [2.0, 0.0, 0.0],  # Directly specify position at origin
    #     'sigma': 1,
    #     'color': (0.0, 1.0, 0.0),  # Red color for visibility
    #     'opacity': 0.9
    # })


    setup_dict = {
        "bound_method": bound_method,
        "render_method": render_method,
        "width": width,
        "height": height,
        "f": f,
        "tile_size": tile_size,
        "partition_per_dim": partition_per_dim,
        "selection_per_dim": selection_per_dim,
        "bg_img_path": bg_img_path,
        "save_folder": save_folder,
        "save_ref": save_ref,
        "save_bound": save_bound,
        "domain_type": domain_type,
        "N_samples": N_samples,
        "input_min": input_min,
        "input_max": input_max,
        "splats": my_splats,
        "camera_z_distance": camera_z_distance,
    }

    # Option 1: Render single input range (original behavior)
    # start_time=time.time()
    # main(setup_dict)
    # end_time = time.time()
    # print(f"Running Time:{(end_time-start_time)/60:.4f} min")

    # Option 2: Render multiple partitions based on FOV
    use_partitions = False
    if use_partitions:
        fov_degrees = 90  # Total field of view in degrees
        fov_radians = np.deg2rad(fov_degrees)
        fov_partitions = 3  
        distance = 5.0  # Maximum distance
        distance_partitions = 1  # Only partition by angle, not distance
        
        start_time = time.time()
        render_partitions(
            setup_dict_base=setup_dict,
            distance=distance,
            distance_partitions=distance_partitions,
            fov=fov_radians,
            fov_partitions=fov_partitions,
            partition_by_angle_only=True
        )
        end_time = time.time()
        print(f"\nTotal Running Time: {(end_time-start_time)/60:.4f} min")
    else:
        # Original single-range rendering
        start_time=time.time()
        main(setup_dict)
        end_time = time.time()
        print(f"Running Time:{(end_time-start_time)/60:.4f} min")

