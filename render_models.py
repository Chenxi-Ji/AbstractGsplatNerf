import torch
import torch.nn as nn

from utils import quaternion_to_rotation_matrix, convert_input_to_pose, alpha_blending

class TransferModel(nn.Module):
    def __init__(self, model, rot, trans, transform_hom, scale, domain_type):
        super(TransferModel, self).__init__()

        self.model = model
        self.rot = rot
        self.trans = trans
        self.transform_hom = transform_hom
        self.scale = scale
        self.domain_type = domain_type

    def preprocess(self, input):
        pose = convert_input_to_pose(input, self.rot, self.trans, self.transform_hom, self.scale, self.domain_type)
        return pose

    def sort_gauss(self, input):
        pose = self.preprocess(input)
        self.model.sort_gauss(pose)

    def crop_gauss(self, input, tile_dict):
        pose = self.preprocess(input)
        self.model.crop_gauss(pose, tile_dict)

    def get_num(self):
        return self.model.get_num()

    def get_gs_batch(self):
        return self.model.get_gs_batch()
    
    def get_color_tile(self):
        return self.model.get_color_tile()
    
    def get_bg_color_tile(self):
        return self.model.get_bg_color_tile()

    def update_model_param(self, idx_start, idx_end, blending_method):
        self.model.update_model_param(idx_start, idx_end, blending_method)

    def forward(self, input):
        pose = self.preprocess(input)
        res = self.model.forward(pose)
        return res

class GsplatRGB(nn.Module):
    def __init__(self, camera_dict, scene_dict_all, bg_color=None, gs_batch=120, gs_max_num=100):
        super(GsplatRGB, self).__init__()

        self.camera_dict = camera_dict
        self.gs_batch = gs_batch
        self.gs_max_num = gs_max_num

        self.preprocess(scene_dict_all, bg_color)

    def preprocess(self, scene_dict_all, bg_color):

        means, quats, opacities, scales, colors  = (scene_dict_all[key] for key in ["means", "quats", "opacities", "scales", "colors"])
        DEVCIE = means.device

        # Process Means and Convariance Matrices in World Coordinates
        N = means.size(0)
        ones = torch.ones(N, 1, device=DEVCIE)
        means_hom_world = torch.cat([means, ones], dim=1)  # [N, 4]

        Rs = quaternion_to_rotation_matrix(quats)
        Ss = torch.diag_embed(scales)
        Ms = Rs@Ss # [N, 3, 3]
        Ms_world = Ms # [N, 3, 3]

        self.scene_dict_all = {
            "means_hom_world": means_hom_world,
            "Ms_world": Ms_world,
            "opacities": opacities,
            "colors": colors,
        }

        if bg_color == None:
            fx, fy, width, height = (self.camera_dict[key] for key in ["fx", "fy", "width", "height"])
            bg_color = torch.zeros((height, width, 3), device=DEVICE)
        self.bg_color = bg_color

    def sort_gauss(self, pose):

        # Extract Parameters
        fx, fy, width, height = (self.camera_dict[key] for key in ["fx", "fy", "width", "height"])
        means_hom_world = self.scene_dict_all["means_hom_world"]
        
        N = means_hom_world.size(0)
        DEVICE = means_hom_world.device
        pose = pose.to(DEVICE)

        # Step 1: Convert from World Coordinates to Camera Coordinates
        means_hom_cam = torch.matmul(pose, means_hom_world[None, :, :].transpose(-1,-2)).transpose(-1,-2)    # [1, N, 4]
        means_cam = means_hom_cam[:, :, :3] # [1, N, 3]
        depth = means_cam[0, :, 2] # [N, ]

        # Step 2: Filter Gaussians based on depth 
        mask = (depth >= 0.01)
        sorted_indices = torch.argsort(depth[mask])

        self.scene_dict_sorted = {
            name: attr[mask][sorted_indices]
            for name, attr in self.scene_dict_all.items()
        }

        # print(f"Number of Gauss after Sorting {mask.sum().item()}")
        
    def crop_gauss(self, pose, tile_dict, alpha_acc_thre = 0.99):

        # Update Tile information
        self.tile_dict = tile_dict
        hl,wl,hu,wu = (self.tile_dict[key] for key in ["hl", "wl", "hu", "wu"])
        colors = self.scene_dict_sorted["colors"]
        self.bg_color_tile = self.bg_color[hl:hu, wl:wu, :]

        # Compute Alpha
        alpha = self.render_alpha(pose, self.scene_dict_sorted).squeeze(-1) # [B, TH, TW, N]
        alpha_max = alpha.amax(dim=(0, 1, 2))

        # Filter Too small Alpha
        idx1, idx2, idx3 =40, 200, 1000
        mask = (alpha_max>0.1) 
        mask[idx1:] = mask[idx1:] & (alpha_max[idx1:] > 0.2) 
        mask[idx2:] = mask[idx2:] & (alpha_max[idx2:] > 0.25) 
        mask[idx3:] = mask[idx3:] & (alpha_max[idx3:] > 0.3) 
        N_masked = mask.sum().item()
            
        if N_masked==0:
            N=0
            self.scene_dict=None

        else:
            alpha_masked = alpha[:, :, :, mask] # [B, TH, TW, NM]
            alpha_shifted = torch.cat([torch.zeros_like(alpha_masked[:,:,:,0:1], dtype=alpha.dtype), alpha_masked[:,:,:,:-1]], dim=-1) # [B, TH, TW, NM]
            transmittance = torch.cumprod((1-alpha_shifted), dim=-1) # [B, TH, TW, NM]

            alpha_acc= torch.cumsum(alpha_masked*transmittance, dim=-1) # [B, TH, TW, NM]
            mask_acc = alpha_acc > alpha_acc_thre

            idx = torch.argmax(mask_acc.int(), dim=-1)
            idx[mask_acc.sum(dim=-1) == 0] = N_masked
            max_idx = idx.max()

            N=max_idx
            self.scene_dict = {
                name: attr[mask][:N]
                for name, attr in self.scene_dict_sorted.items()
            }
            print(f"Tile {hl,wl,hu,wu} Contains {N} Gaussians.")

        # Generate Up Triangel Matrix
        self.triu_mask = torch.triu(torch.ones(N+2, N+2), diagonal=1)
        

    def get_num(self):
        if self.scene_dict == None:
            return 0
        else:
            colors = self.scene_dict["colors"]
            return colors.size(0)

    def get_gs_batch(self):
        return self.gs_batch


    def get_color_tile(self):
        return self.scene_dict["colors"]

    def get_bg_color_tile(self):
        return self.bg_color_tile

    def update_model_param(self, idx_start, idx_end, blending_method):
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.blending_method = blending_method

    def render_alpha(self, pose, scene_dict, eps_max=1.0):

        # Extract Parameters
        fx, fy, width, height = (self.camera_dict[key] for key in ["fx", "fy", "width", "height"])
        means_hom_world, Ms_world, opacities  = (scene_dict[key] for key in ["means_hom_world", "Ms_world", "opacities"])
        hl,wl,hu,wu = (self.tile_dict[key] for key in ["hl", "wl", "hu", "wu"])

        N = opacities.size(0)
        DEVICE = opacities.device
        pose = pose.to(DEVICE)

        # Generate Mesh Grid
        pix_coord = torch.stack(torch.meshgrid(torch.arange(wl,wu), torch.arange(hl,hu), indexing='xy'), dim=-1)
        pix_coord = pix_coord.unsqueeze(0).to(DEVICE)  # [1, TH, TW, 2]

        # Step 1: Convert from World Coordinates to Camera Coordinates
        means_hom_cam = torch.matmul(pose, means_hom_world[None, :, :].transpose(-1,-2)).transpose(-1,-2)    # [1, N, 4]
        means_cam = means_hom_cam[:, :, :3] # [1, N, 3]

        us = means_cam[:, :, 0]
        vs = means_cam[:, :, 1]
        depth = means_cam[:, :,2] # [1, N]

        R_pose = pose[:, :3, :3]  # [1, 3, 3]
        Ms_cam = R_pose[:, None, :, :]@Ms_world[None, :, :, :] # [1, N, 3, 3]

        # Step 2: Prepare Matrix K and J for Coordinate Transformation
        Ks = torch.Tensor([[
            [fx, 0, width/2],
            [0, fy, height/2],
            [0,0,1]
        ]]).to(DEVICE) # [1, 3, 3]

        # tu = torch.min(depth*lim_u, torch.max(-depth*lim_u, us))
        # tv = torch.min(depth*lim_v, torch.max(-depth*lim_v, vs))

        J_00 = fx * depth # [1, N]
        J_02 = -fx * us 
        J_11 = fy * depth
        J_12 = -fy * vs

        J_00 = fx * depth # [1, N]
        J_02 = -fx * us #tu
        J_11 = fy * depth
        J_12 = -fy * vs#tv

        J_row0 = torch.stack([J_00, torch.zeros_like(J_00), J_02], dim=-1)  # [1, N, 3]
        J_row1 = torch.stack([torch.zeros_like(J_00), J_11, J_12], dim=-1)  # [1, N, 3]
        Js = torch.stack([J_row0, J_row1], dim=-2) # [1, N, 2, 3]

        # Step 3: Convert from Camera Coodinates to Pixel Coordinates
        means_hom_pix = means_cam @ Ks.transpose(1, 2) # [1, N, 3]
        means_pix = means_hom_pix[:, :, :2] # [1, N, 2]

        Ms_pix = Js@Ms_cam # [1, N, 2, 3]

        # covs_pix = Ms_pix@Ms_pix.transpose(-1,-2) # [1, N, 2, 2]
        # covs_pix_00 = covs_pix[:, :, 0, 0] # [1, N]
        # covs_pix_01 = covs_pix[:, :, 0, 1] # [1, N]
        # covs_pix_11 = covs_pix[:, :, 1, 1] # [1, N]

        # covs_pix_det = (covs_pix_00*covs_pix_11)-covs_pix_01*covs_pix_01

        Ms_pix_00 = Ms_pix[:, :, 0, 0] # [1, N]
        Ms_pix_01 = Ms_pix[:, :, 0, 1]
        Ms_pix_02 = Ms_pix[:, :, 0, 2]
        Ms_pix_10 = Ms_pix[:, :, 1, 0]
        Ms_pix_11 = Ms_pix[:, :, 1, 1]
        Ms_pix_12 = Ms_pix[:, :, 1, 2]

        covs_pix_det = (Ms_pix_00*Ms_pix_11-Ms_pix_01*Ms_pix_10)**2+(Ms_pix_00*Ms_pix_12-Ms_pix_02*Ms_pix_10)**2+(Ms_pix_01*Ms_pix_12-Ms_pix_02*Ms_pix_11)**2
        # covs_pix_det += 1e-12 # May cause error

        covs_pix_00 = Ms_pix_00**2+Ms_pix_01**2+Ms_pix_02**2
        covs_pix_01 = Ms_pix_00*Ms_pix_10+Ms_pix_01*Ms_pix_11+Ms_pix_02*Ms_pix_12
        covs_pix_11 = Ms_pix_10**2+Ms_pix_11**2+Ms_pix_12**2

        # conics_pix_00 = covs_pix_11/covs_pix_det # [1, N]
        # conics_pix_01 = -covs_pix_01/covs_pix_det
        # conics_pix_11 = covs_pix_00/covs_pix_det

        # conics_pix_0 = torch.stack([conics_pix_00, conics_pix_01], dim=-1) # [1, N, 2]
        # conics_pix_1 = torch.stack([conics_pix_01, conics_pix_11], dim=-1)
        
        # conics_pix = torch.stack([conics_pix_0, conics_pix_1], dim=-2) # [1, N, 2, 2]

        # Step 4: Compute Probability Density and Alpha at Pixel Coordinates
        pix_diff = (pix_coord[:, :, :, None, :]*depth[:, None, None, :, None]-means_pix[:, None, None, :, :])*depth[:, None, None, :, None] #[1, TH, TW, N, 2]
        
        pix_diff_0 = pix_diff[:, :, :, :, 0] #[1, TH, TW, N]
        pix_diff_1 = pix_diff[:, :, :, :, 1]

        # prob_density = pix_diff_0**2*conics_pix_00[:, None, None, :]+2*pix_diff_0*pix_diff_1*conics_pix_01[:, None, None, :]+pix_diff_1**2*conics_pix_11[:, None, None, :] #[1, TH, TW, N]
        #prob_density = 1/covs_pix_det[:, None, None, :]*(pix_diff_0**2*covs_pix_11[:, None, None, :]-2*pix_diff_0*pix_diff_1*covs_pix_01[:, None, None, :]+pix_diff_1**2*covs_pix_00[:, None, None, :]) #[1, TH, TW, N]
        
        prob_density = 1/covs_pix_det[:, None, None, :]*(\
        (pix_diff_0*Ms_pix_10[:, None, None, :]-pix_diff_1*Ms_pix_00[:, None, None, :])**2+\
        (pix_diff_0*Ms_pix_11[:, None, None, :]-pix_diff_1*Ms_pix_01[:, None, None, :])**2+\
        (pix_diff_0*Ms_pix_12[:, None, None, :]-pix_diff_1*Ms_pix_02[:, None, None, :])**2) #[1, TH, TW, N]

        prob_density = prob_density.unsqueeze(-1) #[1, TH, TW, N, 1]

        # return prob_density.unsqueeze(-1)

        alpha = opacities[None, None, None, :, :]*torch.exp(-1/2*prob_density) # [1, TH, TW, N, 1]
        alpha = -torch.nn.functional.relu(-alpha+eps_max)+eps_max 

        return alpha # [1, TH, TW, N, 1]

    # def alpha_blending(self, alpha, colors, blending_method):
    #     return alpha_blending(alpha, colors, blending_method)

    # def render_color(self, pose):

    #     # Batch 
    #     hl,wl,hu,wu = (self.tile_dict[key] for key in ["hl", "wl", "hu", "wu"])
    #     if self.scene_dict is None:
    #         bg_color = self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 3)

    #         return bg_color
    #     else:
    #         N = self.scene_dict["opacities"].size(0)
    #         DEVICE = self.scene_dict["opacities"].device
    #         gs_batch = self.gs_batch

    #         alpha_list = []
    #         colors_list = []

    #         ####
    #         #print(self.alpha_remainder.shape)
    #         colors_batch = self.scene_dict["colors"].view(1, 1, 1, N, 3).repeat(1, hu-hl, wu-wl, 1, 1)
    #         alpha_batch = self.render_alpha(pose, self.scene_dict) # [1, TH, TW, B, 1]

    #         ones = torch.ones((1, hu-hl, wu-wl, 1, 1), device=DEVICE)
    #         bg_color = self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 1, 3)
    #         # print(alpha_batch.shape, self.alpha_remainder.shape, ones.shape)
    #         # alpha_batch = torch.cat([alpha_batch, ones], dim=-2)
    #         # colors_batch = torch.cat([colors_batch, bg_color], dim=-2)
    #         if self.alpha_remainder is None:
    #             alpha_batch = torch.cat([alpha_batch, ones], dim=-2)
    #             colors_batch = torch.cat([colors_batch, bg_color], dim=-2)
    #         else:
    #             alpha_batch = torch.cat([alpha_batch, self.alpha_remainder, ones], dim=-2)
    #             print(colors_batch.shape, self.colors_remainder.shape, bg_color.shape)
    #             colors_batch = torch.cat([colors_batch, self.colors_remainder, bg_color], dim=-2)

    #         colors_combined, alpha_combined = self.alpha_blending(alpha_batch, colors_batch, "fast").split([3,1], dim=-1) 
    #         return colors_combined.squeeze(-2)

    #     ####

    #     if N==0:
    #         print("Warning None Gaussian is selected!")

    #     else:
    #         for epcoh, idx_start in enumerate(range(0, N, gs_batch)):
    #             idx_end = min(idx_start + gs_batch, N)

    #             scene_dict_batch = {
    #                 name: attr[idx_start:idx_end]
    #                 for name, attr in self.scene_dict.items()
    #             }

    #             colors_batch = scene_dict_batch["colors"].view(1, 1, 1, idx_end-idx_start, 3).repeat(1, hu-hl, wu-wl, 1, 1)
    #             alpha_batch = self.render_alpha(pose, scene_dict_batch) # [1, TH, TW, B, 1]

    #             if epcoh <=1:
    #                 colors_combined, alpha_combined = self.alpha_blending(alpha_batch, colors_batch, "slow").split([3,1], dim=-1) # [1, TH, TW, 1, 1], [1, TH, TW, 1, 3]
    #             else:
    #                 colors_combined, alpha_combined = self.alpha_blending(alpha_batch, colors_batch, "fast").split([3,1], dim=-1) 

    #             alpha_list.append(alpha_combined)
    #             colors_list.append(colors_combined)


    #     # Add Background
    #     alpha_list.append(torch.ones((1, hu-hl, wu-wl, 1, 1), device=DEVICE))
    #     colors_list.append(self.bg_color[hl:hu, wl:wu, :].view(1, hu-hl, wu-wl, 1, 3)) 
            
    #     alpha_epoch = torch.cat(alpha_list, dim=-2) # [1, TH, TW, E, 1]
    #     colors_epoch = torch.cat(colors_list, dim=-2) # [1, TH, TW, E, 1]

    #     # Alpha-Blending
    #     colors_out, alpha_out = self.alpha_blending(alpha_epoch, colors_epoch, "slow").split([3,1], dim=-1) # [1, TH, TW, 1, 1], [1, TH, TW, 1, 3]

    #     # Output
    #     res = colors_out.squeeze(-2) # [1, TH, TW, 3]
    #     return res

    def render_color_alpha(self, pose):
        scene_dict = {
            name: attr[self.idx_start:self.idx_end]
            for name, attr in self.scene_dict.items()
        }

        alpha = self.render_alpha(pose, scene_dict) #[1, TH, TW, N, 3]
        colors = scene_dict["colors"].view(1, 1, 1, alpha.size(-2), 3).repeat(1, alpha.size(1), alpha.size(2), 1, 1) #[1, TH, TW, N, 3]
        colors_alpha = alpha_blending(alpha, colors, self.blending_method, self.triu_mask) # [1, TH, TW, N, 4]

        return colors_alpha
    
    def render_alpha_tile(self, pose):
        scene_dict = {
            name: attr[self.idx_start:self.idx_end]
            for name, attr in self.scene_dict.items()
        }

        alpha = self.render_alpha(pose, scene_dict) #[1, TH, TW, N, 3]
        return alpha
        

    def forward(self, input):
        # Extract Input
        pose = input

        return self.render_alpha_tile(pose)
        return self.render_alpha_2(pose)


class AlphaBlending(nn.Module):
    def __init__(self, blending_method, triu_mask):
        super(AlphaBlending, self).__init__()
        self.blending_method=blending_method
        self.triu_mask = triu_mask

    def forward(self, input):
        colors, alpha = input[:, :, :, :, :3], input[:, :, :, :, 3:4]
        return alpha_blending(alpha, colors, self.blending_method, self.triu_mask)
