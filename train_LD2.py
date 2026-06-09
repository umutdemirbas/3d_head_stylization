# Fix NumPy 1.20+ compatibility issue FIRST - before any other imports
import numpy as np
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'typeDict'):
    np.typeDict = {}

import shutil
import argparse
import os
import random
import torchvision
from tqdm import tqdm
from copy import deepcopy
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
import lpips

from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch_utils import misc
import dnnlib
import legacy 
from training.triplane import TriPlaneGenerator
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
   
def fix_seeds(seed, device):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    g = torch.Generator(device=device)
    g.manual_seed(seed)


def find_latest_checkpoint(save_path, prefix='G_', suffix='.pth'):
    if not os.path.isdir(save_path):
        return None, None

    candidates = []
    for fname in os.listdir(save_path):
        if fname == 'G_final.pth':
            continue
        if fname.startswith(prefix) and fname.endswith(suffix):
            try:
                it = int(fname[len(prefix):-len(suffix)])
            except ValueError:
                continue
            candidates.append((it, os.path.join(save_path, fname)))

    if candidates:
        best_it, best_path = max(candidates, key=lambda x: x[0])
        return best_path, best_it

    final_path = os.path.join(save_path, 'G_final.pth')
    if os.path.isfile(final_path):
        return final_path, None

    return None, None

class Coach():
    def __init__(self, diff_ckpt_path, G_ckpt_path,
                 save_path='demo', lr=1e-4, seed=0, device='cuda',
                 yaw_range_front=None, pitch_range=None,
                 **kwargs) -> None:
        self.save_path = save_path
        self.device = device
        self.lr = lr

        fix_seeds(seed, device)

        ## Networks
        self.G = self.set_generator(G_ckpt_path, device).requires_grad_(True).train()
        self.G_frozen = deepcopy(self.G).requires_grad_(False).eval()
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device).eval()

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.lr)

        self.num_inference_steps = 50
        # Single SD pipeline without ControlNets for low GPU memory
        self.pipe = StableDiffusionPipeline.from_pretrained(diff_ckpt_path, safety_checker=None, torch_dtype=torch.float16).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Freeze the diffusion pipeline components - we only use them for inference, not training
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        
        # Enable CPU offload for memory efficiency - this is critical for low GPU memory scenarios
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()
        
        ## Rendering parameters
        self.cam_pivot = torch.tensor([0, 0, 0], device=device)
        self.cam_radius = self.G.rendering_kwargs.get("avg_camera_radius", 2.7)
        self.intrinsics = FOV_to_intrinsics(18.837, device=device)
        self.conditioning_camera_params = self.get_pose(self.cam_pivot, self.intrinsics, yaw=0, pitch=0.2, cam_radius=self.cam_radius, device=device)

        self.yaw_range_front = yaw_range_front if yaw_range_front is not None else [-np.pi/3, np.pi/3]
        self.pitch_range = pitch_range if pitch_range is not None else [-np.pi/6, np.pi/6]

    def ldis_loss(self, Ti_prime, Tj_prime, Ti, Tj):
        """
        Loss proposed in https://arxiv.org/abs/2312.16837
        """
        transformed_distance = torch.norm(Ti_prime - Tj_prime, p=2, dim=1) ** 2 # ||Ti' - Tj'||
        original_distance = torch.norm(Ti - Tj, p=2, dim=1) ** 2 # ||Ti - Tj||
        original_distance = original_distance + 1e-8
        ldis = torch.abs((transformed_distance / original_distance) - 1)
        return ldis.mean()

    @staticmethod
    def set_generator(ckpt_path, device):
        with dnnlib.util.open_url(ckpt_path) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        del G
        return G_new

    @staticmethod
    def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=[-0.35,0.35], pitch_range=[-0.15,0.15], cam_radius=2.7, device='cuda', return_yaw=False):
        if yaw is None:
            yaw = np.random.uniform(yaw_range[0], yaw_range[1])
        if pitch is None:
            pitch = np.random.uniform(pitch_range[0], pitch_range[1])
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
        if return_yaw:
            return c, yaw
        return c

    def normalize(self, t):
        return (t - t.min()) / (t.max() - t.min() + 1e-6)

    @staticmethod
    def sample(curr_pipe, prompt,
               start_step=0, start_latents=None,
               guidance_scale=7.5, num_inference_steps=50,
               num_images_per_prompt=1, do_classifier_free_guidance=True,
               negative_prompt='', 
               return_noise_pred=False, 
               max_num_inference_steps=None, 
               device='cuda'):
        """
        Simplified sampling without ControlNet for low memory usage.
        """
        text_embeddings = curr_pipe._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        curr_pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=device)
            start_latents *= curr_pipe.scheduler.init_noise_sigma

        latents = start_latents.clone()

        if max_num_inference_steps is None:
            max_num_inference_steps = num_inference_steps
            
        for i in (range(start_step, max_num_inference_steps)):
            t = curr_pipe.scheduler.timesteps[i]
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = curr_pipe.scheduler.scale_model_input(latent_model_input, t).to(torch.float16)

            # Simple UNet forward pass without ControlNet
            noise_pred = curr_pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
            alpha_t = curr_pipe.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = curr_pipe.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
            latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

        if return_noise_pred:
            predicted_x0 = torch.from_numpy(curr_pipe.decode_latents(predicted_x0).transpose(0,3,1,2)).to(device).to(torch.float32)
            return guidance_scale * (noise_pred_text - noise_pred_uncond), predicted_x0, noise_pred
        
        images = curr_pipe.decode_latents(latents.to(torch.float16))
        images = curr_pipe.numpy_to_pil(images)
        return images

    def low_rank_approximation(self, grad, k=4):
        """
        Apply low-rank approximation to the gradient tensor while keeping the batch dimension intact.
        Weigh the top-4 singular values with 100%, 75%, 50%, and 25%, respectively.
        """
        B, C, H, W = grad.shape
        grad_approx = torch.zeros_like(grad)

        for b in range(B):
            grad_flat = grad[b].view(C, H * W)

            grad_flat = grad_flat.to(torch.float32)
            try:
                U, S, V = torch.svd(grad_flat)
            except: # In case of ill-conditioned matrix 
                return grad

            k = min(4, S.size(0))
            weights = torch.tensor([1.0, 0.75, 0.5, 0.25], device=grad.device)[:k]
            S_k = S[:k] * weights
            U_k = U[:, :k]
            V_k = V[:, :k]

            grad_low_rank = torch.mm(U_k, torch.mm(torch.diag(S_k), V_k.t()))
            grad_approx[b] = grad_low_rank.view(C, H, W)

        grad_approx = grad_approx.to(grad.dtype)
        return grad_approx

    @staticmethod
    def instance_norm(img_src, img_tgt):
        mean_a = img_src.mean(dim=(2, 3), keepdim=True)
        std_a = img_src.std(dim=(2, 3), keepdim=True) + 1e-5

        mean_b = img_tgt.mean(dim=(2, 3), keepdim=True)
        std_b = img_tgt.std(dim=(2, 3), keepdim=True) + 1e-5

        norm_tgt = (img_tgt - mean_b) / std_b
        result = norm_tgt * std_a + mean_a

        return result

    def score_distillation(self, pipe, img, img_frozen, in_prompt, start_step_range, guidance_scale=7.5,
                           base_weight=1.0, grad_mask=None, use_lowrank=False, lowrank_k=4,
                           grad_div_scale=1000, use_SDS=False, retain_graph=False):
        """
        Simplified score distillation without ControlNet and mirror losses.
        Performs score distillation and returns E[x_0|y]
        """
        latent = pipe.vae.encode(img.half())
        l = 0.18215 * latent.latent_dist.sample()

        with torch.no_grad():
            start_step = random.randint(*start_step_range)
            pipe.scheduler.set_timesteps(self.num_inference_steps)
            noise = torch.randn_like(l, device=self.device)
            noisy_l = pipe.scheduler.add_noise(l, noise, pipe.scheduler.timesteps[start_step])

            score, x0hat, _ = self.sample(curr_pipe=pipe,
                    prompt=in_prompt, 
                    negative_prompt='',
                    start_latents=noisy_l, start_step=start_step, 
                    num_inference_steps=self.num_inference_steps,
                    max_num_inference_steps=start_step+1,
                    return_noise_pred=True,
                    guidance_scale=guidance_scale)

        if use_lowrank:
            score_rank1 = self.low_rank_approximation(score, k=random.choice([1,2,3,4]) if lowrank_k==-1 else lowrank_k)
            score = self.instance_norm(score, score_rank1)
        
        if use_SDS:
            grad = (score-noise) * torch.sqrt(1 - pipe.scheduler.alphas_cumprod[start_step])
        else: # likelihood distillation
            grad = score * torch.sqrt(pipe.scheduler.alphas_cumprod[start_step])
        
        if grad_mask is not None:
            grad *= grad_mask
        
        grad /= grad_div_scale
        grad = grad.clamp(-1,1)
        grad = torch.nan_to_num(grad, 0, 0, 0)
        
        loss_score = base_weight * SpecifyGradient.apply(l, grad)
        loss_score.backward(retain_graph=retain_graph)

        return x0hat

    def get_batch(self, c, bs):
        """
        Generate a single batch of images. No mirror views for memory efficiency.
        """
        z = torch.from_numpy(np.random.randn(bs, self.G.z_dim)).to(self.device)
        ws = self.G.backbone.mapping(z, self.conditioning_camera_params.repeat(bs,1), truncation_psi=0.75, truncation_cutoff=14)

        input_dict = self.G.synthesis(ws, c.repeat(bs,1), forward_full=True, generate_background=False, return_triplanes=True)
        input_image = input_dict['image']
        input_mask = input_dict['image_mask']
        input_triplane = input_dict['triplanes']

        with torch.no_grad():
            input_dict_frozen = self.G_frozen.synthesis(ws, c.repeat(bs,1), forward_full=True, generate_background=False, return_triplanes=True)
            input_image_frozen = input_dict_frozen['image']
            input_triplane_frozen = input_dict_frozen['triplanes']
            input_mask_frozen = input_dict['image_mask']

        return z, ws, input_image, input_triplane, input_image_frozen, input_triplane_frozen, input_mask_frozen

    def render_multi_view(self, ws, num_angles=5, num_pitch_angles=1, yaw_range=None, pitch_range=None, bs=1,
                          return_masks=False, generator=None, no_grad=True):
        if num_angles <= 1 and num_pitch_angles <= 1:
            return None

        if yaw_range is None:
            yaw_range = self.yaw_range_front
        if pitch_range is None:
            pitch_range = self.pitch_range

        yaws = np.linspace(yaw_range[0], yaw_range[1], max(1, num_angles))
        pitches = np.linspace(pitch_range[0], pitch_range[1], max(1, num_pitch_angles))

        c_list = []
        for pitch in pitches:
            for yaw in yaws:
                c = self.get_pose(self.cam_pivot, self.intrinsics, yaw=float(yaw), pitch=float(pitch), cam_radius=self.cam_radius, device=self.device)
                c_list.append(c.repeat(bs, 1))

        c_all = torch.cat(c_list, dim=0)
        repeat_dims = [len(c_list)] + [1] * (ws.ndim - 1)
        ws_all = ws.repeat(*repeat_dims)

        if generator is None:
            generator = self.G

        if no_grad:
            with torch.no_grad():
                render_dict = generator.synthesis(ws_all, c_all, forward_full=True, generate_background=False, return_triplanes=False)
        else:
            render_dict = generator.synthesis(ws_all, c_all, forward_full=True, generate_background=False, return_triplanes=False)

        images = render_dict['image']
        if return_masks:
            masks = render_dict['image_mask']
            return images, masks

        return images

    def train(self, 
              prompt='Portrait of a werewolf',
              bs=1, total_it=10_000, start_it=0,
              enable_ldis=True,
              enable_grad_masking=True,
              use_SDS=False,
              grad_div_scale=1000,
              guidance_scale=7.5,
              
              base_weight=0.75,
              tweedie_base_weight=1.0,
              lpips_base_weight=0.05,
              
              use_lowrank=True,
              lowrank_k=4,
              
              base_start_step_range=(35,49),
              num_angles=1,
              num_pitch_angles=1,
              
              # make mapping trainable by default for stronger learning
              unfreeze_mapping=1,
              # whether to train bias terms (0/1). default 0 to preserve original behaviour
              train_bias=0,

              freq_log_ckpt=250,
              freq_log_imgs=100,
              **kwargs
              ):
        """
        Optimized training loop for low GPU memory (RTX 2080 Ti, ~7-8GB).
        - Single view only (no multi-view grid)
        - No ControlNets (edge or depth)
        - No mirror losses
        - Keeps LPIPS and LDIS losses
        - CPU offload enabled on SD pipeline
        """

        # Freeze everything first, then selectively unfreeze modules to train.
        self.G.freeze_layers()
        requires_grad(self.G.backbone.synthesis, flag=True)
        requires_grad(self.G.superresolution, flag=True)

        # Optionally unfreeze mapping network for stronger updates
        if unfreeze_mapping:
            if hasattr(self.G.backbone, 'mapping'):
                requires_grad(self.G.backbone.mapping, flag=True)

        # Control bias training explicitly
        if train_bias:
            for n, p in self.G.named_parameters():
                if n.endswith('.bias'):
                    p.requires_grad = True
        else:
            self.G.freeze_bias()
        
        ## Get a single batch prior to training for LDIS loss
        c = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)
        z0, ws0, input_image0, input_triplane0, input_image_frozen0, input_triplane_frozen0, input_mask_frozen0 = self.get_batch(c, bs)

        if start_it >= total_it:
            print(f"Resume iteration {start_it} is already at or past total_it={total_it}. Nothing to do.")
            return

        for it in tqdm(range(start_it, total_it)):
            c = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)
            z, ws, input_image, input_triplane, input_image_frozen, input_triplane_frozen, input_mask_frozen = self.get_batch(c, bs)

            ## Single view distillation (no mirror, no grid)
            loss_single = {}
            # Retain graph if we have multi-view distillation or other losses coming
            has_other_losses = tweedie_base_weight > 0.0 or enable_ldis or lpips_base_weight > 0.0
            has_multiview = (num_angles > 1 or num_pitch_angles > 1)
            retain_for_later = has_multiview or has_other_losses
            
            x0hat = self.score_distillation(pipe=self.pipe, 
                                            img=input_image, img_frozen=input_image_frozen,
                                            start_step_range=base_start_step_range,
                                            in_prompt=prompt, 
                                            base_weight=base_weight,
                                            grad_mask=input_mask_frozen if enable_grad_masking else None,
                                            use_lowrank=use_lowrank,
                                            lowrank_k=lowrank_k,
                                            use_SDS=use_SDS,
                                            grad_div_scale=grad_div_scale,
                                            guidance_scale=guidance_scale,
                                            retain_graph=retain_for_later,
                                            )
            
            # Detach x0hat since its gradients were already applied in score_distillation
            x0hat = x0hat.detach()

            if num_angles > 1 or num_pitch_angles > 1:
                yaws = np.linspace(self.yaw_range_front[0], self.yaw_range_front[1], max(1, num_angles))
                pitches = np.linspace(self.pitch_range[0], self.pitch_range[1], max(1, num_pitch_angles))
                view_positions = [(float(yaw), float(pitch)) for pitch in pitches for yaw in yaws]

                for view_idx, (yaw, pitch) in enumerate(view_positions):
                    c_view = self.get_pose(self.cam_pivot, self.intrinsics,
                                            yaw=yaw, pitch=pitch,
                                            cam_radius=self.cam_radius, device=self.device)

                    view_dict = self.G.synthesis(ws, c_view.repeat(bs, 1),
                                                 forward_full=True, generate_background=False,
                                                 return_triplanes=False)
                    view_image = view_dict['image']
                    view_mask = view_dict['image_mask'] if enable_grad_masking else None

                    with torch.no_grad():
                        view_dict_frozen = self.G_frozen.synthesis(ws, c_view.repeat(bs, 1),
                                                                   forward_full=True, generate_background=False,
                                                                   return_triplanes=False)
                        view_image_frozen = view_dict_frozen['image']

                    # Retain graph for all but the last view, and for that last view we also need to retain
                    # if we have other losses to backward through
                    is_last_view = (view_idx == len(view_positions) - 1)
                    should_retain = not is_last_view or has_other_losses
                    self.score_distillation(pipe=self.pipe,
                                            img=view_image, img_frozen=view_image_frozen,
                                            start_step_range=base_start_step_range,
                                            in_prompt=prompt,
                                            base_weight=base_weight,
                                            grad_mask=view_mask,
                                            use_lowrank=use_lowrank,
                                            lowrank_k=lowrank_k,
                                            use_SDS=use_SDS,
                                            grad_div_scale=grad_div_scale,
                                            guidance_scale=guidance_scale,
                                            retain_graph=should_retain)

            # Keep LDIS and LPIPS losses
            if tweedie_base_weight>0.0:
                loss_single['loss_E0hat'] = tweedie_base_weight * torch.square(x0hat - 0.5*(input_image+1)).mean()  
            if enable_ldis:
                loss_single['ldis_loss'] = self.ldis_loss(input_triplane0, input_triplane, input_triplane_frozen0, input_triplane_frozen)
            if lpips_base_weight>0.0:
                loss_single['loss_E0hat_lpips'] = lpips_base_weight * self.lpips_loss_fn(2*(x0hat-0.5), input_image).mean()

            total_loss = 0
            for loss_value in loss_single.values():
                total_loss += loss_value
            total_loss.backward()
            self.G_optim.step()
            self.G_optim.zero_grad()

            if it % freq_log_imgs == 0:
                with torch.no_grad():
                    torchvision.utils.save_image(torch.cat([input_image, input_image_frozen, input_image0, input_image_frozen0, 2*(x0hat-0.5)],dim=-1),
                                                f'{self.save_path}/sv_{str(it).zfill(4)}.jpg', normalize=True, value_range=(-1,1))

                    if num_angles > 1 or num_pitch_angles > 1:
                        sample_z = torch.randn(1, self.G.z_dim, device=self.device)
                        sample_ws = self.G.backbone.mapping(sample_z, self.conditioning_camera_params, truncation_psi=0.75, truncation_cutoff=14)
                        multi_images = self.render_multi_view(sample_ws, num_angles=num_angles, num_pitch_angles=num_pitch_angles,
                                                             yaw_range=self.yaw_range_front, pitch_range=self.pitch_range)
                        if multi_images is not None:
                            torchvision.utils.save_image(multi_images, f'{self.save_path}/multi_{str(it).zfill(4)}.jpg', normalize=True, value_range=(-1,1), nrow=num_angles)

            if it % freq_log_ckpt == 0:
                torch.save(self.G.state_dict(), os.path.join(self.save_path, f'G_{str(it).zfill(4)}.pth'))

            ## Compute LDIS with previous batch 
            input_image0 = input_image.clone().detach()
            input_image_frozen0 = input_image_frozen.clone().detach()
            input_triplane0 = input_triplane.clone().detach()
            input_triplane_frozen0 = input_triplane_frozen.clone().detach()

        # Save a final checkpoint after training completes.
        torch.save(self.G.state_dict(), os.path.join(self.save_path, 'G_final.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', type=str, default="")

    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--total_it", type=int, default=5_000)
    parser.add_argument("--enable_ldis", type=int, default=1)
    parser.add_argument("--enable_grad_masking", type=int, default=1)
    parser.add_argument("--use_SDS", type=int, default=0)
    parser.add_argument('--grad_div_scale', type=float, default=1000)
    parser.add_argument('--guidance_scale', type=float, default=7.5)

    parser.add_argument('--base_weight', type=float, default=0.75)
    parser.add_argument('--tweedie_base_weight', type=float, default=0.1)
    parser.add_argument("--lpips_base_weight", type=float, default=0.05)

    parser.add_argument("--use_lowrank", type=int, default=1)
    parser.add_argument("--lowrank_k", type=int, default=4)

    parser.add_argument('--base_start_step_range', nargs='+', type=int, default=[35,49])
    parser.add_argument('--num_angles', type=int, default=1)
    parser.add_argument('--num_pitch_angles', type=int, default=1)
    parser.add_argument('--yaw_range_front', nargs=2, type=float, default=[-1.0471975511965976, 1.0471975511965976],
                        help='Yaw range for multi-view sampling, in radians: min max')
    parser.add_argument('--pitch_range', nargs=2, type=float, default=[-0.5235987755982988, 0.5235987755982988],
                        help='Pitch range for multi-view sampling, in radians: min max')
    parser.add_argument('--unfreeze_mapping', type=int, default=1, help='Unfreeze mapping network (1) or keep frozen (0)')
    parser.add_argument('--train_bias', type=int, default=0, help='Train bias parameters (1) or keep biases frozen (0)')

    parser.add_argument("--freq_log_imgs", type=int, default=100)
    parser.add_argument("--freq_log_ckpt", type=int, default=250)

    parser.add_argument('--diff_ckpt_path', type=str, default="")
    parser.add_argument('--G_ckpt_path', type=str, default="")

    parser.add_argument('--save_path', type=str, default="work_dirs/demo")
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default doubled from 1e-4)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run training on')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in save_path if available')

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))
    os.makedirs(args.save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.save_path, os.path.basename(__file__)))
    with open(os.path.join(args.save_path, "args_log_train.txt"), "w") as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

    resume_ckpt, resume_iter = None, 0
    if args.resume:
        resume_ckpt, resume_iter = find_latest_checkpoint(args.save_path)
        if resume_ckpt is not None:
            if resume_iter is None:
                print(f"Found final checkpoint {resume_ckpt}. Training appears complete.")
                exit(0)
            print(f"Resuming from checkpoint {resume_ckpt} at iteration {resume_iter}.")
            args.start_it = resume_iter
        else:
            args.start_it = 0
    else:
        args.start_it = 0

    coach = Coach(**vars(args))
    if resume_ckpt is not None:
        coach.G.load_state_dict(torch.load(resume_ckpt, map_location=args.device))
        coach.G_frozen = deepcopy(coach.G).requires_grad_(False).eval()
    coach.train(**vars(args))
