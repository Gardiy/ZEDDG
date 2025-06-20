import json
import os
import torch
import types
from diffusers import DDIMScheduler

try:
    from diffusers.models.attention import CrossAttention # deprecated version of github
except ImportError:
    from diffusers.models.attention import Attention as CrossAttention # other version of diffusers

from icecream import ic
from torchvision import transforms

from utils.sd_pipeline_img import ZEDDGStableDiffusionPipeline
from utils.attention_forward import new_forward
from utils.ddim_inversion import Inversion, load_512, save_images
from utils.multi_guide import block_scramble

from configs.diff_config import VARIATION_CFG
from typing import Any, Callable, Dict, List, Optional, Union

class ZEDDG():

    def __init__(self, device= "cpu"):
        # variation
        #attn_cfgs = VARIATION_CFG["self_attn"]
        #device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        self.dtype=torch.float32

        self.ldm_stable = ZEDDGStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=self.dtype
        ).to(device)
        self.generator = torch.Generator(device=device)
        self.ldm_stable.scheduler = self.scheduler
        self._attent_modulate()
        self.ldm_stable.enable_model_cpu_offload()
        self.ldm_stable.enable_xformers_memory_efficient_attention()
        self.inversion = Inversion(self.ldm_stable, VARIATION_CFG["path_seecode"] ,
                              VARIATION_CFG["guidance_scale"], 
                              VARIATION_CFG["num_ddim_steps"], 
                              VARIATION_CFG["invert_steps"])

    def latte(self, image_path, principal = False):
        im, craw = load_512(image_path)  #load image
        seecoder_latents = self.inversion.seecoder.encode(craw) #encode
        self.inversion.init_seecoder(seecoder_latents) #load for inversion
        x_ts =  self.inversion.ddim_inversion(im, dtype=self.dtype) #invert, x_ts,[50 latent]
        if principal:
            return x_ts, seecoder_latents
        else:
            return x_ts[-1]

    @torch.no_grad()
    def generate(self, x_t_in, x_ts, seecoder_latents):
        images = self.ldm_stable(
        generator=self.generator, # generator cuda
        latents=x_t_in, # x_t latent noise,  condicional
        prompt_embeds=seecoder_latents,
        num_inference_steps = VARIATION_CFG["num_ddim_steps"],
        guidance_scale = VARIATION_CFG["guidance_scale"],
        chain = x_ts, # last latent diffusion
        t_early = VARIATION_CFG["t_early"],
        output_type = 'np', # or latent
        ).images
        return images
    
    def _attent_modulate(self):
        for module in self.ldm_stable.unet.modules():
            if isinstance(module, CrossAttention):
                # use a placeholder function for the original forward.
                module.ori_forward = module.forward
                module.cfg = VARIATION_CFG.copy()["self_attn"]
                module.init_step = 1000
                module.step_size = module.init_step // VARIATION_CFG.copy()["inference"]["ddim_step"]
                module.t_align = module.cfg["t_align"]
                module.editing_early_steps = 1000
                module.forward = types.MethodType(new_forward, module)

    