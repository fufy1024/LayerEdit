import os
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as T
import torch.nn as nn
import torch
from diffusers import DDIMScheduler
sys.path.append('./edit/')
from layeredit.sdxl import sdxl 
from layeredit.inversion import Inversion
from lavis.models import load_model_and_preprocess

device = torch.device('cuda')

from layeredit.edit_pipeline import edit_pipe
from glob import glob

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.float16)
seed = 1234
seed_everything(seed)

def save_image(images, edit_path, save_path=''):
    images[-1].save(edit_path)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    if save_path:
        new_im.save(save_path)
    print('save_path:',save_path)


def init_model(model_path, model_dtype="fp16", num_ddim_steps=50):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    if model_dtype == "fp16":
        torch_dtype = torch.float16
    elif model_dtype == "fp32":
        torch_dtype = torch.float32

    pipe = sdxl.from_pretrained(model_path, torch_dtype=torch_dtype, use_safetensors=True, variant=model_dtype,scheduler=scheduler) 
    pipe.to(device)
    inversion = Inversion(pipe,num_ddim_steps)
    return pipe, inversion, device

def inverse(inversion, input_image, source_prompt = ''):
    # if the input is a folder, collect all the images as a list
    if os.path.isdir(input_image):
        l_img_paths = sorted(glob(os.path.join(input_image, "*.png")))
    else:
        l_img_paths = [input_image]

    for img_path in l_img_paths:
        img = Image.open(img_path).resize((1024,1024), Image.Resampling.LANCZOS)
        # generate the caption        
        if source_prompt !='':
            prompt_str = source_prompt
        else:
            # load the BLIP model
            model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))
            _image = vis_processors["eval"](img).unsqueeze(0).to(device)
            prompt_str = model_blip.generate({"image": _image})[0]
        (x_inv_image, x_dec_img), x_inv, latent_list, prompt_embeds, pooled_prompt_embeds = inversion.invert(np.array(img), prompt_str, inv_batch_size=1) 
        return latent_list, x_inv[0].unsqueeze(0), prompt_str, prompt_embeds, pooled_prompt_embeds

def edit(pipe, steps, image_path, mask_paths, fg_prompt, bg_prompt, bg_negative, latent, latent_list, edit_path, save_path, device, fg_negative='artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'):
    prompts = [bg_prompt] + fg_prompt + ['']
    neg_prompts = [bg_negative] + [fg_negative] * (len(fg_prompt)+1)
    pipe.scheduler.set_timesteps(steps)
    # initialize the controller for multi-layer diffusion
    controller = edit_pipe(image_path, mask_paths, prompts, pipe, num_ddim_steps=steps, device=device)
    # generate the image with the foreground object editing prompts
    img_gen = pipe( controller=controller, prompt=prompts, latents=latent, negative_prompt=neg_prompts)
    save_image(img_gen, edit_path, save_path)

