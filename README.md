# LayerEdit (AAAI-2026)
This repository contains the official implementation of the paper [LayerEdit: Disentangled Multi-Object Editing via Conflict-Aware Multi-Layer Learning](https://arxiv.org/abs/2511.08251).

## 📖 Introduction
Text-driven multi-object image editing which aims to precisely modify multiple objects within an image based on text descriptions, has recently attracted considerable interest. Existing works primarily follow the localize-editing paradigm, focusing on independent object localization and editing while neglecting critical inter-object interactions. However, this work points out that the neglected attention entanglements in inter-object conflict regions, inherently hinder disentangled multi-object editing, leading to either inter-object editing leakage or intra-object editing constraints. We thereby propose a novel multi-layer disentangled editing framework LayerEdit, a training-free method which, for the first time, through precise object-layered decomposition and coherent fusion, enables conflict-free object-layered editing. Departing from conventional paradigm, the core idea of LayerEdit lies in fundamentally shifting the focus from mere intra-object target regions to inter-object conflict regions, which enables: conflict-free object disentangled editing through interobject conflict awareness and suppression; and structurecoherent object-layered fusion through inter-object structural modeling.

![teaser](assets/model_MOE.jpg)

## ✨ News ✨

- [2025/11/08] 🎉 LayerEdit has been accepted to AAAI 2026! 🎉
- [2025/11/20]  We release the code for LayerEdit! Let's edit together! 😍
  
## ToDo
- [ ] **Release inference code that combines grounding-sam (more accurate segmentation, but potentially slower, and at the same time, grounding-sam has no reasoning ability).** 

## ⚡️ Quick Start

### 🔧 Requirements and Installation

Install the requirements
```bash
conda create -n LayerEdit python=3.10.12
conda activate LayerEdit
pip install -r requirements.txt
```

### ✍️ Editing a multi-object image (providing some cases)
```bash
from edit.main import  edit, inverse, init_model

# case 1
input_image = "./example/1/init_image.png"
mask_paths = ["./example/1/mask_1.png", "./example/1/mask_2.png"]
source_prompt = "a right bird and a left bird"
fg_prompt = ["a crochet bird", "a origami bird"]

# case 2
input_image = "./example/0/init_image.png"
mask_paths = ["./example/0/mask_1.png", "./example/0/mask_2.png"]
source_prompt = "the right dog and the left dog"
fg_prompt = ["a baby leopard", "a rabbit"]

# case 3
input_image = "./example/2/923000000005.jpg"
mask_paths = ["./example/2/left-paint.jpg", "./example/2/right-paint.jpg", "./example/2/table-lamp.jpg", "./example/2/bed.jpg"]
source_prompt = "the left painting and the right painting and a table lamp and a bed"
fg_prompt = ["a painting of a mona lisa", "a painting of sea", "", "a sofa"]

NUM_DDIM_STEPS = 50
edit_path  = './output/edit.png'    
save_path = './output/save.png'
pipe, inversion, device = init_model(
    model_path="/mnt/bn/fufengy-lf/huggingface_models/stable-diffusion-xl-base-1.0",
    num_ddim_steps=NUM_DDIM_STEPS,
)
latent_list, latent, prompt_str, prompt_embeds, pooled_prompt_embeds = inverse(
    inversion, input_image, source_prompt=source_prompt
)
edit(
    pipe=pipe,
    steps=NUM_DDIM_STEPS,
    image_path=input_image,
    mask_paths=mask_paths,
    fg_prompt=fg_prompt,
    bg_prompt=prompt_str,
    bg_negative=prompt_str,
    latent=latent,
    latent_list=latent_list,
    edit_path=edit_path,
    save_path=save_path,
    device=device,
)
```

### 🌟 Inference Codes
```
python edit_image.py
```


## Citation

🌟 If you find our work helpful, please consider citing our paper and leaving valuable stars

```
@article{fu2025layeredit,
  title={LayerEdit: Disentangled Multi-Object Editing via Conflict-Aware Multi-Layer Learning},
  author={Fu, Fengyi and Huang, Mengqi and Zhang, Lei and Mao, Zhendong},
  journal={arXiv preprint arXiv:2511.08251},
  year={2025}
}
```
