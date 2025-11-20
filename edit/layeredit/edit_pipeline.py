import torch
import numpy as np
from layeredit.ptp_controller import make_controller
from layeredit.ptp_utils import register_attention_control
from PIL import Image
import torch.nn.functional as F
from layeredit.model import dilated_eroded_mask, LayerFusion

DIM = 128

def get_remove_rate_snr(alpha_bar_t, k=5, i_thres=30):
    snr = torch.sqrt(alpha_bar_t / (1 - alpha_bar_t))
    ratio = snr[i_thres] / snr
    return torch.sigmoid(k * (ratio - 1)).squeeze()

def pro_object_mask(mask_path):
    object_mask = [
        np.array(
            Image.open(mask_path_).convert("L").resize((1024, 1024), Image.BILINEAR)
        )
        for mask_path_ in mask_path
    ]
    object_mask = [
        dilated_eroded_mask(object_mask_, core=20)[0] for object_mask_ in object_mask
    ]
    object_mask = torch.stack(object_mask, 0)
    object_mask[object_mask > 0] = 1.0
    object_mask[object_mask <= 0] = 0.0
    object_mask = F.interpolate(
        object_mask.unsqueeze(1), size=(DIM, DIM), mode="bilinear"
    ).squeeze(1)

    return object_mask


def edit_pipe(
    image_path,
    mask_path,
    prompts,
    model,
    num_ddim_steps,
    self_replace_steps=0.1,
    cross_replace_steps=0.5,
    masa_control=False,
    ID_LorA=None,
    device=None,
    resize_pro=False,
    move_pro=False,
    resize_scale=[],
    move_direction=[],
    move_scale=[],
):  
    #print(model.scheduler.betas.device, model.scheduler.device)
    timesteps = model.scheduler.timesteps.to(model.scheduler.betas.device)
    beta_t, alpha_t, alpha_bar_t = (
        model.scheduler.betas[timesteps],
        model.scheduler.alphas[timesteps],
        model.scheduler.alphas_cumprod[timesteps],
    )
    timesteps_args = {"beta_t": beta_t, "alpha_t": alpha_t, "alpha_bar_t": alpha_bar_t}
    controller = make_controller(
        prompts,
        model,
        num_ddim_steps,
        is_replace_controller=False,
        cross_replace_steps=cross_replace_steps,
        self_replace_steps=self_replace_steps,
        masa_control=masa_control,
    )
    prompts = controller.prompts

    if mask_path == "":
        object_mask, controller.text_nps, controller.text_nps_id, controller.top_k = (
            SAM(image_path, prompts[0], output_path="")
        )
    else:
        object_mask = pro_object_mask(mask_path)

    object_mask_all = torch.sum(object_mask, 0)
    object_mask_all[object_mask_all > 0] = 1
    # get conflict region mask
    remove_mask = []
    for i_, mask_ in enumerate(object_mask):
        mask_ = object_mask_all - mask_  # use other object mask as conflict region mask
        mask_[mask_ < 0] = 0
        remove_mask.append(mask_)
    remove_mask.append(object_mask_all)
    remove_mask = torch.stack(remove_mask, 0).unsqueeze(1)

    for prompt_id, prompt in enumerate(prompts[1:-1]):
        if prompt == "":
            remove_mask[prompt_id] += object_mask[prompt_id]
            remove_mask[remove_mask > 0] = 1

    lb = LayerFusion(
        remove_mask=remove_mask,
        prompts=prompts,
        tokenizer=model.tokenizer,
        object_mask=object_mask.unsqueeze(1),
        object_mask_all=object_mask_all,
    )
    # for Geometric Structural Editing
    lb.resize_pro, lb.move_pro, lb.resize_scale, lb.move_direction, lb.move_scale = (
        resize_pro,
        move_pro,
        resize_scale,
        move_direction,
        move_scale,
    )

    lb.K_remove_rate = get_remove_rate_snr(
        timesteps_args["alpha_bar_t"], k=5, i_thres=40
    )
    lb.Q_remove_rate = get_remove_rate_snr(
        timesteps_args["alpha_bar_t"], k=5, i_thres=20
    )
    lb.ID_LorA = torch.tensor([0] * (len(prompts) - 1)).to(
        device
    )  #  0 for apperance editing，1 for layout editing

    controller.layer_fusion = lb
    register_attention_control(model, controller)
    return controller
