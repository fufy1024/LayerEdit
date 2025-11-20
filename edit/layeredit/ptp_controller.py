import torch
import numpy as np
import abc
from layeredit.model import get_word_inds
from typing import Optional, Union, Tuple, List, Callable, Dict
import layeredit.seq_aligner as seq_aligner
import cv2
LOW_RESOURCE = False

def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


class LocalBlend:
    def __call__(self, x_t, attention_store):
        return x_t

    def __init__(self, prompts: List[str], words, tokenizer,device,num_ddim_steps,substruct_words=None, start_blend=0.2,
                 mask_threshold=0.6,x_t_replace=True, max_num_words=77, **kwargs):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, max_num_words)
        self.tokenizer=tokenizer
        self.mask_threshold=mask_threshold
        self.model_device=device
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, self.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, max_num_words)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, self.tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(self.model_device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(self.model_device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0
        self.mask=None
        self.x_t_replace=x_t_replace

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    if self.masa_control is False :
                        self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.layer_fusion is not None:
            x_t = self.layer_fusion(x_t)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 64 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    def count_layers(self,place_in_unet,is_cross):
        if self.last_status=='up' and place_in_unet=='down':
            self.self_layer=0
            self.cross_layer=0
        self.last_status=place_in_unet
        if is_cross is True:
            self.cross_layer=self.cross_layer+1
        else:
            self.self_layer=self.self_layer+1

    def replace_self_attention_kv(self, q, k, v, heads, Layer_S=1):  # for masactrl
        if (
            self.layer_fusion is not None
            and self.self_layer >= self.masa_start_layer
            and self.cur_step >= self.masa_start_step
        ):
            kv = torch.cat([k, v], dim=0)
            split_kv = kv.split(heads, dim=0)
            new_kv = torch.stack(split_kv, dim=0)
            split_new_kv = torch.chunk(new_kv, chunks=4, dim=0)
            new_kv = torch.stack(split_new_kv, dim=0)
            batch_size = new_kv.shape[0]
            assert batch_size > 1, "masa control : batch_size > 1"
            pro_kv = (
                torch.ones(new_kv[:, Layer_S:].shape).to(new_kv.device)
            ) * self.layer_fusion.ID_LorA[None, :, None, None, None]
            kv_save = new_kv[:, Layer_S:].clone()
            new_kv[:, Layer_S:] = (
                new_kv[:, 0].unsqueeze(1).expand_as(new_kv[:, Layer_S:]) * pro_kv
                + (1 - pro_kv) * kv_save
            )
            new_kv = new_kv.reshape(-1, *new_kv.shape[-2:])
            k, v = new_kv.chunk(2)
        return q, k, v

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str, Layer_S=1):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            L_pro = len(attn[Layer_S:])
            if is_cross and self.rep_in_cross_attn:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[Layer_S:] = attn_repalce_new[-L_pro:]
            elif self.rep_in_self_attn:
                attn[Layer_S:] = self.replace_self_attention(
                    attn_base, attn_repalce, place_in_unet
                )[-L_pro:]
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,model,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],masa_control=False):
        super(AttentionControlEdit, self).__init__()
        self.prompts = prompts
        self.model_dtype=model.unet.dtype
        self.tokenizer=model.tokenizer
        self.model_device=model.unet.device
        self.batch_size = len(self.prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(self.prompts, num_steps, cross_replace_steps,
                                                                            self.tokenizer).to(self.model_device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.masa_control = masa_control
        self.masa_start_step, self.masa_start_layer = 10, 10
        self.last_save = None
        self.self_layer = 0
        self.cross_layer = 0
        self.last_status = "up"
        self.rep_in_self_attn = True
        self.rep_in_cross_attn = True 
        self.layer_fusion = None
    

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int,model, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,masa_control=False):
        super(AttentionReplace, self).__init__(prompts, num_steps,model, cross_replace_steps, self_replace_steps, local_blend,masa_control=masa_control)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to(self.model_device)
        self.mapper=self.mapper.to(self.model_dtype)

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper[:att_replace.shape[0]]].permute(2, 0, 1, 3) 
        attn_replace = attn_base_replace * self.alphas[:att_replace.shape[0]] + att_replace * (1 - self.alphas[:att_replace.shape[0]])
        return attn_replace

    def __init__(self, prompts, num_steps: int, model,cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,masa_control=False):
        super(AttentionRefine, self).__init__(prompts, num_steps,model, cross_replace_steps, self_replace_steps, local_blend,masa_control=masa_control)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(self.prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.model_device), alphas.to(self.model_device)        
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, model,cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None,masa_control=False):
        super(AttentionReweight, self).__init__(prompts, num_steps,model, cross_replace_steps, self_replace_steps,
                                                local_blend,masa_control=masa_control)
        self.equalizer = equalizer.to(self.model_device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]],model):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, model.tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(prompts: List[str], model,num_ddim_steps, is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, blend_words=None, equilizer_params=None,masa_control=False,**kwargs) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words,model.tokenizer,model.unet.device,num_ddim_steps,**kwargs)
    if is_replace_controller: 
        controller = AttentionReplace(prompts, num_ddim_steps, model,cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, local_blend=lb,masa_control=masa_control)
    else: 
        controller = AttentionRefine(prompts, num_ddim_steps, model,cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, local_blend=lb,masa_control=masa_control)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"],model)
        controller = AttentionReweight(prompts, num_ddim_steps,model, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb,
                                       controller=controller,masa_control=masa_control)
    controller.num_ddim_steps = num_ddim_steps
    return controller
