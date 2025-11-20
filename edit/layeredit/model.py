import numpy as np
import torch
from torchvision.utils import save_image
from skimage.morphology import dilation, erosion
import torch.nn.functional as nnf
from layeredit.feature_scale import feature_resize_mask, feature_move, mask_resize

DIM = 128
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def generate_mask_Bernoulli(size, r=1):
    """
    generate Bernoulli mask with probability r
    """
    mask = np.random.binomial(1, r, size=(size, size))
    return torch.tensor(mask).to(torch.float16).to(device)


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def alpha_loss_inlatent_pro(alpha, x_t, object_mask, object_mask_all):
    L_x_t_1 = torch.norm(
        x_t[-1:] * (1 - torch.sum(alpha, 0))
        + torch.sum(x_t[1:-1] * alpha, 0)
        - x_t[0:1],
        p=2,
    )
    L_pos = torch.norm(torch.relu(-alpha * object_mask), p=2)
    L_sum = torch.norm(torch.sum(alpha, 0) - object_mask_all, p=2)
    alpha_loss = L_x_t_1 + L_pos + L_sum
    return alpha_loss


def dilated_eroded_mask(mask_s, core=3):
    try:
        mask = np.array(mask_s.squeeze().cpu())
    except:
        mask = mask_s.squeeze()
    selem = np.ones((core, core))
    dilated_mask = dilation(mask, selem)
    eroded_mask = erosion(mask, selem)
    try:
        return torch.as_tensor(
            dilated_mask, dtype=mask_s.dtype, device=mask_s.device
        ).view(mask_s.shape), torch.as_tensor(
            eroded_mask, dtype=mask_s.dtype, device=mask_s.device
        ).view(
            mask_s.shape
        )
    except:
        return torch.as_tensor(dilated_mask, dtype=torch.float32, device=device).view(
            mask_s.shape
        ), torch.as_tensor(eroded_mask, dtype=torch.float32, device=device).view(
            mask_s.shape
        )


class LayerFusion:
    def __call__(self, x_t):
        # Time-dependent Region Removing
        self.curr_K_remove, self.curr_Q_remove = (
            self.K_remove_rate[self.counter],
            self.Q_remove_rate[self.counter],
        )
        self.K_remove_mask_Bernoulli = generate_mask_Bernoulli(
            DIM, r=self.curr_K_remove
        )
        self.Q_remove_mask_Bernoulli = generate_mask_Bernoulli(
            DIM, r=self.curr_Q_remove
        )
        self.K_remove_mask = (
            self.remove_mask * self.K_remove_mask_Bernoulli[None, None, :, :]
        )  # [B-1,1,DIM,DIM]
        self.Q_remove_mask = (
            self.remove_mask * self.Q_remove_mask_Bernoulli[None, None, :, :]
        )  # [B-1,1,DIM,DIM]

        # Transparency alpha update
        self.alpha = self.alpha.to(x_t.dtype)
        torch.set_grad_enabled(True)
        alpha_loss = alpha_loss_inlatent_pro(
            self.alpha.requires_grad_(True),
            x_t.requires_grad_(True),
            self.object_mask.requires_grad_(True),
            self.object_mask_all.requires_grad_(True),
        )
        alpha_grad = (
            torch.autograd.grad(
                alpha_loss * self.sg_loss_rescale, self.alpha, allow_unused=True
            )[0]
            / self.sg_loss_rescale
        )
        torch.set_grad_enabled(False)
        self.alpha = self.alpha - alpha_grad.clone()
        self.alpha *= self.object_mask

        self.counter += 1
        return x_t

    def __init__(
        self,
        remove_mask,
        prompts=None,
        tokenizer=None,
        object_mask=None,
        object_mask_all=None,
    ):
        self.counter = 0
        self.remove_mask = remove_mask
        self.K_remove_mask, self.Q_remove_mask = remove_mask, remove_mask
        self.object_mask = object_mask
        self.object_mask_all = object_mask_all
        self.remove_mask_rate = generate_mask_Bernoulli(DIM, r=1)

        self.alpha = self.object_mask.clone()
        self.sg_loss_rescale = 1000

        self.masa_start_step, self.masa_start_layer = 30, 10

        self.tokenizer = tokenizer
