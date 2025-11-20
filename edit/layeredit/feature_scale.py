import numpy as np
from PIL import Image
import torch
import math
import torch.nn.functional as F
import cv2

def feature_center(M):  #
    total_sum = torch.sum(M).cuda()
    x_weighted_sum = torch.sum(torch.arange(M.shape[1])[:, None] * M)
    y_weighted_sum = torch.sum(torch.arange(M.shape[0])[None, :] * M)
    x_c = x_weighted_sum / total_sum
    y_c = y_weighted_sum / total_sum
    return x_c / M.shape[0], y_c / M.shape[1]


def mask_resize(mask, scale=1, pro="zoom"):
    x_c, y_c = feature_center(mask.cpu())
    if pro != "zoom":
        mask1 = mv_op(mask[None, None, :, :], pro, scale=scale)

    elif pro == "zoom":
        res = mask.shape[1]
        new_res = int(res * scale)
        mask_new = F.interpolate(
            mask[None, None, :, :], size=(new_res, new_res), mode="bilinear"
        ).cuda()  # [1,1,res*scale,res*scale]
        if scale > 1:
            x_S = int(x_c * (new_res - res))
            y_S = int(y_c * (new_res - res))

            mask1 = mask_new[:, :, x_S : x_S + res, y_S : y_S + res]
        elif scale < 1:
            x_S = int(x_c * (res - new_res))
            y_S = int(y_c * (res - new_res))

            mask1 = torch.zeros(mask[None, None, :, :].shape).cuda()
            mask1[:, :, x_S : x_S + new_res, y_S : y_S + new_res] = mask_new

    return mask1.squeeze(), x_c, y_c


def feature_resize_mask(attn, mask, scale=1):
    # attn: [B*h,res*res,D], mask :[res,res]
    D1, D2, D3 = attn.shape
    res = int(math.sqrt(D2))
    new_res = int(res * scale)
    attn = (
        attn.permute(0, 2, 1).reshape(D1, D3, D2).reshape(D1, D3, res, res)
    )  # [D1,D3,res,res]
    attn_new = F.interpolate(
        attn, size=(new_res, new_res), mode="bilinear"
    ).cuda()  # [D1,D3,res*scale,res*scale]
    mask_new = F.interpolate(
        mask[None, None, :, :], size=(new_res, new_res), mode="bilinear"
    ).cuda()  # [1,1,res*scale,res*scale]

    attn = attn.cuda()
    x_c, y_c = feature_center(mask.cpu())

    if scale > 1:
        x_S = int(x_c * (new_res - res))
        y_S = int(y_c * (new_res - res))
        attn = (
            attn_new[:, :, x_S : x_S + res, y_S : y_S + res]
            * mask_new[:, :, x_S : x_S + res, y_S : y_S + res]
            + (1 - mask_new[:, :, x_S : x_S + res, y_S : y_S + res]) * attn
        )
        mask1 = mask_new[:, :, x_S : x_S + res, y_S : y_S + res]

    elif scale < 1:
        x_S = int(x_c * (res - new_res))  # , int( x_c * res + new_res/2 )
        y_S = int(y_c * (res - new_res))  # , int( y_c * res + new_res/2 )
        attn[:, :, x_S : x_S + new_res, y_S : y_S + new_res] = (
            attn_new * mask_new
            + (1 - mask_new) * attn[:, :, x_S : x_S + new_res, y_S : y_S + new_res]
        )

        mask1 = torch.zeros(mask[None, None, :, :].shape).cuda()
        mask1[:, :, x_S : x_S + new_res, y_S : y_S + new_res] = mask_new

    attn = attn.reshape(D1, D3, -1).permute(0, 2, 1)  # 如作为attn 返回
    mask1 = mask1.flatten()[None, :, None]

    return attn, mask1


def mv_op(mp, op, scale=0.2, ones=False, flip=None):
    _, b, H, W = mp.shape
    if ones == False:
        new_mp = torch.zeros_like(mp)
    else:
        new_mp = torch.ones_like(mp)
    K = int(scale * W)
    if op == "right":
        new_mp[:, :, :, K:] = mp[:, :, :, 0 : W - K]
    elif op == "left":
        new_mp[:, :, :, 0 : W - K] = mp[:, :, :, K:]
    elif op == "down":
        new_mp[:, :, K:, :] = mp[:, :, 0 : W - K, :]
    elif op == "up":
        new_mp[:, :, 0 : W - K, :] = mp[:, :, K:, :]
    if flip is not None:
        new_mp = torch.flip(new_mp, dims=flip)

    return new_mp


def feature_move(attn, mask, move_scale=0, move_direction="left"):
    # attn: [B*h,res*res,D], mask :[res,res]
    D1, D2, D3 = attn.shape
    res = int(math.sqrt(D2))
    move_res = int(res * move_scale)
    attn = (
        attn.permute(0, 2, 1).reshape(D1, D3, D2).reshape(D1, D3, res, res)
    )  # [D1,D3,res,res]
    attn_new = mv_op(attn, move_direction, scale=move_scale)
    mask_new = mv_op(mask[None, None, :, :], move_direction, scale=move_scale)
    attn = attn_new * mask_new + (1 - mask_new) * attn
    attn = attn.reshape(D1, D3, -1).permute(0, 2, 1)
    mask_new = mask_new.flatten()[None, :, None]
    return attn, mask_new


def resize_image_with_mask(img, mask, scale=2):
    # input : img:(512, 512, 3) ; mask:(512, 512) （numpy）
    # output: img_blackboard:(512, 512, 3) ; mask_blackboard:(512, 512) （numpy）；
    if scale == 1:
        return img, mask, None
    img_blackboard = img.copy()  # canvas
    mask_blackboard = np.zeros_like(mask)

    M = cv2.moments(mask)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    scale_factor = [scale, scale]
    resized_img = cv2.resize(
        img, None, fx=scale_factor[0], fy=scale_factor[1], interpolation=cv2.INTER_AREA
    )
    resized_mask = cv2.resize(
        mask, None, fx=scale_factor[0], fy=scale_factor[1], interpolation=cv2.INTER_AREA
    )
    new_cx, new_cy = cx * scale_factor[0], cy * scale_factor[1]

    for y in range(resized_mask.shape[0]):
        for x in range(resized_mask.shape[1]):
            if (
                0 <= cy - (new_cy - y) < img.shape[0]
                and 0 <= cx - (new_cx - x) < img.shape[1]
            ):
                mask_blackboard[int(cy - (new_cy - y)), int(cx - (new_cx - x))] = (
                    resized_mask[y, x]
                )
                img_blackboard[int(cy - (new_cy - y)), int(cx - (new_cx - x))] = (
                    resized_img[y, x]
                )
    return img_blackboard, mask_blackboard, (cx, cy)
