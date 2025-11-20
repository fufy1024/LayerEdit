import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from layeredit.feature_scale import feature_resize_mask,feature_move

@torch.no_grad()
def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        self.counter = 0 
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, Layer_S=1): 
            x = hidden_states.clone() 
            context = encoder_hidden_states
            is_cross = context is not None
            b, i, j = x.shape 
            H = W = int(math.sqrt(i))
            controller.h = self.heads
            x_k, x_q = x.clone(), x.clone()
            # Time-dependent Region Removing
            if (controller.layer_fusion.K_remove_mask is not None or controller.layer_fusion.Q_remove_mask is not None) and self.counter <= 40:
                x_k, x_q = x_k.reshape(b, H, W, j), x_q.reshape(b, H, W, j)
                K_conflict_mask = controller.layer_fusion.K_remove_mask
                Q_conflict_mask = controller.layer_fusion.Q_remove_mask

                K_conflict_mask = F.interpolate((1 - K_conflict_mask).to(dtype=torch.float32).clone(), size=(H, W), mode='bilinear').reshape(-1, H, W).unsqueeze(-1)
                Q_conflict_mask = F.interpolate((1 - Q_conflict_mask).to(dtype=torch.float32).clone(), size=(H, W), mode='bilinear').reshape(-1, H, W).unsqueeze(-1)
       
                x_k[int(b/2)+Layer_S:, :, :] = (x_k[int(b/2)+Layer_S:, :, :] * K_conflict_mask) 
                x_q[int(b/2)+Layer_S:, :, :] = (x_q[int(b/2)+Layer_S:, :, :] * Q_conflict_mask) 
                x_k, x_q = x_k.reshape(b, i, j), x_q.reshape(b, i, j)
            

            if is_cross: 
                q = self.to_q(x_q) 
                k = self.to_k(context)
                v = self.to_v(context)         
            else:
                q = self.to_q(x_q) 
                k = self.to_k(x_k)  
                v = self.to_v(hidden_states) 
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            if hasattr(controller, 'count_layers'):
                controller.count_layers(place_in_unet,is_cross)

            if controller.masa_control and not is_cross: # masa control
                q,k,v=controller.replace_self_attention_kv(q, k, v, self.heads) 
        
            sim = torch.einsum("b i d, b j d -> b i j", q.clone(), k.clone()) * self.scale  # [b*h, res*res, D1] 
            attn = sim.softmax(dim=-1)
            attn  = controller(attn , is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v) # [b*h, res*res, D2]
            out = self.batch_to_head_dim(out)  # [b, res*res, D2*h] 

            # background preserve 
            if self.counter <= 40: 
                object_mask_all = F.interpolate(controller.layer_fusion.object_mask_all[None,None,:,:], size=(H, W), mode='bilinear').cuda().reshape(1,-1,1)
                #cond
                out[b//2+Layer_S:] = out[b//2+Layer_S:]*object_mask_all + (out[b//2:b//2+1].repeat(b//2-Layer_S,1,1))*(1-object_mask_all)
                #uncond
                out[Layer_S:b//2] = out[Layer_S:b//2]*object_mask_all + (out[0:1].repeat(b//2-Layer_S,1,1))*(1-object_mask_all)

            # use Transparency alpha for fusion:
            alpha= F.interpolate(controller.layer_fusion.alpha, size=(H, W), mode='bilinear').reshape(-1,1,H*W,1)  # [N_o, 1, res*res, 1]
            out=out.requires_grad_(True)
            # for Geometric Structural Editing
            if controller.layer_fusion.move_pro:
                alpha = alpha.squeeze(-1).squeeze(1).reshape(-1,H,W)            
                for ri_ , move_direction in enumerate(controller.layer_fusion.move_direction): 
                    if move_direction!="":
                        move_scale = controller.layer_fusion.move_scale[ri_] 
                        #cond
                        out_new, mask_new = feature_move(out[int(b//2)+Layer_S+ri_:int(b//2)+Layer_S+ri_+1].clone(),alpha[ri_],move_scale=move_scale,move_direction=move_direction)
                        out[-1:] =  out[-1] * (1-mask_new) + mask_new*out_new
                        #uncond
                        out_new, mask_new = feature_move(out[Layer_S+ri_:Layer_S+ri_+1].clone(),alpha[ri_],move_scale=move_scale,move_direction=move_direction)
                        out[b//2-1:b//2] =  out[b//2-1:b//2] * (1-mask_new) + mask_new*out_new
                        alpha[ri_]=torch.zeros(alpha[ri_].shape).to(alpha.device)
                alpha = alpha.reshape(-1,H*W).unsqueeze(-1).unsqueeze(1)
            
            if controller.layer_fusion.resize_pro and self.counter>=0: 
                alpha = alpha.squeeze(-1).squeeze(1).reshape(-1,H,W)
                for ri_ , resize_scale in enumerate(controller.layer_fusion.resize_scale):
                    if resize_scale!=1:
                        #cond
                        out_new, mask_new = feature_resize_mask(out[int(b//2)+Layer_S+ri_:int(b//2)+Layer_S+ri_+1].clone(),alpha[ri_],scale=resize_scale)
                        out[-1:] =  out[-1:] * (1-mask_new) + mask_new*out_new
                        #uncond
                        out_new, mask_new = feature_resize_mask(out[Layer_S+ri_:Layer_S+ri_+1].clone(),alpha[ri_],scale=resize_scale)
                        out[int(b//2)-1:int(b//2)] =  out[int(b//2)-1:int(b//2)] * (1-mask_new) + mask_new*out_new
                        alpha[ri_]=torch.zeros(alpha[ri_].shape).to(alpha.device)
                alpha = alpha.reshape(-1,H*W).unsqueeze(-1).unsqueeze(1)
            
            # layerfusion
            out[-1:] = out[-1:]* (1-torch.sum(alpha,0))  + torch.sum( out[b//2+Layer_S:-1] * alpha.squeeze(1), 0)[None,:,:] 
            out[b//2-1:b//2] = out[b//2-1:b//2]* (1-torch.sum(alpha,0))  + torch.sum( out[Layer_S:b//2-1] * alpha.squeeze(1), 0)[None,:,:] 

            out=out.requires_grad_(False)
    
            self.counter += 1
            return to_out(out)
        
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count
