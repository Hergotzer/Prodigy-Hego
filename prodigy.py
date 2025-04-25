import math
from typing import TYPE_CHECKING, Any
import torch
import torch.optim
import numpy as np
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class Prodigy(torch.optim.Optimizer):
    
    def __init__(self, params, lr=1.0,
                betas=(0.9, 0.999), beta3=None,
                eps=1e-8, weight_decay=0, decouple=True, 
                use_bias_correction=False, safeguard_warmup=False,
                d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                fsdp_in_use=False, slice_p=1, stochastic_rounding=False,
                ortho_strength=0.0, ortho_decay=0, ortho_normalize=False,
                ortho_decay_padding=0.0, dropout_prob=0.0, sia=False,
                sia_p=0.1, sia_range=(0.9, 1.1), sia_shift=True, a_noise=0.0,
                group_m=1.0, sia_smooth=None):
        
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

        # Adding the new arguments to the defaults dictionary
        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p, stochastic_rounding=stochastic_rounding,
                        ortho_strength=ortho_strength,
                        ortho_decay=ortho_decay,
                        ortho_normalize=ortho_normalize,
                        ortho_decay_padding=0.0,
                        dropout_prob=dropout_prob,
                        sia=sia, sia_p=sia_p, sia_range=sia_range, sia_shift=sia_shift,
                        a_noise=a_noise, group_m=group_m, sia_smooth=sia_smooth)

        self.global_step = 0  
        
        self.current_ortho_strength = 0

        self.d0 = d0
        super().__init__(params, defaults)
        
        group_m_setting = defaults.get('group_m', 1.0)

        if isinstance(group_m_setting, (list, tuple)):
            for i, g in enumerate(self.param_groups):
                g['group_multiplier'] = group_m_setting[i] if i < len(group_m_setting) else 1.0
        else:
            for g in self.param_groups:
                g['group_multiplier'] = group_m_setting

    @torch.no_grad()
    def copy_stochastic_(self, target, source):
        
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )
        result.add_(source.view(dtype=torch.int32))
        result.bitwise_and_(-65536)  # Mask lower 16 bits
        target.copy_(result.view(dtype=torch.float32))

    @torch.no_grad()
    def smart_copy(self, target, source, stochastic_rounding, smart_delete_source):
        
        if target is source:
            return

        if stochastic_rounding and target.dtype == torch.bfloat16 and source.dtype == torch.float32:
            self.copy_stochastic_(target, source)
        else:
            target.copy_(source)

        if smart_delete_source:
            del source
            
    def orthograd(self, p, ortho_strength=1.0, global_step=0, ortho_decay=None, ortho_decay_padding=0.0, ortho_normalize=True):
    
        w = p.view(-1)  # Flatten weights
        g = p.grad.view(-1)  # Flatten gradients

        # Compute cosine decay for strength if decay is enabled
        if ortho_decay is not None and ortho_decay > 0:
            padding_steps = int(ortho_decay * ortho_decay_padding)  # Round down to next integer
            decay_steps = ortho_decay - 2 * padding_steps

            if global_step < padding_steps:
                ortho_strength = ortho_strength  # Full strength in the padding phase
            elif global_step >= padding_steps + decay_steps:
                ortho_strength = 0.0  # Zero strength in the final padding phase
            else:
                # Cosine decay in the middle phase
                decay_progress = (global_step - padding_steps) / decay_steps
                ortho_strength = ortho_strength * (1 + math.cos(math.pi * decay_progress)) / 2

            self.current_ortho_strength = ortho_strength

        # Compute orthogonal component
        proj = torch.dot(w, g) / (torch.dot(w, w) + 1e-30)  # Projection factor
        g_orth = g - w * proj  # Remove parallel component

        # Optional normalization with safeguard
        if ortho_normalize:
            norm_g_orth = g_orth.norm(2)
            if norm_g_orth > 1e-30:  # Prevent division by zero
                g_orth = g_orth * (g.norm(2) / norm_g_orth)

        # Blend standard and orthogonal updates
        g_final = (1 - ortho_strength) * g + ortho_strength * g_orth
    
        return g_final.view(p.grad.shape)  # Restore original shape
        
    def compute_sia_multipliers(self, group, device):
        sia_p = group.get('sia_p', 0.05)
        sia_range = group.get('sia_range', (0.95, 1.05))
        params = group['params']

        if len(params) < 2 or sia_p <= 0:
            return torch.ones(len(params), device=device)  # No adjustment needed

        num_params = len(params)
        sample_size = max(2, int(num_params * sia_p))

        # === Sampling ===
        if sia_p >= 1.0:
            sample_indices = list(range(num_params))  # all parameters
        else:
            if group.get('sia_shift', False):
                stride = max(1, num_params // sample_size)
                offset = self.global_step % stride
                sample_indices = list(range(offset, num_params, stride))
                if 0 not in sample_indices:
                    sample_indices.insert(0, 0)
                if (num_params - 1) not in sample_indices:
                    sample_indices.append(num_params - 1)
            else:
                if sample_size >= num_params:
                    sample_indices = list(range(num_params))
                else:
                    sample_indices = [
                        round(i * (num_params - 1) / (sample_size - 1)) for i in range(sample_size)
                    ]

        # === Compute performance scores ===
        scores = []
        valid_indices = []

        for idx in sample_indices:
            p = params[idx]
            state = self.state.get(p, None)

            if p.grad is None or state is None or 'exp_avg_sq' not in state:
                continue

            p0 = state.get('p0', p.data)
            if p0.shape != p.data.shape:
                try:
                    p0 = p0.view_as(p.data)
                except RuntimeError:
                    p0 = p.data  # fallback

            diff = (p.data - p0).norm()  # stays as tensor
            stability = state['exp_avg_sq'].sqrt().mean() + 1e-8  # stays as tensor
            score = diff / stability

            scores.append(score)
            valid_indices.append(idx)

        if len(scores) < 2:
            return torch.ones(len(params), device=device)  # Not enough valid samples

        # === Dense Path: If all parameters were sampled, skip PCHIP and normalize directly ===
        if sia_p >= 1.0 and len(valid_indices) == len(params):
            scores_tensor = torch.stack(scores)  # All scores, GPU-native

            # Smoothing block
            if group.get('sia_smooth', None) is not None:
                alpha = group['sia_smooth']
                if '_sia_smoothed_scores' not in group or group['_sia_smoothed_scores'] is None:
                    group['_sia_smoothed_scores'] = scores_tensor.clone()
                else:
                    group['_sia_smoothed_scores'] = (
                        group['_sia_smoothed_scores'] * alpha +
                        scores_tensor * (1 - alpha)
                    )
                scores_tensor = group['_sia_smoothed_scores']  # use smoothed version for normalization

            s_min = scores_tensor.min()
            s_max = scores_tensor.max()

            if s_max == s_min:
                normalized = torch.full_like(scores_tensor, (sia_range[0] + sia_range[1]) / 2)
            else:
                normalized = ((scores_tensor - s_min) / (s_max - s_min)) * (sia_range[1] - sia_range[0]) + sia_range[0]

            return normalized.to(device=device, dtype=p.data.dtype)  # Return GPU tensor

        # === Sparse Sampling Path: Interpolation on CPU ===
        sorted_pairs = sorted(zip(valid_indices, scores))
        valid_indices, scores = zip(*sorted_pairs)
        valid_indices = np.array(valid_indices)
        scores = torch.stack(scores).to(torch.float32).cpu().numpy()  # Transfer scores to CPU safely

        interpolator = PchipInterpolator(valid_indices, scores, extrapolate=True)
        full_indices = np.arange(len(params))
        interpolated_scores = interpolator(full_indices)

        # Smoothing block
        if group.get('sia_smooth', None) is not None:
            alpha = group['sia_smooth']
            if '_sia_smoothed_scores' not in group or group['_sia_smoothed_scores'] is None:
                group['_sia_smoothed_scores'] = torch.tensor(interpolated_scores, dtype=torch.float32)
            else:
                prev = group['_sia_smoothed_scores']
                group['_sia_smoothed_scores'] = prev * alpha + torch.tensor(interpolated_scores, dtype=torch.float32) * (1 - alpha)
            interpolated_scores = group['_sia_smoothed_scores'].numpy()  # Overwrite for normalization

        s_min = interpolated_scores.min()
        s_max = interpolated_scores.max()

        if s_max == s_min:
            normalized = np.full_like(interpolated_scores, (sia_range[0] + sia_range[1]) / 2)
        else:
            normalized = ((interpolated_scores - s_min) / (s_max - s_min)) * (sia_range[1] - sia_range[0]) + sia_range[0]

        return torch.tensor(normalized, device=device, dtype=p.data.dtype)

    def step(self, closure=None):
        """Performs a single optimization step with optional orthogonal gradient update."""
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d * lr * bias_correction

        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']
            stochastic = group['stochastic_rounding']
            ortho_strength = group['ortho_strength']  # Assuming ortho_strength is a group parameter
            

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                
                
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data

                # Apply orthogonalization if ortho_strength is not 0
                #if ortho_strength != 0.0:
                #    grad = self.orthograd(p, ortho_strength=ortho_strength, global_step=self.global_step, ortho_decay=group['ortho_decay'], ortho_normalize=group['ortho_normalize'])

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)
    
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0

                    state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                    if p.any():
                        state['p0'] = p.flatten()[::slice_p].detach().clone()
                    else:
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(p.data).detach()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()
                    state['dropout_prob'] = group['dropout_prob']  # Assign tensor-specific dropout probability
                    
                    state['grad_magnitude'] = p.grad.norm() if p.grad is not None else torch.tensor(0.0, device=p.device)

                exp_avg_sq = state['exp_avg_sq']

                s = state['s']
                p0 = state['p0']
    
                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p]
                    d_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()

                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()
                    
            for group in self.param_groups:
                if group.get('sia', False):
                    device = group['params'][0].device if group['params'] else 'cpu'
                    group['_sia_multipliers'] = self.compute_sia_multipliers(group, device)
                    group['_sia_param_map'] = {p: idx for idx, p in enumerate(group['params'])}

        d_hat = d

        if d_denom == 0:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                torch.distributed.all_reduce(dist_tensor, op=torch.distributed.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Use the tensor-specific dropout probability
                if torch.rand(1).item() < state['dropout_prob']:
                    self.tensors_skipped += 1
                    continue  # Skip update for this tensor
                
                grad = p.grad.data

                # Apply orthogonalization again if necessary
                if ortho_strength != 0.0:
                    grad = self.orthograd(p, ortho_strength=ortho_strength, global_step=self.global_step, ortho_decay=group['ortho_decay'], ortho_normalize=group['ortho_normalize'])

                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq'] #XX

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)
                    
                sia_multiplier = 1.0
                if group.get('sia', False):
                    param_index = group['_sia_param_map'][p]
                    sia_multiplier = group['_sia_multipliers'][param_index].item()
                    
                group_multiplier = group.get('group_multiplier', 1.0)

                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    updated = p.data.clone()
                    updated.addcdiv_(exp_avg, denom, value=-dlr * sia_multiplier * group_multiplier)
                else:
                    updated = p.data.clone()
                    updated.addcdiv_(grad, denom, value=-dlr * d * sia_multiplier * group_multiplier)
                    
                if group.get('a_noise', 0) > 0:
                    noise = torch.randn_like(p.data) * dlr * d * group['a_noise']
                    updated.add_(noise)

                self.smart_copy(p.data, updated, stochastic, True)
                

            group['k'] = k + 1

        self.global_step += 1      

        return loss
