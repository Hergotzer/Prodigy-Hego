import math
from typing import TYPE_CHECKING, Any
import torch
import torch.optim

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
                ortho_strength=0.0, ortho_decay=0, ortho_normalize=False):
        
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
                        ortho_strength=ortho_strength,  # New parameter
                        ortho_decay=ortho_decay,        # New parameter
                        ortho_normalize=ortho_normalize) # New parameter

        # Initialize the global_step variable
        self.global_step = 0  # This will be incremented during training
        
        self.current_ortho_strength = 0
        #self.gradient_differences = []

        self.d0 = d0
        super().__init__(params, defaults)

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
            
    def orthograd(self, p, ortho_strength=1.0, global_step=0, ortho_decay=None, ortho_normalize=True):
        
        w = p.view(-1)  # Flatten weights
        g = p.grad.view(-1)  # Flatten gradients

        # Compute cosine decay for strength if decay is enabled
        if ortho_decay is not None and ortho_decay > 0:
            ortho_strength = ortho_strength * (1 + math.cos(math.pi * min(global_step / ortho_decay, 1))) / 2
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
        
        #diff = torch.norm(g - g_orth).item()
        #self.gradient_differences.append(diff)

        return g_final.view(p.grad.shape)  # Restore original shape

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
                grad = p.grad.data

                # Apply orthogonalization again if necessary
                if ortho_strength != 0.0:
                    grad = self.orthograd(p, ortho_strength=ortho_strength, global_step=self.global_step, ortho_decay=group['ortho_decay'], ortho_normalize=group['ortho_normalize'])

                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    updated = p.data.clone()
                    updated.addcdiv_(exp_avg, denom, value=-dlr)
                    self.smart_copy(p.data, updated, stochastic, True)
                else:
                    updated = p.data.clone()
                    updated.addcdiv_(grad, denom, value=-dlr * d)
                    self.smart_copy(p.data, updated, stochastic, True)

            group['k'] = k + 1

        self.global_step += 1
        #if ortho_strength != 0.0:
        #    print(f"ortho_strength of step = {self.current_ortho_strength:.6f}")
        
        #if self.gradient_differences:
        #    max_diff = max(self.gradient_differences)
        #    avg_diff = sum(self.gradient_differences) / len(self.gradient_differences)

        #    print(f"Max gradient difference: {max_diff:.6f}")
        #    print(f"Avg gradient difference: {avg_diff:.6f}")

            # Clear the array for the next step
        #    self.gradient_differences.clear()

        return loss