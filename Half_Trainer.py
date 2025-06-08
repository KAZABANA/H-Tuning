import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from gradient_pruning.pruning_utils import fast_random_mask_like
import torch.quantization
def set_eval_mode(model):
    for module in model.modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            module.eval()
        else:
            module.train()

def set_eval_mode_bn(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            module.eval()

class HalfTrainer():
    def __init__(self, optimizer, lr_scheduler,EWC_worker, args):
        self.optimizer = optimizer
        self.sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.gradient_sparsity = args.gradient_sparsity
        self.lr_scheduler = lr_scheduler
        self.sparse_grad_random_seed = np.random.randint(1000000000)
        self.EWC_worker = EWC_worker
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)## maybe sample a z along the grad computed at minibatch
            # if name == 'transformer_layers.5.self_attention.c_attn.lora_B':
            #     print(z)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.

    def zo_forward(self, model, inputs, labels, train_cls=False):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        #model.eval()
        # model.encoder_layers.eval()
        # model.transformer_layers.eval()
        set_eval_mode(model.encoder_layers)
        set_eval_mode(model.transformer_layers)  ## You can choose to freeze bn too.
        if train_cls:
            with torch.no_grad():
                outputs = model.feature_extraction(inputs)
            for layer in model.classifier:
                outputs = layer(outputs)
            loss = F.binary_cross_entropy_with_logits(outputs, labels) #+ ewc_loss
            loss.backward()
            return loss.detach()
        else:
            with torch.inference_mode():
                outputs = model(inputs)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)#+self.args.EWC_coef * self.EWC_worker.penalty(model)
            return loss.detach()
    def get_grad_sparsity_by_name(self, name):
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]

    def ho_get_all_grad(self, model, inputs, labels):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        self.named_parameters_to_optim_nozero = []
        for name, param in model.named_parameters():
            if param.requires_grad and name.find('classifier') == -1:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.
            elif param.requires_grad and name.find('classifier') > -1:
                self.named_parameters_to_optim_nozero.append((name, param))
                param.grad = None

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs, labels)
        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs, labels, train_cls=True)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs, labels, train_cls=True)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            ## update other para
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q

        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def ho_step(self, model, inputs, labels):
        self.optimizer.zero_grad(set_to_none=True)
        _ = self.ho_get_all_grad(model, inputs, labels)

        zo_grad = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                zo_grad.update({n: deepcopy(p.grad)})

        self.optimizer.zero_grad(set_to_none=True)
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = None

        model.train()
        set_eval_mode_bn(model)
        inputs = inputs[0:self.args.bp_batch,:,:,:]
        labels = labels[0:self.args.bp_batch,:]
        outputs = model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        if self.args.no_grad_correct:
            for name, param in model.named_parameters():
                if param.requires_grad and name.find('classifier') > -1:
                    param.grad = zo_grad[name]
        else:
            for name, param in model.named_parameters():
                if param.requires_grad and name.find('classifier') == -1:
                    param.grad = self.args.coef * param.grad + (1-self.args.coef) * torch.norm(param.grad) * zo_grad[name] / (torch.norm(zo_grad[name])+1e-6)
                elif param.requires_grad and name.find('classifier') > -1:
                    param.grad = zo_grad[name]
        self.optimizer.step()
        self.lr_scheduler.step()

    def tune_with_pure_bp(self, model, inputs, labels):
        outputs = model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)  # + ewc_loss
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

