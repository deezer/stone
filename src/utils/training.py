from typing import Any, Dict, List, Tuple

import gin  # type: ignore
import torch


@gin.configurable  # type: ignore
def clip_gradients(model: Any, clip_grad: float = 3.0) -> Any:
    """
    Clip_grad: Maximal parameter gradient norm if using gradient clipping.
    Clipping with norm .3 ~ 1.0 can help optimization for larger architectures.
    0 for disabling.
    """
    if clip_grad != 0:
        for _, p in model.named_parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data)
                param_norm = p.grad.data.norm(2)
                clip_coef = clip_grad / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
                # if torch.any(torch.isinf(p.data)) or torch.any(torch.isnan(p.data)):
                #     print("weight", p.data)
                # if torch.any(torch.isinf(p.grad.data)) or torch.any(
                #     torch.isnan(p.grad.data)
                # ):
                #     print("grad", p.grad.data)
    return model


def get_params_groups(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Regularizers allow penalties on layer parameters during optimization.
    These penalties are summed into the loss function that the network optimizes.
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            # we do not regularize biases nor Norm parameters
            not_regularized.append(param)
        else:
            # only the kernels
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def update_optimizer(
    optimizer: torch.optim.Optimizer, lr_step: float, wd_step: float
) -> torch.optim.Optimizer:
    # update schedules
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_step
        if i == 0:  # only the first group is regularized
            param_group["weight_decay"] = wd_step
    return optimizer


@gin.configurable  # type: ignore
def get_optimizer(
    model,
    optimizer_type: str = "adamw",
    weight_decay: float = 0.2,
    betas: Tuple[float, float] = (0.9, 0.98),
) -> torch.optim.Optimizer:
    params_groups = get_params_groups(model)
    assert optimizer_type in ["adamw", "sgd", "adam", "adagrad", "rmsprop"]
    if optimizer_type == "adam":
        return torch.optim.Adam(params_groups)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            params_groups, betas=betas, weight_decay=weight_decay
        )  # to use with ViTs
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif optimizer_type == "adagrad":
        return torch.optim.Adagrad(params_groups)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(params_groups)

def cleanup() -> None:
    dist.destroy_process_group()
