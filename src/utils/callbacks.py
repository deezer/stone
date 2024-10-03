import os
from typing import Any, Callable, Dict, Tuple

import gin  # type: ignore
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore


TORCH_LOAD_KEYS = [
    "stone",
    "optimizer",
    "loss_fn",
]

@gin.configurable  # type: ignore
def get_writer(base_path: str) -> Any:
    base_path = base_path.replace("models", "tensorboad_logs")
    print("\t save_tensorflow_dir: {}".format(base_path))
    return SummaryWriter(base_path)  # type: ignore


def save_fn(save_dict: Dict[str, Any], model: Any, save_dir: str) -> None:
    save_dict["stone"] = model.stone.state_dict()
    save_dict["optimizer"] = model.optimizer.state_dict()
    save_dict["loss_fn"] = model.loss_fn.state_dict()
    torch.save(save_dict, save_dir)
    return


@gin.configurable  # type: ignore
def restart_from_checkpoint(
    model,
    save_dict,
    ckpt_path,
    ckp_name="last_iter.pt",
):
    """
    Re-start from checkpoint
    """
    ckpt_path = os.path.join(ckpt_path, ckp_name)
    if os.path.isfile(ckpt_path):
        print("Found checkpoint at {}".format(ckpt_path))
        # open checkpoint file
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        for key in save_dict.keys():
            if key in TORCH_LOAD_KEYS:
                attr = getattr(model, key)
                attr.load_state_dict(checkpoint[key])
            if key != "args" and key != "audio":  # and key != "gin_info":
                save_dict[key] = checkpoint[key]
    save_fn(save_dict, model, ckpt_path)
    return model, save_dict


@gin.configurable  # type: ignore
def add_audio_tensorboard(
    writer: Any,
    batch: torch.Tensor,
    type_set: str,
    epoch: int,
    sr: int=22050
) -> None:
    for i in range(10):
        writer.add_audio(
            "{}/{}/{}/seg1.wav".format(type_set, epoch, i),
            batch[i, :, 0:1],
            global_step=epoch,
            sample_rate=sr,
        )
        writer.add_audio(
            "{}/{}/{}/seg2.wav".format(type_set, epoch, i),
            batch[i, :, 1:2],
            global_step=epoch,
            sample_rate=sr,
        )
    return


def add_losses_tensorboard(writer: Any, progress_bar: Any, epoch: int) -> float:
    val_loss = -1.0
    for key in ["train_loss", "val_loss"]:
        loss = progress_bar._values[key][0] / progress_bar._values[key][1]
        split, key = key.split("_")
        writer.add_scalar("{}/{}".format(split, key), loss, epoch)
        if split == "val":
            val_loss = loss
    return val_loss


def add_schdules_tensorboard(
    model: Any,
    writer: Any,
    epoch: int,
    approach: str,
) -> None:
    gs = model.n_steps * epoch

    writer.add_scalar("Schedules/Learning_rate", model.lr_schedule[gs], epoch)
    writer.add_scalar("Schedules/Weight_decay", model.wd_schedule[gs], epoch)

    return


def get_idx(approach: str, batch: Any) -> torch.Tensor:
    if approach == "supervised":
        idx = torch.randperm(len(batch["audio"]))
    else:
        idx = torch.randperm(batch.shape[1])
    return idx


def add_callbacks(
    model: Any,
    sr: int,
    writer: Any,
    approach: str,
    batch: Any,
    epoch: int,
    text: str,
    add_audio: bool = False
) -> None:
    idx = get_idx(approach, batch)
    if text == "train":
        add_schdules_tensorboard(model, writer, epoch, approach)
    if add_audio:
        for i, data in enumerate(batch):
            add_audio_tensorboard(
                writer, data, "{}/{}".format(text, i), sr, epoch
            )
    return


def stop_condition(x: Dict[str, Any]) -> bool:
    return any([np.isnan(v) or np.isinf(v) for v in x.values()])


@gin.configurable
class NaNLoopCallback:
    def __init__(self, save_dir: str, patience: int = 30):
        self.patience = patience
        self.counter = 0
        self.save_dir = save_dir
        self.stop = False

    def __call__(
        self,
        is_nan: bool,
        model_wrapper: Any,
        save_dict: Dict[str, Any],
        ds_train: Any,
        n_steps: int,
        epoch: int,
    ) -> Tuple[Any, Any, Dict[str, Any], int]:

        if not is_nan:
            self.counter = 0
            epoch += 1
            self.stop = False
        elif is_nan:
            print("\nLoss is NaN restarting the loop")
            model_wrapper, ds_train, save_dict = self.load_checkpoint(
                model_wrapper, save_dict, ds_train, n_steps
            )
            if self.counter < self.patience:
                self.counter += 1
            else:
                self.stop = True
                print(
                    f"""\nLoss has been NaN for {self.patience:.1f} epochs
                    \nStopping the training!"""
                )
        return model_wrapper, ds_train, save_dict, epoch

    def load_checkpoint(
        self,
        model_wrapper: Any,
        save_dict: Dict[str, Any],
        ds_train: Any,
        n_steps: int,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Loading preivous checkpoint."""
        print("\nModel outputs NaN for {} epochs!".format(self.counter))
        model_wrapper, save_dict = restart_from_checkpoint(
            model_wrapper,
            save_dict,
            self.save_dir,
            ckp_name="best_model.pt",
        )
        ds_train.dur_index = int(save_dict["epoch"] * n_steps)
        return model_wrapper, ds_train, save_dict


@gin.configurable
class EarlyStoppingCallback:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Adapted from https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(
        self,
        save_dir: str,
        best_score: float = np.Inf,
        patience: int = 300,
        verbose: bool = True,
        delta: int = 0,
        save_until_epoch: int = 0,
        save_fn: Callable[[Dict[str, Any], Any, str], None] = save_fn,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.save_until_epoch = save_until_epoch
        self.best_score = best_score
        self.stop = False
        self.delta = delta
        self.save_dir = os.path.join(save_dir, "best_model.pt")
        self.save_fn = save_fn

    def __call__(
        self, val_loss: float, model_wrapper: Any, save_dict: Dict[str, Any], epoch: int
    ) -> None:
        if self.best_score == np.Inf or epoch < self.save_until_epoch:
            self.save_checkpoint(model_wrapper, val_loss, save_dict)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            print(
                f"""\n Val loss did not decrease from {self.best_score}
                \n EarlyStopping counter: {self.counter} out of {self.patience}"""
            )
            if self.counter >= self.patience:
                self.stop = True
                print(
                    f"""\nLoss has not increased from {self.best_score:.6f}
                    after {self.patience:.1f}"""
                )
        else:
            self.save_checkpoint(model_wrapper, val_loss, save_dict)
            self.counter = 0
        return

    def save_checkpoint(
        self, model_wrapper: Any, val_loss: float, save_dict: Dict[str, Any]
    ) -> None:
        """Saves model when validation loss decrease."""
        if not np.isnan(val_loss):
            if self.verbose:
                print(
                    f"\nValidation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model!"
                )
            self.best_score = val_loss
            self.save_fn(save_dict, model_wrapper, self.save_dir)
        return
