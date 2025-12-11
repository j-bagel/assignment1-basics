from cs336_basics.optimizer import AdamW, learning_rate_scheduler
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data_loader import get_batch
from cs336_basics.nn_utils import gradient_clipping
import torch
from torch import nn
from typing import Type
import os
from tqdm import tqdm


class MyTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Type[torch.optim.Optimizer],
            optimizer_args: dict,
            lr_scheduler_args: dict | None,
            loss_fn,
            train_dataset,
            max_steps: int,
            log_steps: int,
            save_steps: int,
            batch_size: int,
            context_length: int,
            output_folder: str | os.PathLike,
            max_grad_l2_norm: float | None = None,
            device=None,
            db=None
    ):
        """
        db: an wandb instance
        """
        self.model = model
        self.optimizer = optimizer(**optimizer_args)
        self.lr_scheduler_args = lr_scheduler_args
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.db = db
        self.batch_size = batch_size
        self.context_length = context_length
        self.output_folder = output_folder
        self.max_grad_l2_norm = max_grad_l2_norm
        self.checkpoints_folder = os.path.join(output_folder, "checkpoints")
        self.device = device

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.checkpoints_folder, exist_ok=True)

    def train(self):
        self.model.train()

        pbar = tqdm(range(self.max_steps), desc="Training", unit="step")
        for step in pbar:
            data, target = get_batch(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                context_length=self.context_length,
                device=self.device
            )

            if self.lr_scheduler_args:
                lr = learning_rate_scheduler(it=step, **self.lr_scheduler_args)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
            else:
                # just for tracking
                lr = None
                for group in self.optimizer.param_groups:
                    lr =  group["lr"]
                    break

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()

            # Compute grad norm BEFORE optimizer step
            if self.max_grad_l2_norm is not None:
                grad_norm = gradient_clipping(self.model.parameters(), max_l2_norm=self.max_grad_l2_norm)
            else:
                grad_norm = gradient_clipping(self.model.parameters(), max_l2_norm=float("inf"))

            self.optimizer.step()

            # Update tqdm progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}" if lr else "N/A")

            if step % self.log_steps == 0:
                if self.db is not None:
                    self.db.log({
                        "train_loss": loss.item(),
                        "grad_norm": grad_norm,
                        "global_step": step,
                        "learning_rate": lr
                    })

            if step > 0 and step % self.save_steps == 0:
                checkpoint_path = os.path.join(self.checkpoints_folder, f"checkpoint_step_{step}.pt")
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)

