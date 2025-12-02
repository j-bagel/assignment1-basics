from cs336_basics.optimizer import AdamW, learning_rate_scheduler
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data_loader import get_batch
import torch
from torch import nn
from typing import Type
import os


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
        self.checkpoints_folder = os.path.join(output_folder, "checkpoints")
        self.device = device

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.checkpoints_folder, exist_ok=True)

    def train(self):
        self.model.train()

        for step in range(self.max_steps):
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
            self.optimizer.step()

            if step % self.log_steps == 0:
                if self.db is not None:
                    self.db.log({
                        "train_loss": loss.item(),
                        "global_step": step,
                        "learning_rate": lr
                    })
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

            if step > 0 and step % self.save_steps == 0:
                checkpoint_path = os.path.join(self.checkpoints_folder, f"checkpoint_step_{step}.pt")
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)

