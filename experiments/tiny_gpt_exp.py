import math
import os
import numpy as np
import argparse
import torch
import time

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.trainer import MyTrainer
from cs336_basics.models import TransformerLM
from cs336_basics.optimizer import AdamW

import wandb


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_lr', type=float, default=2e-3)
    parser.add_argument('--min_lr', type=float, default=2e-5)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.95)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--total_tokens', type=int, default=int(4e7))
    args = parser.parse_args()

    # data dir
    train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/tokenized_tinystories/train.npy')
    train_dataset = np.load(train_path, mmap_mode='r')

    # get a few key args
    device = 'mps:0'
    max_seq_len = 256
    max_steps = math.floor(args.total_tokens / args.batch_size / max_seq_len)

    model = TransformerLM(
        vocab_size=10000,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        max_seq_len=max_seq_len,
        theta=10000,
        device=device,
        dtype=torch.float32
    )
    model = torch.compile(model, backend="aot_eager")
    # sanity check
    expected_device = torch.device(device)
    for param in model.parameters():
        assert param.device == expected_device, f"Found param on {param.device}, expected {expected_device}"
    print("Device sanity check passed!")

    # init wandb
    wandb.init(
        project="Tiny GPT with Tinystories",
        config=vars(args)
    )

    # output folder for MyTrainer
    output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/outputs', wandb.run.name)
    os.makedirs(output_folder, exist_ok=True)

    # optimizer
    optimizer = AdamW
    optimizer_args = {
        'params': model.parameters(),
        'betas': (args.beta_1, args.beta_2),
        'weight_decay': args.weight_decay,
    }

    # lr scheduler
    lr_scheduler_args = {
        'max_learning_rate': args.max_lr,
        'min_learning_rate': args.min_lr,
        'warmup_iters': math.floor(args.warmup_ratio * max_steps),
        'cosine_cycle_iters': max_steps,
    }

    # run training
    trainer = MyTrainer(
        model=model,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        lr_scheduler_args=lr_scheduler_args,
        loss_fn=cross_entropy,
        train_dataset=train_dataset,
        max_steps=max_steps,
        log_steps=1,
        save_steps=500,
        batch_size=args.batch_size,
        context_length=max_seq_len,
        output_folder=output_folder,
        device=device,
        db=wandb
    )
    trainer.train()
    
    # Save final model (unwrap if compiled)
    # torch.compile wraps the model, so we need to access _orig_mod to get clean state_dict
    if hasattr(model, '_orig_mod'):
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, os.path.join(output_folder, "model.pt"))
    print(f"Model saved to {os.path.join(output_folder, 'model.pt')}")

    print(f"Training finished in {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()

# restful-galaxy-12 Mean Loss: 1.8179 ± 0.0044
