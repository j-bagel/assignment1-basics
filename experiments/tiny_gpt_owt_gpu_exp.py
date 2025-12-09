import math
import os
import numpy as np
import argparse
import torch
import time

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.trainer import MyTrainer
from cs336_basics.models import TransformerLM, TransformerLMWithTiedWeights
from cs336_basics.optimizer import AdamW

import wandb


def main():
    # Enable TF32 for matmul and cudnn benchmark
    torch.set_float32_matmul_precision('high')

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=96)  # 128 will blow H100 (80GB)
    parser.add_argument('--max_lr', type=float, default=2e-3)
    parser.add_argument('--min_lr', type=float, default=2e-5)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.95)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--total_tokens', type=int, default=int(3e8))
    args = parser.parse_args()

    # data dir
    train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/tokenized_owt/train.npy')
    train_dataset = np.load(train_path, mmap_mode='r')

    # get a few key args
    device = 'cuda:0'
    max_seq_len = 512
    max_steps = math.floor(args.total_tokens / args.batch_size / max_seq_len)
    print(f"---- Total number of optimization steps: {max_steps} ----\n")

    model = TransformerLM(
        vocab_size=32000,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        max_seq_len=max_seq_len,
        theta=10000,
        device=device,
        dtype=torch.float32
    )
    model = torch.compile(model)
    # sanity check
    expected_device = torch.device(device)
    for param in model.parameters():
        assert param.device == expected_device, f"Found param on {param.device}, expected {expected_device}"
    print("Device sanity check passed!")

    # init wandb
    lr_name = f"{args.max_lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    warmup_name = f"{args.warmup_steps:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    wandb.init(
        project="Tiny GPT with OWT on H100, context len 512, batch size",
        config=vars(args),
        name=f"batch_{args.batch_size}__lr_{lr_name}__warmup_{warmup_name}"
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
        'warmup_iters': math.floor(args.warmup_steps),
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
        save_steps=5000,
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

    # Evaluation
    print("\n---- Starting Evaluation ----")
    from cs336_basics.data_loader import get_batch
    
    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/tokenized_owt/valid.npy')
    valid_dataset = np.load(valid_path, mmap_mode='r')
    
    model.eval()
    eval_batch_size = 32
    eval_num_batches = 200
    losses = []
    
    with torch.no_grad():
        for _ in range(eval_num_batches):
            data, target = get_batch(
                dataset=valid_dataset,
                batch_size=eval_batch_size,
                context_length=max_seq_len,
                device=device
            )
            output = model(data)
            loss = cross_entropy(output, target)
            losses.append(loss.item())
    
    eval_loss = np.mean(losses)
    total_minutes = (time.time() - start_time) / 60
    
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Total time: {total_minutes:.2f} minutes")
    
    wandb.log({
        "eval_loss": eval_loss,
        "total_minutes": total_minutes
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()
