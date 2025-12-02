import torch
import os
import numpy as np
import json
from cs336_basics.data_loader import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.models import TransformerLM


def main():
    num_batches = 200
    batch_size = 32
    context_length = 256
    device = 'mps:0'
    loss_fn = cross_entropy

    np.random.seed(42)

    # data dir
    eval_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/tokenized_tinystories/valid.npy')
    eval_dataset = np.load(eval_path, mmap_mode='r')

    # model dir
    output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'sample_model')

    # Load model
    print("Loading model...")
    model = TransformerLM(
        vocab_size=10000,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        max_seq_len=context_length,
        theta=10000,
        device=device,
        dtype=torch.float32
    )
    
    # Load saved weights
    model_path = os.path.join(output_folder, 'model.pt')
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove _orig_mod. prefix if present (from torch.compile)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Evaluate
    print(f"Evaluating on {num_batches} batches ({num_batches * batch_size} samples)...")
    loss_list = []
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for i in range(num_batches):
            data, target = get_batch(
                dataset=eval_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            output = model(data)
            loss = loss_fn(output, target)
            loss_list.append(loss.item())
            
            if (i + 1) % 20 == 0:
                print(f"  Batch {i+1}/{num_batches}, Current loss: {loss.item():.4f}")

    eval_loss = np.mean(loss_list)
    eval_sem = np.std(loss_list) / np.sqrt(len(loss_list))  # Standard error of the mean
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Mean Loss: {eval_loss:.4f} ± {eval_sem:.4f}")
    print(f"  Perplexity: {np.exp(eval_loss):.4f}")
    print(f"{'='*50}\n")
    
    # Save results to file
    results = {
        'eval_loss': float(eval_loss),
        'eval_stderr': float(eval_sem),  # Standard error of the mean
        'perplexity': float(np.exp(eval_loss)),
        'num_batches': num_batches,
        'batch_size': batch_size,
        'context_length': context_length
    }
    
    results_path = os.path.join(output_folder, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

