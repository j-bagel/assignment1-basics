import torch
import os
from cs336_basics.models import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def main():
    device = 'mps:0'
    context_length = 512
    max_tokens = 512

    # Load tokenizer
    print("Loading tokenizer...")
    tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_owt')
    vocab_path = os.path.join(tok_dir, 'vocab.pkl')
    merges_path = os.path.join(tok_dir, 'merges.pkl')
    special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

    # Load model
    print("Loading model...")
    output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/outputs', 'final_2')

    d_model = 768
    layers = 8

    model = TransformerLM(
        vocab_size=32000,
        num_layers=layers,
        d_model=d_model,
        num_heads=16,
        d_ff=int(d_model / 3 * 8 / 64) * 64,
        max_seq_len=context_length,
        theta=10000,
        device=device,
        dtype=torch.float32
    )
    
    model_path = os.path.join(output_folder, 'model.pt')
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}\n")

    # Prompts (different lengths, so we process batch_size=1)
    prompts = [
        "Once upon a time, there was a little beetle named Bob.",
        "Fire department medics responded and pronounced the motorcyclist deceased at the scene. King County Medical Examiner’s Office personnel responded and took custody of the deceased man.",
        "By just about any measure, Chicago dominates its opponents when Keith is on the ice versus when he's off — possession, high-quality chances, goals, etc."
    ]

    # Different sampling configurations to try
    configs = [
        {"temperature": 0.0, "top_p": 1.0},
        {"temperature": 0.7, "top_p": 1.0},
        {"temperature": 1.0, "top_p": 1.0},
        {"temperature": 0.7, "top_p": 0.95},
        {"temperature": 1.0, "top_p": 0.95},
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 1.0, "top_p": 0.9},
    ]

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)
        
        # Encode prompt
        input_ids = tok.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        for config in configs:
            temp = config["temperature"]
            top_p = config["top_p"]
            
            print(f"\n--- temperature={temp}, top_p={top_p} ---")
            
            # Generate
            output_ids = model.generate(
                input_ids=input_tensor,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temp
            )
            
            # Decode
            output_text = tok.decode(output_ids[0].tolist())
            print(output_text)
        
        print("\n")


if __name__ == "__main__":
    main()

