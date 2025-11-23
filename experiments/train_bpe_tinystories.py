import time
import os
import pickle
from cs336_basics.train_bpe import train_bpe
import cProfile
import pstats
import tracemalloc


TEST_RUN = False

if TEST_RUN:
    txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                            'data/TinyStoriesV2-GPT4-train-small.txt')
else:
    txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                            'data/TinyStoriesV2-GPT4-train.txt')


def main():
    start_time = time.time()
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(txt_path, 10000, special_tokens, num_processes=8)

    # Save vocab, merges, and special_tokens
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_tinystories')
    os.makedirs(output_dir, exist_ok=True)

    if not TEST_RUN:
        with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
        with open(os.path.join(output_dir, 'merges.pkl'), 'wb') as f:
            pickle.dump(merges, f)
        with open(os.path.join(output_dir, 'special_tokens.pkl'), 'wb') as f:
            pickle.dump(special_tokens, f)

        print(f"Saved vocab, merges, and special_tokens to {output_dir}")
        print(f"Run time {time.time() - start_time: .2f} seconds.")


if __name__ == "__main__":
    if TEST_RUN:
        # --- Memory profiling start ---
        tracemalloc.start()

        # --- CPU profiling start ---
        profiler = cProfile.Profile()
        profiler.enable()

        # --- Run your main code ---
        main()

        # --- CPU profiling end ---
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats(30)  # print top 30 slowest functions

        # --- Memory profiling end ---
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nCurrent memory usage: {current / (1024**2): .2f} MB")
        print(f"Peak memory usage:    {peak / (1024**2): .2f} MB")
    else:
        # --- Run your main code ---
        main()
