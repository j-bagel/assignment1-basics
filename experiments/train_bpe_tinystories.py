import time
import os
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
    vocab, merges = train_bpe(txt_path, 10000, ["<|endoftext|>"], num_processes=8)

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
