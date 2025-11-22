import os

train_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'data/TinyStoriesV2-GPT4-train.txt'
)

smaller_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'data/TinyStoriesV2-GPT4-train-smaller.txt'
)

MAX_LINES = 3_000_000

with open(train_path, 'r', encoding='utf-8') as fin, \
     open(smaller_path, 'w', encoding='utf-8') as fout:

    for i, line in enumerate(fin):
        if i >= MAX_LINES:
            break
        fout.write(line)

print("Done. Wrote", min(MAX_LINES, i+1), "lines.")
