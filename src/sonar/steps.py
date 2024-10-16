import os
from torch.utils.data import DataLoader

from datasets import (
    load_dataset,
    interleave_datasets
)

seed = 42

print("Loading datasets...")

bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized")
monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized")

data_en_fr_files = {
    "train": f"{bilingual_data_dir}/eng-fra/train.eng_Latn-fra_Latn_chunks/train.eng_Latn-fra_Latn-*.jsonl"
}
data_en_fr = load_dataset("json", data_files=data_en_fr_files, streaming=True)

data_fr_files = {
    "train": f"{monolingual_data_dir}/fra/train.fra_Latn-fra_Latn_chunks/train.fra_Latn-fra_Latn-*.jsonl"
}
data_fr = load_dataset("json", data_files=data_fr_files, streaming=True)

data_en_files = {
    "train": f"{monolingual_data_dir}/eng/part1/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl"
}
data_en = load_dataset("json", data_files=data_en_files, streaming=True)

data_en_2_files = {
    "train": f"{monolingual_data_dir}/eng/part2/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl"
}
data_en_2 = load_dataset("json", data_files=data_en_2_files, streaming=True)

strategy = "first_exhausted"
all_train_data = interleave_datasets([data_en_fr["train"], data_fr["train"], data_en["train"], data_en_2["train"]], probabilities=[4/8, 2/8, 1/8, 1/8], seed=seed, stopping_strategy=strategy)

print("Interleaving strategy", strategy)

data_loader = DataLoader(all_train_data, batch_size=2048, num_workers=40)

# Initialize counters
total_elements = 0

# Loop through the DataLoader
for batch in data_loader:
    # Count the elements in the current batch
    total_elements += len(batch['source_sentence'])
    if total_elements % 1_000_000 == 0:
        print(f"Processed {total_elements} elements")

# Print total counts
print("Interleaving strategy", strategy)    
print(f"Total elements: {total_elements}")
