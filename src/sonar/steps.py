import os
from torch.utils.data import DataLoader

from datasets import (
    load_dataset,
    interleave_datasets
)

seed = 42
fraction_en_fr = 5
fraction_fr = 2
fraction_en = 1

print("Loading datasets...")

bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/concatenated")
monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/concatenated")

data_en_fr_files = {
    "train": f"{bilingual_data_dir}/eng-fra/train.eng_Latn-fra_Latn_chunks/train.eng_Latn-fra_Latn-*.jsonl"
}
data_en_fr = load_dataset("json", data_files=data_en_fr_files, streaming=True)
data_en_fr = data_en_fr.shuffle(seed=seed, buffer_size=10_000)

data_fr_files = {
    "train": f"{monolingual_data_dir}/fra/train.fra_Latn-fra_Latn_chunks/train.fra_Latn-fra_Latn-*.jsonl"
}
data_fr = load_dataset("json", data_files=data_fr_files, streaming=True)
data_fr = data_fr.shuffle(seed=seed, buffer_size=10_000)

data_en_files = {
    "train": f"{monolingual_data_dir}/eng/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl"
}
data_en = load_dataset("json", data_files=data_en_files, streaming=True)
data_en = data_en.shuffle(seed=seed, buffer_size=10_000)

all_train_data = interleave_datasets([data_en_fr["train"], data_fr["train"], data_en["train"]], probabilities=[fraction_en_fr/8, fraction_fr/8, fraction_en/8], seed=seed)

data_loader = DataLoader(all_train_data, batch_size=32, num_workers=4)

# Initialize counters
total_elements = 0

# Loop through the DataLoader
for batch in data_loader:
    # Count the elements in the current batch
    total_elements += len(batch['source_sentence'])

# Print total counts
print(f"Total elements: {total_elements}")
