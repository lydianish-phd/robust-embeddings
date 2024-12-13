import os, argparse
from torch.utils.data import DataLoader
from datasets import (
    load_dataset,
    interleave_datasets
)
import torch

# Function to determine batch size based on available GPU memory
def get_dynamic_batch_size(fallback_batch_size=16, factor=256):
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
        # Estimate a batch size based on GPU memory (e.g., 256 samples per GB of memory)
        batch_size = int(total_memory / (1024**3) * factor)
        return max(batch_size, fallback_batch_size)  # Ensure batch size isn't too small
    else:
        # Default batch size for CPU
        return fallback_batch_size


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-workers", help="number of workers for data loading", type=int, default=8)
    args = parser.parse_args()

    print("Loading datasets...")

    tokenized_bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized/rosonar")
    tokenized_monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized/rosonar")

    all_metadata = {
        "en-fr": {
            "input_dir_prefix": f"{tokenized_bilingual_data_dir}/eng-fra/",
            "lang_pair": "eng_Latn-fra_Latn"
        },
        "fr": {
            "input_dir_prefix": f"{tokenized_monolingual_data_dir}/fra/",
            "lang_pair": "fra_Latn-fra_Latn"
        },
        "en_1": {
            "input_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part1/",
            "lang_pair": "eng_Latn-eng_Latn"
        },
        "en_2": {
            "input_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part2/",
            "lang_pair": "eng_Latn-eng_Latn"
        }
    }

    tokenized_data = {}
    for lang_pair, metadata in all_metadata.items():
        data_files = { "train": f"{metadata['input_dir_prefix']}/train.{metadata['lang_pair']}_chunks/train.{metadata['lang_pair']}-*.jsonl" }
        tokenized_data[lang_pair] = load_dataset("json", data_files=data_files, streaming=True)
        tokenized_data[lang_pair] = tokenized_data[lang_pair].shuffle(seed=args.seed, buffer_size=10_000)

    tokenized_train_data = interleave_datasets([data["train"] for data in tokenized_data.values()], probabilities=[4/8, 2/8, 1/8, 1/8], seed=args.seed, stopping_strategy="all_exhausted")

    batch_size = get_dynamic_batch_size()
    print(f"Using batch size of {batch_size}")
    
    data_loader = DataLoader(tokenized_train_data, batch_size=batch_size, num_workers=args.dataloader_workers, pin_memory=True)

    # Initialize counters
    n_steps = 0

    # Loop through the DataLoader
    for batch in data_loader:
        n_steps += 1
        if n_steps % 1000 == 0:
            print(f"Processed {n_steps} steps...")

    # Print total counts
    print(f"Total steps for a batch size of {batch_size}: {n_steps}")
