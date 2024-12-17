import os, argparse
from torch.utils.data import DataLoader
from datasets import (
    load_dataset,
    interleave_datasets
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", help="batch size", type=int, default=8192)
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

    print(f"Using batch size of {args.batch_size}")
    
    data_loader = DataLoader(tokenized_train_data, batch_size=args.batch_size, num_workers=args.dataloader_workers, pin_memory=True)

    # Initialize counters
    n_steps = 0
    n_samples = 0
    incomplete_batch_sizes = []

    # Loop through the DataLoader
    for batch in data_loader:
        current_batch_size = len(batch)
        n_samples += current_batch_size
        if current_batch_size < args.batch_size:
            incomplete_batch_sizes.append(current_batch_size)
        n_steps += 1
        if n_steps % 100 == 0:
            print(f"Processed {n_steps} steps...")

    # Print total counts
    print(f"Total steps for a batch size of {args.batch_size}: {n_steps}")
    print(f"Total samples: {n_samples}")
    print(f"Incomplete batches: num = {len(incomplete_batch_sizes)}, sum of samples = {sum(incomplete_batch_sizes)}")  
