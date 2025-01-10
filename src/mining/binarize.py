import os, argparse
from datasets import load_dataset
from .preprocess import write_to_file

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="dataset split", type=str, choices=["train", "valid"], default="train")
    parser.add_argument("--shard", help="shard number", type=int, default=0)
    args = parser.parse_args()

    bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/concatenated")
    monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/concatenated")
    tokenized_bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized/rolaser")
    tokenized_monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized/rolaser")

    all_metadata = {
        "en-fr": {
            "input_dir_prefix": f"{bilingual_data_dir}/eng-fra/",
            "output_dir_prefix": f"{tokenized_bilingual_data_dir}/eng-fra/",
            "lang_pair": "eng_Latn-fra_Latn"
        },
        "fr": {
            "input_dir_prefix": f"{monolingual_data_dir}/fra/",
            "output_dir_prefix": f"{tokenized_monolingual_data_dir}/fra/",
            "lang_pair": "fra_Latn-fra_Latn"
        },
        "en_1": {
            "input_dir_prefix": f"{monolingual_data_dir}/eng/part1/",
            "output_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part1/",
            "lang_pair": "eng_Latn-eng_Latn"
        },
        "en_2": {
            "input_dir_prefix": f"{monolingual_data_dir}/eng/part2/",
            "output_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part2/",
            "lang_pair": "eng_Latn-eng_Latn"
        },
        "en_2_ugc": {
            "input_dir_prefix": f"{monolingual_data_dir}/eng/part2_ugc/",
            "output_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part2_ugc/",
            "lang_pair": "eng_Latn-eng_Latn"
        }
    }

    for lang_pair, metadata in all_metadata.items():
        input_file = f"{metadata['output_dir_prefix']}/{args.split}.{metadata['lang_pair']}_chunks/{args.split}.{metadata['lang_pair']}-{args.shard}.jsonl"
        if os.path.exists(input_file):
            print(f"Binarizing {lang_pair} dataset...")
            data = load_dataset("json", data_files={args.split: input_file})
            write_to_file(data, output_dir_prefix=metadata["output_dir_prefix"], lang_pair=metadata["lang_pair"], shard=args.shard, filetype="parquet")

