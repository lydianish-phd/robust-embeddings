import os, argparse
from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
from laser_encoders.laser_tokenizer import LaserTokenizer
from laser_encoders.models import SentenceEncoder
from laser_encoders import initialize_tokenizer, initialize_encoder
from datasets import load_dataset
import torch

def tokenize_and_embed_inputs(
        examples, 
        teacher_model: SentenceEncoder,
        teacher_tokenizer: LaserTokenizer
    ):
    teacher_target_input = teacher_tokenizer.tokenize_batch(examples["target_sentence"])
    teacher_tgt_embeds = torch.tensor(teacher_model.encode_sentences(teacher_target_input))
    return {
        "teacher_tgt_embeds": teacher_tgt_embeds
    }

def preprocess_data(
        data, 
        teacher_model,
        teacher_tokenizer,
        num_proc
    ):
    return data.map(
            tokenize_and_embed_inputs,
            batched=True,
            batch_size=8192,
            fn_kwargs={
                "teacher_model": teacher_model, 
                "teacher_tokenizer": teacher_tokenizer, 
            },
            remove_columns=["source_lang", "target_lang"],
            num_proc=num_proc
        )

def write_to_file(data, output_dir_prefix, lang_pair, shard, filetype="parquet"):
    for split, split_dataset in data.items():
        output_dir = f"{output_dir_prefix}/{split}.{lang_pair}_chunks"
        os.makedirs(output_dir, exist_ok=True)
        if filetype == "jsonl":
            split_dataset.to_json(
                f"{output_dir}/{split}.{lang_pair}-{shard}.jsonl"
            )
        split_dataset.to_parquet(
            f"{output_dir}/{split}.{lang_pair}-{shard}.parquet"
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="dataset split", type=str, choices=["train", "valid"], default="train")
    parser.add_argument("--shard", help="shard number", type=int, default=0)
    parser.add_argument("--num-processes", help="number of processes", type=int, default=1)
    parser.add_argument("--filetype", help="filetype to save tokenized data", type=str, choices=["jsonl", "parquet"], default="parquet")
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

    print("Defining initialisation checkpoint...")

    xlm_checkpoint = "cardiffnlp/twitter-xlm-roberta-base"
    xlm_checkpoint_path = os.path.join(os.environ["MODELS"], xlm_checkpoint)

    print("Loading teacher model...")

    teacher_model = initialize_encoder(laser="laser2")
    
    print("Loading tokenizers...")

    teacher_tokenizer = initialize_tokenizer(laser="laser2")
    student_tokenizer = XLMRobertaTokenizerFast.from_pretrained(xlm_checkpoint_path)

    for lang_pair, metadata in all_metadata.items():
        output_file = f"{metadata['output_dir_prefix']}/{args.split}.{metadata['lang_pair']}_chunks/{args.split}.{metadata['lang_pair']}-{args.shard}.{args.filetype}"
        if os.path.exists(output_file):
            print(f"Skipping {lang_pair} dataset...")
            continue

        print(f"Loading {lang_pair} dataset...")
        data_files = { args.split: f"{metadata['input_dir_prefix']}/{args.split}.{metadata['lang_pair']}_chunks/{args.split}.{metadata['lang_pair']}-{args.shard}.jsonl" }
        data = load_dataset("json", data_files=data_files)
        print(f"Tokenizing {lang_pair} dataset...")
        tokenized_data = preprocess_data(data, teacher_model, teacher_tokenizer, args.num_processes)
        print(f"Writing {lang_pair} dataset to disk...")
        write_to_file(tokenized_data, output_dir_prefix=metadata["output_dir_prefix"], lang_pair=metadata["lang_pair"], shard=args.shard, filetype=args.filetype)

