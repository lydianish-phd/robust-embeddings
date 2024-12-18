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
        teacher_tokenizer: LaserTokenizer, 
        student_tokenizer: XLMRobertaTokenizer, 
        max_length: int = 512,
    ):
    teacher_target_input = teacher_tokenizer.tokenize_batch(examples["target_sentence"])
    teacher_tgt_embeds = torch.tensor(teacher_model.encode_sentences(teacher_target_input))

    student_src_ids_and_masks = student_tokenizer(
        examples["source_sentence"], 
        padding="max_length", 
        max_length=max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    student_tgt_ids_and_masks = student_tokenizer(
        examples["target_sentence"], 
        padding="max_length", 
        max_length=max_length, 
        truncation=True, 
        return_tensors="pt"
    )   
    
    model_inputs = {
        "teacher_tgt_embeds": teacher_tgt_embeds,
        "student_src_ids": student_src_ids_and_masks["input_ids"],
        "student_src_masks": student_src_ids_and_masks["attention_mask"],
        "student_tgt_ids": student_tgt_ids_and_masks["input_ids"],
        "student_tgt_masks": student_tgt_ids_and_masks["attention_mask"]
    }
    return model_inputs

def preprocess_data(
        data, 
        teacher_model,
        teacher_tokenizer, 
        student_tokenizer, 
        max_length, 
        num_proc
    ):
    return data.map(
            tokenize_and_embed_inputs,
            batched=True,
            batch_size=8192,
            fn_kwargs={
                "teacher_model": teacher_model, 
                "teacher_tokenizer": teacher_tokenizer, 
                "student_tokenizer": student_tokenizer, 
                "max_length": max_length
            },
            remove_columns=["source_lang", "source_sentence", "target_lang", "target_sentence"],
            num_proc=num_proc
        )

def write_to_jsonl(data, output_dir_prefix, lang_pair, shard):
    for split, split_dataset in data.items():
        output_dir = f"{output_dir_prefix}/{split}.{lang_pair}_chunks"
        os.makedirs(output_dir, exist_ok=True)
        split_dataset.to_json(
            f"{output_dir}/{split}.{lang_pair}-{shard}.jsonl"
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="dataset split", type=str, choices=["train", "valid"], default="train")
    parser.add_argument("--shard", help="shard number", type=int, default=0)
    parser.add_argument("--num_proc", help="number of processes", type=int, default=1)
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
    max_length = 512

    for lang_pair, metadata in all_metadata.items():
        output_file = f"{metadata['output_dir_prefix']}/{args.split}.{metadata['lang_pair']}_chunks/{args.split}.{metadata['lang_pair']}-{args.shard}.jsonl"
        if os.path.exists(output_file):
            print(f"Skipping {lang_pair} dataset...")
            continue

        print(f"Loading {lang_pair} dataset...")
        data_files = { args.split: f"{metadata['input_dir_prefix']}/{args.split}.{metadata['lang_pair']}_chunks/{args.split}.{metadata['lang_pair']}-{args.shard}.jsonl" }
        data = load_dataset("json", data_files=data_files)
        print(f"Tokenizing {lang_pair} dataset...")
        tokenized_data = preprocess_data(data, teacher_model, teacher_tokenizer, student_tokenizer, max_length, args.num_proc)
        print(f"Writing {lang_pair} dataset to disk...")
        write_to_jsonl(tokenized_data, output_dir_prefix=metadata["output_dir_prefix"], lang_pair=metadata["lang_pair"], shard=args.shard)

