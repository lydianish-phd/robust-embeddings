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
            batch_size=20_000,
            fn_kwargs={
                "teacher_model": teacher_model, 
                "teacher_tokenizer": teacher_tokenizer, 
                "student_tokenizer": student_tokenizer, 
                "max_length": max_length
            },
            remove_columns=["source_sentence", "target_sentence"],
            num_proc=num_proc
        )

def write_to_jsonl(data, output_dir_prefix, lang_pair, num_shards):
    for split, split_dataset in data.items():
        output_dir = f"{output_dir_prefix}/{split}.{lang_pair}_chunks"
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_shards[split]):
            split_dataset.shard(
                num_shards=num_shards[split],
                index=i, contiguous=True
            ).to_json(
                f"{output_dir}/{split}.{lang_pair}-{i}.jsonl"
            )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", help="number of processes", type=int, default=8)
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

    teacher_model = initialize_encoder(lang="english")
    
    print("Loading tokenizers...")

    teacher_tokenizer = initialize_tokenizer(lang="english")
    student_tokenizer = XLMRobertaTokenizerFast.from_pretrained(xlm_checkpoint_path)
    max_length = 512
    num_shards = {
        "train": 1000,
        "valid": 32
    }

    for lang_pair, metadata in all_metadata.items():
        print(f"Loading {lang_pair} dataset...")
        data_files = { split: f"{metadata['input_dir_prefix']}/{split}.{metadata['lang_pair']}_chunks/{split}.{metadata['lang_pair']}-*.jsonl" for split in ["train", "valid"] }
        data = load_dataset("json", data_files=data_files)
        print(f"Tokenizing {lang_pair} dataset...")
        tokenized_data = preprocess_data(data, teacher_model, teacher_tokenizer, student_tokenizer, max_length, num_proc=args.num_processes)
        print(f"Writing {lang_pair} dataset to disk...")
        write_to_jsonl(tokenized_data, output_dir_prefix=metadata["output_dir_prefix"], lang_pair=metadata["lang_pair"], num_shards=num_shards)
    
