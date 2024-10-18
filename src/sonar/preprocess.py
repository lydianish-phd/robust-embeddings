import os, argparse
from sonar.models.sonar_text.loader import load_sonar_tokenizer
from datasets import load_dataset
import torch

def _tokenize_and_pad(tokenizer, sentence, max_length, pad_idx):
    tensor = tokenizer(sentence)
    padding_length = max_length - tensor.size(0)
    if padding_length <= 0:
        return tensor[:max_length]
    padding = torch.full(torch.Size([padding_length]), fill_value=pad_idx)
    padded_tensor = torch.cat((tensor, padding), dim=0)
    return padded_tensor[:max_length]

def tokenize_inputs(examples, tokenizer, max_seq_len=512):
    pad_idx = tokenizer.vocab_info.pad_idx
    src_sentence_ids = []
    for source_lang, sentence in zip(examples["source_lang"], examples["source_sentence"]):
        src_tokenizer = tokenizer.create_encoder(lang=source_lang)
        src_sentence_ids.append(_tokenize_and_pad(src_tokenizer,sentence,max_seq_len,pad_idx))
    
    tgt_sentence_ids = []
    for target_lang, sentence in zip(examples["target_lang"], examples["target_sentence"]):
        tgt_tokenizer = tokenizer.create_encoder(lang=target_lang)
        tgt_sentence_ids.append(_tokenize_and_pad(tgt_tokenizer,sentence,max_seq_len,pad_idx))
    
    model_inputs = {
        "src_sentence_ids": src_sentence_ids,
        "tgt_sentence_ids": tgt_sentence_ids
    }
    return model_inputs

def tokenize_data(data, tokenizer, max_seq_len, num_proc):
    return data.map(
            tokenize_inputs,
            batched=True,
            batch_size=20_000,
            fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
            remove_columns=["source_lang", "source_sentence", "target_lang", "target_sentence"],
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
    tokenized_bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized")
    tokenized_monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized")

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
            "input_dir_p refix": f"{monolingual_data_dir}/eng/part2_ugc/",
            "output_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part2_ugc/",
            "lang_pair": "eng_Latn-eng_Latn"
        }
    }

    print("Loading tokenizer...")

    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
    max_seq_len = 512

    for lang_pair, metadata in all_metadata.items():
        print(f"Loading {lang_pair} dataset...")
        data_files = { split: f"{metadata['input_dir_prefix']}/{split}.{metadata['lang_pair']}_chunks/{split}.{metadata['lang_pair']}-*.jsonl" for split in ["train", "valid"] }
        data = load_dataset("json", data_files=data_files)
        num_shards = {
            "train": data["train"].n_shards,
            "valid": data["valid"].n_shards
        }
        print(f"Tokenizing {lang_pair} dataset...")
        tokenized_data = tokenize_data(data, tokenizer, max_seq_len, num_proc=args.num_processes)
        print(f"Writing {lang_pair} dataset to disk...")
        write_to_jsonl(tokenized_data, output_dir_prefix=metadata["output_dir_prefix"], lang_pair=metadata["lang_pair"], num_shards=num_shards)
    
