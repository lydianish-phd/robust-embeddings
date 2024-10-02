import os, argparse

from sonar.models.sonar_text.loader import (
    load_sonar_tokenizer,
)
from rosonar_distillation import (
    tokenize_inputs
)
from datasets import (
    load_dataset
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", help="number of processes", type=int, default=8)
    args = parser.parse_args()

    print("Loading datasets...")

    bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/concatenated")
    tokenized_bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized")
    monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/concatenated")
    tokenized_monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized")

    data_en_fr_files = {
        "train": f"{bilingual_data_dir}/eng-fra/train.eng_Latn-fra_Latn_chunks/train.eng_Latn-fra_Latn-*.jsonl",
        "valid": f"{bilingual_data_dir}/eng-fra/valid.eng_Latn-fra_Latn_chunks/valid.eng_Latn-fra_Latn-*.jsonl"
    }
    data_en_fr = load_dataset("json", data_files=data_en_fr_files)

    # data_fr_files = {
    #     "train": f"{monolingual_data_dir}/fra/train.fra_Latn-fra_Latn_chunks/train.fra_Latn-fra_Latn-*.jsonl",
    #     "valid": f"{monolingual_data_dir}/fra/valid.fra_Latn-fra_Latn_chunks/valid.fra_Latn-fra_Latn-*.jsonl"
    # }
    # data_fr = load_dataset("json", data_files=data_fr_files)

    # data_en_1_files = {
    #     "train": f"{monolingual_data_dir}/eng/part1/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
    #     "valid": f"{monolingual_data_dir}/eng/part1/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
    # }
    # data_en_1 = load_dataset("json", data_files=data_en_1_files)

    # data_en_2_ugc_files = {
    #     "train": f"{monolingual_data_dir}/eng/part2_ugc/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
    #     "valid": f"{monolingual_data_dir}/eng/part2_ugc/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
    # }        
    # data_en_2_ugc = load_dataset("json", data_files=data_en_2_ugc_files)

    # data_en_2_files = {
    #     "train": f"{monolingual_data_dir}/eng/part2/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
    #     "valid": f"{monolingual_data_dir}/eng/part2/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
    # }
    # data_en_2 = load_dataset("json", data_files=data_en_2_files)

    print("Loading tokenizers...")

    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
    tokenizers = {
        "eng_Latn": tokenizer.create_encoder(lang="eng_Latn"),
        "fra_Latn": tokenizer.create_encoder(lang="fra_Latn")
    }

    print("Tokenizing dataset...")

    max_seq_len = 512

    tokenized_data_en_fr = data_en_fr.map(
        tokenize_inputs,
        batched=True,
        batch_size=10_000,
        fn_kwargs={"tokenizers": tokenizers, "max_seq_len": max_seq_len, "pad_idx": tokenizer.vocab_info.pad_idx},
        remove_columns=["source_lang", "source_sentence", "target_lang", "target_sentence"],
        num_procs=args.num_processes
    )
    
    n_shards = {
        "train": 1000,
        "valid": 32
    }

    for split, split_dataset in tokenized_data_en_fr.items():
        for i in range(n_shards[split]):
            split_dataset.shard(
                num_shards=n_shards[split],
                index=i, contiguous=True
            ).to_json(
                f"{tokenized_bilingual_data_dir}/eng-fra/{split}.eng_Latn-fra_Latn_chunks/{split}.eng_Latn-fra_Latn-{i}.jsonl"
            )

