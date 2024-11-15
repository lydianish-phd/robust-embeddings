import os, argparse

from laser_encoders import (
    initialize_encoder, 
    initialize_tokenizer
)
from rolaser_model import (
    RoLaserConfig,
    RoLaserModel
)
from rolaser_distillation import (
    DataCollatorForRoLaserDistillation,
    RoLaserDistillationTrainer,
    compute_metrics,
)
from datasets import (
    load_dataset,
    interleave_datasets
)
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    XLMRobertaTokenizerFast
)
from accelerate import Accelerator

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", help="path to output directory", type=str)
    parser.add_argument("--resume-last", help="whether to resume training from last checkpoint", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="inverse_sqrt")
    parser.add_argument("--ugc-en", help="use artificial UGC English in training data", type=bool, default=True)
    parser.add_argument("--dataloader-workers", help="number of workers for data loading", type=int, default=8)
    args = parser.parse_args()

    accelerator = Accelerator()

    print("Loading datasets...")

    bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/concatenated")
    monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/concatenated")

    all_metadata = {
        "en-fr": {
            "input_dir_prefix": f"{bilingual_data_dir}/eng-fra/",
            "lang_pair": "eng_Latn-fra_Latn"
        },
        "fr": {
            "input_dir_prefix": f"{monolingual_data_dir}/fra/",
            "lang_pair": "fra_Latn-fra_Latn"
        },
        "en_1": {
            "input_dir_prefix": f"{monolingual_data_dir}/eng/part1/",
            "lang_pair": "eng_Latn-eng_Latn"
        },
        "en_2": {
            "input_dir_prefix": f"{monolingual_data_dir}/eng/part2/",
            "lang_pair": "eng_Latn-eng_Latn"
        },
        "en_2_ugc": {
            "input_dir_prefix": f"{monolingual_data_dir}/eng/part2_ugc/",
            "lang_pair": "eng_Latn-eng_Latn"
        }
    }

    data = {}
    for lang_pair, metadata in all_metadata.items():
        if lang_pair == "en_2" and args.ugc_en:
            continue
        elif lang_pair == "en_2_ugc" and not args.ugc_en:
            continue
        data_files = { split: f"{metadata['input_dir_prefix']}/{split}.{metadata['lang_pair']}_chunks/{split}.{metadata['lang_pair']}-*.jsonl" for split in ["train", "valid"] }
        data[lang_pair] = load_dataset("json", data_files=data_files, streaming=True)
        data[lang_pair] = data[lang_pair].shuffle(seed=args.seed, buffer_size=10_000)
    
    all_train_data = interleave_datasets([data["train"] for data in data.values()], probabilities=[4/8, 2/8, 1/8, 1/8], seed=args.seed, stopping_strategy="all_exhausted")
    all_valid_data = interleave_datasets([data["valid"] for data in data.values()], seed=args.seed, stopping_strategy="all_exhausted")

    print("Defining initialisation checkpoint...")

    xlm_checkpoint = "cardiffnlp/twitter-xlm-roberta-base"
    xlm_checkpoint_path = os.path.join(os.environ["MODELS"], xlm_checkpoint)

    print("Loading tokenizers...")

    # teacher_tokenizer = initialize_tokenizer(lang="english")
    student_tokenizer = XLMRobertaTokenizerFast.from_pretrained(xlm_checkpoint_path)

    # print("Loading teacher model...")

    # teacher_model = accelerator.prepare(initialize_encoder(lang="english"))

    print("Initializing student model...")

    student_model_config = RoLaserConfig.from_pretrained(xlm_checkpoint_path, output_size=1024, pooling="max")
    student_model = RoLaserModel.from_pretrained(xlm_checkpoint_path, config=student_model_config, ignore_mismatched_sizes=True) #ignore the pooling layer from the checkpoint

    # print("Instantiating data collator...")

    # max_seq_len = 512

    # data_collator = DataCollatorForRoLaserDistillation(
    #     teacher_tokenizer=teacher_tokenizer, 
    #     student_tokenizer=student_tokenizer, 
    #     max_length=max_seq_len,
    #     teacher_padding_value=teacher_model.pad_index,
    #     return_tensors="pt"
    # )
    # data_collator = accelerator.prepare(data_collator)

    print("Training student model...")

    checkpoint_dir = f"{args.output_dir}/models"
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        log_level="info",
        logging_dir=f"{args.output_dir}/tensorboard",
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="tensorboard",
        push_to_hub=False,
        dataloader_num_workers=args.dataloader_workers,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=args.accumulation_steps,
        eval_accumulation_steps=args.accumulation_steps,
        remove_unused_columns=False,
        max_steps=10_267_792, # steps needed to exhaust all en-fr data
        warmup_steps=10_000,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        save_steps=20_000,
        logging_steps=100,
        eval_steps=20_000,
        label_names=['teacher_tgt_ids'],
        seed=args.seed,
        resume_from_checkpoint=checkpoint_dir
    )

    trainer = RoLaserDistillationTrainer(
        student_model=student_model,
        # teacher_model=teacher_model,
        # teacher_tokenizer=teacher_tokenizer,
        args=training_args,
        train_dataset=all_train_data,
        eval_dataset=all_valid_data,
        # data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    resume_from_checkpoint = args.resume_last and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

