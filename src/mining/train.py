import os, argparse

from rolaser_sentence_encoder import RoLaserSentenceEncoder
from rolaser_distillation import (
    DataCollatorForRoLaserDistillation,
    RoLaserDistillationTrainer,
    compute_metrics,
)
from datasets import (
    load_dataset,
    interleave_datasets,
    concatenate_datasets,
)
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)
from accelerate import Accelerator

DATA_SEED_OFFSET = 100

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
    parser.add_argument("--single-mse-loss", help="use single MSE loss for distillation", default=False, action="store_true")
    parser.add_argument("--pooling-mode", help="pooling mode for student model", type=str, default="mean")
    args = parser.parse_args()

    accelerator = Accelerator()

    print("Loading datasets...")

    tokenized_bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized/rolaser")
    tokenized_monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized/rolaser")

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
        },
        "en_2_ugc": {
            "input_dir_prefix": f"{tokenized_monolingual_data_dir}/eng/part2_ugc/",
            "lang_pair": "eng_Latn-eng_Latn"
        }
    }

    tokenized_data = {}
    for lang_pair, metadata in all_metadata.items():
        if lang_pair == "en_2" and args.ugc_en:
            continue
        elif lang_pair == "en_2_ugc" and not args.ugc_en:
            continue
        data_files = { split: f"{metadata['input_dir_prefix']}/{split}.{metadata['lang_pair']}_chunks/{split}.{metadata['lang_pair']}-*.parquet" for split in ["train", "valid"] }
        tokenized_data[lang_pair] = load_dataset("parquet", data_files=data_files, streaming=True)
        tokenized_data[lang_pair]["train"] = tokenized_data[lang_pair]["train"].shuffle(seed=args.seed+DATA_SEED_OFFSET, buffer_size=10_000)
    
    tokenized_train_data = interleave_datasets([data["train"] for data in tokenized_data.values()], probabilities=[4/8, 2/8, 1/8, 1/8], seed=args.seed+DATA_SEED_OFFSET, stopping_strategy="first_exhausted")
    tokenized_valid_data = concatenate_datasets([data["valid"] for data in tokenized_data.values()])

    print("Defining initialisation checkpoint...")

    xlm_checkpoint = "cardiffnlp/twitter-xlm-roberta-base"
    xlm_checkpoint_path = os.path.join(os.environ["MODELS"], xlm_checkpoint)

    print("Initializing student model...")

    student_model = RoLaserSentenceEncoder(xlm_checkpoint_path, pooling_mode=args.pooling_mode)

    print("Instantiating data collator...")

    data_collator = accelerator.prepare(DataCollatorForRoLaserDistillation())

    print("Training student model...")

    checkpoint_dir = f"{args.output_dir}/models"
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        fp16=True,
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
        max_steps=3_000_000, # "first exhausted" is approximately at 2_270_671 steps
        warmup_steps=10_000,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        save_steps=20_000,
        logging_steps=100,
        eval_steps=20_000,
        label_names=['teacher_tgt_embeds'],
        seed=args.seed,
        resume_from_checkpoint=checkpoint_dir
    )

    trainer = RoLaserDistillationTrainer(
        student_model=student_model,
        single_mse_loss=args.single_mse_loss,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_valid_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    resume_from_checkpoint = args.resume_last and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

