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
    parser.add_argument("--model-name", help="name of the model to train", type=str, default="rosonar")
    parser.add_argument("--resume-last", help="whether to resume training from last checkpoint", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="inverse_sqrt")
    parser.add_argument("--ugc-en", help="use artificial UGC English in training data", type=bool, default=True)
    args = parser.parse_args()

    accelerator = Accelerator()

    print("Loading datasets...")

    bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/concatenated")
    monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/concatenated")

    data_en_fr_files = {
        "train": f"{bilingual_data_dir}/eng-fra/train.eng_Latn-fra_Latn_chunks/train.eng_Latn-fra_Latn-*.jsonl",
        "valid": f"{bilingual_data_dir}/eng-fra/valid.eng_Latn-fra_Latn_chunks/valid.eng_Latn-fra_Latn-*.jsonl"
    }
    data_en_fr = load_dataset("json", data_files=data_en_fr_files, streaming=True)
    data_en_fr = data_en_fr.shuffle(seed=args.seed, buffer_size=10_000)

    data_fr_files = {
        "train": f"{monolingual_data_dir}/fra/train.fra_Latn-fra_Latn_chunks/train.fra_Latn-fra_Latn-*.jsonl",
        "valid": f"{monolingual_data_dir}/fra/valid.fra_Latn-fra_Latn_chunks/valid.fra_Latn-fra_Latn-*.jsonl"
    }
    data_fr = load_dataset("json", data_files=data_fr_files, streaming=True)
    data_fr = data_fr.shuffle(seed=args.seed, buffer_size=10_000)

    data_en_1_files = {
        "train": f"{monolingual_data_dir}/eng/part1/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
        "valid": f"{monolingual_data_dir}/eng/part1/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
    }
    data_en_1 = load_dataset("json", data_files=data_en_1_files, streaming=True)
    data_en_1 = data_en_1.shuffle(seed=args.seed, buffer_size=10_000)

    if args.ugc_en:
        data_en_2_files = {
            "train": f"{monolingual_data_dir}/eng/part2_ugc/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
            "valid": f"{monolingual_data_dir}/eng/part2_ugc/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
        }        
    else:
        data_en_2_files = {
            "train": f"{monolingual_data_dir}/eng/part2/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
            "valid": f"{monolingual_data_dir}/eng/part2/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
        }
    data_en_2 = load_dataset("json", data_files=data_en_2_files, streaming=True)
    data_en_2 = data_en_2.shuffle(seed=args.seed, buffer_size=10_000)


    all_train_data = interleave_datasets([data_en_fr["train"], data_fr["train"], data_en_1["train"], data_en_2["train"]], probabilities=[4/8, 2/8, 1/8, 1/8], seed=args.seed)
    all_valid_data = interleave_datasets([data_en_fr["valid"], data_fr["valid"], data_en_1["valid"], data_en_2["valid"]], seed=args.seed)

    print("Loading tokenizers...")

    teacher_tokenizer = initialize_tokenizer(lang="english")
    student_tokenizer = XLMRobertaTokenizerFast.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")

    print("Loading teacher model...")

    teacher_model = accelerator.prepare(initialize_encoder(lang="english"))

    print("Initializing student model...")

    student_model_config = RoLaserConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", output_size=1024, pooling="max")
    student_model = RoLaserModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=student_model_config)

    print("Instantiating data collator...")

    max_seq_len = 512

    data_collator = DataCollatorForRoLaserDistillation(
        teacher_tokenizer=teacher_tokenizer, 
        student_tokenizer=student_tokenizer, 
        teacher_padding_value=teacher_model.pad_index, 
        max_length=max_seq_len,
        return_tensors="pt"
    )
    data_collator = accelerator.prepare(data_collator)

    print("Training student model...")

    checkpoint_dir = f"{args.output_dir}/models/{args.model_name}"
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        log_level="info",
        bf16=True,
        logging_dir=f"{args.output_dir}/tensorboard",
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="tensorboard",
        push_to_hub=False,
        dataloader_num_workers=64,
        auto_find_batch_size=True, # per_device_train_batch_size=8,
        gradient_accumulation_steps=args.accumulation_steps,
        eval_accumulation_steps=args.accumulation_steps,
        remove_unused_columns=False,
        max_steps=130_000,
        warmup_steps=8_000,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        save_steps=10_000,
        logging_steps=100,
        eval_steps=10_000,
        label_names=['teacher_tgt_pieces'],
        seed=args.seed,
        resume_from_checkpoint=checkpoint_dir
    )

    trainer = RoLaserDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=all_train_data,
        eval_dataset=all_valid_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    resume_from_checkpoint = args.resume_last and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

