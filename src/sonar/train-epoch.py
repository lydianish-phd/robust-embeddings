import os, argparse

from sonar.models.sonar_text.loader import (
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
from sonar_distillation import (
    DataCollatorForSonarDistillation,
    SonarDistillationTrainer,
    compute_metrics,
    tokenize_inputs,
    load_student_encoder_from_checkpoint
)
from datasets import (
    load_dataset,
    interleave_datasets
)
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback
)
from accelerate import Accelerator

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", help="path to output directory", type=str)
    parser.add_argument("--last-checkpoint", help="path to last saved checkpoint", type=str)
    parser.add_argument("--seed", type=int, default=42)
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

    data_en_files = {
        "train": f"{monolingual_data_dir}/eng/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
        "valid": f"{monolingual_data_dir}/eng/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
    }
    data_en = load_dataset("json", data_files=data_en_files, streaming=True)
    data_en = data_en.shuffle(seed=args.seed, buffer_size=10_000)

    # all_train_data = interleave_datasets([data_fr["train"], data_en["train"]], probabilities=[0.5, 0.5], seed=args.seed)
    # all_valid_data = interleave_datasets([data_fr["valid"], data_en["valid"]], probabilities=[0.5, 0.5], seed=args.seed)

    all_train_data = interleave_datasets([data_en_fr["train"], data_fr["train"], data_en["train"]], probabilities=[0.625, 0.25, 0.125], seed=args.seed)
    all_valid_data = interleave_datasets([data_en_fr["valid"], data_fr["valid"], data_en["valid"]], seed=args.seed)

    print("Loading tokenizers...")

    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
    tokenizers = {
        "eng_Latn": tokenizer.create_encoder(lang="eng_Latn"),
        "fra_Latn": tokenizer.create_encoder(lang="fra_Latn")
    }

    print("Tokenizing dataset...")

    max_seq_len = 512

    tokenized_train_data = all_train_data.map(tokenize_inputs, batched=True, drop_last_batch=True, fn_kwargs={"tokenizers": tokenizers, "max_seq_len": max_seq_len, "pad_idx": tokenizer.vocab_info.pad_idx})
    tokenized_train_data = tokenized_train_data.remove_columns(["source_lang", "source_sentence", "target_lang", "target_sentence"])

    tokenized_valid_data = all_valid_data.map(tokenize_inputs, batched=True, drop_last_batch=True, fn_kwargs={"tokenizers": tokenizers, "max_seq_len": max_seq_len, "pad_idx": tokenizer.vocab_info.pad_idx})
    tokenized_valid_data = tokenized_valid_data.remove_columns(["source_lang", "source_sentence", "target_lang", "target_sentence"])

    print("Instantiating data collator...")

    data_collator = DataCollatorForSonarDistillation(tokenizer=tokenizer)
    data_collator = accelerator.prepare(data_collator)

    print("Loading teacher model...")

    teacher_model = accelerator.prepare(load_sonar_text_encoder_model("text_sonar_basic_encoder"))

    print("Initializing student model...")

    nllb_checkpoint_path = os.path.join(os.environ["MODELS"], "nllb600m/nllb200densedst600mcheckpoint")
    student_model = load_student_encoder_from_checkpoint(nllb_checkpoint_path)

    print("Training student model...")

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/models",
        log_level="info",
        fp16=False,
        logging_dir=f"{args.output_dir}/tensorboard",
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="tensorboard",
        push_to_hub=False,
        auto_find_batch_size=True, # per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        num_train_epochs=1,
        warmup_steps=8000,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        save_steps=20000,
        logging_steps=10000,
        eval_steps=20000,
        label_names=['tgt_sentence_ids', 'tgt_seq_lens', 'tgt_batch_seq_len'],
        #prediction_loss_only=True, 
        seed=args.seed,
        resume_from_checkpoint=args.last_checkpoint
    )

    trainer = SonarDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_valid_data,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(resume_from_checkpoint=True)

