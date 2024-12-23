import os, argparse

from transformers.trainer_callback import TrainerControl, TrainerState

from sonar.models.sonar_text.loader import (
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
from rosonar_distillation import (
    DataCollatorForRoSonarDistillation,
    RoSonarDistillationTrainer,
    load_student_encoder_from_checkpoint,
    compute_metrics,
)
from datasets import (
    load_dataset,
    interleave_datasets,
    IterableDataset
)
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    get_inverse_sqrt_schedule,
    AdamW
)
from accelerate import Accelerator

STEPS_PER_EPOCH = 320_000
EFFECTIVE_BATCH_SIZE = 2048
SAMPLES_PER_EPOCH = STEPS_PER_EPOCH * EFFECTIVE_BATCH_SIZE
class CustomIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: IterableDataset,
        samples: int
    ):
        """
        Args:
            dataset: The dataset to wrap.
            samples: The number of samples to yield from the dataset
        """
        super().__init__(
            dataset._ex_iterable, 
            dataset._info, 
            dataset._split, 
            dataset._formatting, 
            dataset._shuffling, 
            dataset._distributed, 
            dataset._token_per_repo_id
        )
        self.samples = samples

    def __len__(self):
        return self.samples

class ResetInverseSrqtSchedulerCallback(TrainerCallback):
    def __init__(self, num_steps_per_epoch, num_warmup_steps, timescale=None):
        """
        Args:
            num_warmup_steps: Number of steps for the warm-up phase.
            timescale: Timescale for the inverse square root decay. Defaults to num_warmup_steps.
        """
        self.optimizer = None
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_warmup_steps = num_warmup_steps
        self.timescale = timescale if timescale else num_warmup_steps
        self.scheduler = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Reset the scheduler at the beginning of each epoch.
        """
        if state.epoch > 0:
            print(f"Resetting learning rate scheduler at the start of epoch {round(state.epoch) + 1}")
            self.optimizer = kwargs["optimizer"]
            new_warmup_steps = round(state.epoch) * self.num_steps_per_epoch + self.num_warmup_steps
            new_timescale = round(state.epoch) * self.num_steps_per_epoch + self.timescale
            self.scheduler = get_inverse_sqrt_schedule(
                optimizer=self.optimizer,
                num_warmup_steps=new_warmup_steps,
                timescale=new_timescale,
                last_epoch=state.epoch
            )

    def on_step_end(self, args, state, control, **kwargs):
        """
        Step the scheduler after each batch.
        """
        if self.scheduler:
            self.scheduler.step()

    def on_log(self, args, state, control, **kwargs):
        print("log", kwargs["optimizer"].state_dict())
        print("last lr", self.scheduler.get_last_lr())


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
    parser.add_argument("--dataloader-workers", help="number of workers for data loading", type=int, default=8)
    args = parser.parse_args()

    accelerator = Accelerator()

    print("Loading datasets...")

    tokenized_bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/tokenized/rosonar")
    tokenized_monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/tokenized/rosonar")

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
        data_files = { split: f"{metadata['input_dir_prefix']}/{split}.{metadata['lang_pair']}_chunks/{split}.{metadata['lang_pair']}-*.jsonl" for split in ["train", "valid"] }
        tokenized_data[lang_pair] = load_dataset("json", data_files=data_files, streaming=True)
        tokenized_data[lang_pair] = tokenized_data[lang_pair].shuffle(seed=args.seed, buffer_size=10_000)
    
    tokenized_train_data = interleave_datasets([data["train"] for data in tokenized_data.values()], probabilities=[4/8, 2/8, 1/8, 1/8], seed=args.seed, stopping_strategy="all_exhausted")
    tokenized_train_data = CustomIterableDataset(tokenized_train_data, samples=SAMPLES_PER_EPOCH) # samples needed to exhaust all data
    tokenized_valid_data = interleave_datasets([data["valid"] for data in tokenized_data.values()], seed=args.seed, stopping_strategy="all_exhausted")

    print("Loading tokenizer...")

    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
    
    print("Instantiating data collator...")

    data_collator = DataCollatorForRoSonarDistillation(tokenizer=tokenizer)
    data_collator = accelerator.prepare(data_collator)

    print("Loading teacher model...")

    teacher_model = accelerator.prepare(load_sonar_text_encoder_model("text_sonar_basic_encoder"))

    print("Initializing student model...")

    nllb_checkpoint_path = os.path.join(os.environ["MODELS"], "nllb600m/nllb200densedst600mcheckpoint")
    student_model = load_student_encoder_from_checkpoint(nllb_checkpoint_path)

    print("Training student model...")

    checkpoint_dir = f"{args.output_dir}/models/{args.model_name}"
    tensorboard_dir = f"{args.output_dir}/tensorboard/{args.model_name}"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        bf16=True,
        log_level="info",
        logging_dir=tensorboard_dir,
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
        num_train_epochs=10,
        warmup_steps=8_000,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        save_steps=10_000,
        logging_steps=100,
        eval_steps=10_000,
        label_names=["tgt_sentence_ids", "tgt_seq_lens", "tgt_batch_seq_len"],
        seed=args.seed,
        resume_from_checkpoint=checkpoint_dir
    )

    trainer = RoSonarDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_valid_data,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=10),
            ResetInverseSrqtSchedulerCallback(num_steps_per_epoch=STEPS_PER_EPOCH, num_warmup_steps=8_000)
        ]
    )

    resume_from_checkpoint = args.resume_last and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

