import os

from transformers import TrainingArguments, Trainer, DefaultDataCollator
from datasets import load_dataset

from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.sequence import SequenceBatch, PaddingMask
from fairseq2.data import Collater
from sonar.inference_pipelines.utils import extract_sequence_batch
from sonar.models.sonar_text.loader import (
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
    convert_sonar_text_encoder_checkpoint
)
from sonar.models.sonar_text.builder import (
    create_sonar_text_encoder_model, 
    sonar_text_encoder_archs
)

from typing import Any, Dict, List

import torch
from torch.nn import MSELoss, Embedding
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator
accelerator = Accelerator()

class DataCollatorForSonarDistillation(DefaultDataCollator):
  def __init__(self, tokenizer: NllbTokenizer, return_tensors: str = "pt"):
    super().__init__(return_tensors)
    self.padding_value = tokenizer.vocab_info.pad_idx

  def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
    src_sentence_ids = [ torch.tensor(row["src_sentence_ids"], dtype=torch.int) for row in features ]
    src_sentence_ids = pad_sequence(src_sentence_ids, batch_first=True, padding_value=self.padding_value)
    tgt_sentence_ids = [ torch.tensor(row["tgt_sentence_ids"], dtype=torch.int) for row in features ]
    tgt_sentence_ids = pad_sequence(tgt_sentence_ids, batch_first=True, padding_value=self.padding_value)

    batch = {
        "src_sentence_ids": src_sentence_ids,
        "src_seq_lens": (src_sentence_ids != self.padding_value).sum(-1).unsqueeze(0),
        "src_batch_seq_len": torch.tensor([src_sentence_ids.shape[1]], dtype=torch.int),
        "tgt_sentence_ids": tgt_sentence_ids,
        "tgt_seq_lens": (tgt_sentence_ids != self.padding_value).sum(-1).unsqueeze(0),
        "tgt_batch_seq_len": torch.tensor([tgt_sentence_ids.shape[1]], dtype=torch.int)
    }
    return batch

class SonarDistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = MSELoss(reduction="sum")
        self.device = accelerator.device
        self.teacher.to(self.device)
        self.teacher.eval()

    def compute_loss(self, student, inputs, return_outputs=False):
        student_source_output = self.student(
            SequenceBatch(
                seqs=inputs["src_sentence_ids"].to(self.device),
                padding_mask=PaddingMask(inputs["src_seq_lens"][0], inputs["src_batch_seq_len"][0]).to(self.device)
              )
        )
        student_target_output = self.student(
            SequenceBatch(
                seqs=inputs["tgt_sentence_ids"].to(self.device),
                padding_mask=PaddingMask(inputs["tgt_seq_lens"][0], inputs["tgt_batch_seq_len"][0]).to(self.device)
              )
        )
        with torch.no_grad():
            teacher_target_output = self.teacher(
                SequenceBatch(
                    seqs=inputs["tgt_sentence_ids"].to(self.device),
                    padding_mask=PaddingMask(inputs["tgt_seq_lens"][0], inputs["tgt_batch_seq_len"][0]).to(self.device)
                  )
            )
        distillation_loss = self.loss_function(teacher_target_output.sentence_embeddings, student_source_output.sentence_embeddings) + self.loss_function(teacher_target_output.sentence_embeddings, student_target_output.sentence_embeddings)

        student_source_output_dict = {
            "encoded_seqs": student_source_output.encoded_seqs,
            "sentence_embeddings": student_source_output.sentence_embeddings,
            "padding_mask": student_source_output.padding_mask
        }

        return (distillation_loss, student_source_output_dict) if return_outputs else distillation_loss

def normalize_format_en_fr(examples):
    source_langs = ["fra_Latn"] * len(examples["translation"])
    target_langs = ["eng_Latn"] * len(examples["translation"])
    inputs = [example["fr"] for example in examples["translation"]]
    targets = [example["en"] for example in examples["translation"]]
    outputs = {
        "source_lang": source_langs,
        "source_sentence": inputs,
        "target_lang": target_langs,
        "target_sentence": targets
    }
    return outputs

def tokenize_inputs(examples, tokenizers, max_seq_len):
    src_sentence_ids = [ tokenizers[source_lang](sentence)[:max_seq_len] for source_lang, sentence in zip(examples["source_lang"], examples["source_sentence"]) ]
    tgt_sentence_ids = [ tokenizers[target_lang](sentence)[:max_seq_len] for target_lang, sentence in zip(examples["target_lang"], examples["target_sentence"]) ]
    model_inputs = {
        "src_sentence_ids": src_sentence_ids,
        "tgt_sentence_ids": tgt_sentence_ids
    }
    return model_inputs

def get_student_model_config():
    cfg = sonar_text_encoder_archs.get_config("basic")
    cfg.num_encoder_layers = cfg.num_decoder_layers = 12
    cfg.ffn_inner_dim = 4096
    return cfg

def get_nllb_checkpoint_encoder(nllb_checkpoint, student_config):
    nllb_checkpoint_encoder = {
        "state_dict": {},
        "embed_tokens": Embedding.from_pretrained(nllb_checkpoint["model"]["encoder.embed_tokens.weight"])
    }

    prefix = "encoder."
    for key, value in nllb_checkpoint['model'].items():
        if key.startswith(prefix):
            nllb_checkpoint_encoder["state_dict"][key[len(prefix):]] = value

    return convert_sonar_text_encoder_checkpoint(nllb_checkpoint_encoder, student_config)

if __name__=="__main__":
    
    print("Loading dataset...")

    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.001)

    print("Formatting dataset...")

    formatted_books = books.map(normalize_format_en_fr, batched=True)

    print("Loading tokenizers...")

    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
    tokenizers = {
        "eng_Latn": tokenizer.create_encoder(lang="eng_Latn"),
        "fra_Latn": tokenizer.create_encoder(lang="fra_Latn")
    }

    print("Tokenizing dataset...")

    max_seq_len = 512

    tokenized_books = formatted_books.map(tokenize_inputs, batched=True, fn_kwargs={"tokenizers": tokenizers, "max_seq_len": max_seq_len})
    tokenized_books = tokenized_books.remove_columns(["id", "translation", "source_lang", "source_sentence", "target_lang", "target_sentence"])

    print("Instantiating data collator...")

    data_collator = DataCollatorForSonarDistillation(tokenizer=tokenizer)

    print("Loading teacher model...")

    teacher_model = load_sonar_text_encoder_model("text_sonar_basic_encoder", device=accelerator.device)

    print("Instantiating student model...")

    student_config = get_student_model_config()
    student_model = create_sonar_text_encoder_model(student_config)

    print("Initializing student model...")

    nllb_checkpoint_path = os.path.join(os.environ["MODELS"], "nllb600m/nllb200densedst600mcheckpoint")
    nllb_checkpoint = torch.load(nllb_checkpoint_path)
    student_model_init = get_nllb_checkpoint_encoder(nllb_checkpoint, student_config)
    student_model.load_state_dict(student_model_init["model"])

    print("Training teacher model...")

    experiment_dir = os.path.join(os.environ["EXPERIMENTS"], "robust-embeddings/sonar/draft_experiment")

    training_args = TrainingArguments(
        output_dir=experiment_dir,
        fp16=False,
        logging_dir=f"{experiment_dir}/logs",
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="tensorboard",
        push_to_hub=False,
        per_device_train_batch_size=8,
        remove_unused_columns=False,
        max_steps=1000,
        save_steps=100,
        logging_steps=100,
        label_names=['tgt_sentence_ids', 'tgt_seq_lens', 'tgt_batch_seq_len'],
        prediction_loss_only=True
    )

    trainer = SonarDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        data_collator=data_collator,
    )

    trainer.train()

