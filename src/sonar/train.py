from datasets import load_dataset
from sonar.models.sonar_text.loader import load_sonar_text_encoder_model, load_sonar_tokenizer
from sonar.models.sonar_text.builder import create_sonar_text_encoder_model, sonar_text_encoder_archs
from transformers import TrainingArguments, Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F

books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)

tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")
en_tokenizer = tokenizer.create_encoder(lang="eng_Latn")
fr_tokenizer = tokenizer.create_encoder(lang="fra_Latn")

def normalize_format_en_fr(examples):
    source_langs = ["eng_Latn"] * len(examples["translation"])
    target_langs = ["fra_Latn"] * len(examples["translation"])
    inputs = [example["en"] for example in examples["translation"]]
    targets = [example["fr"] for example in examples["translation"]]
    outputs = {
        "source_lang": source_langs,
        "source_sentence": inputs,
        "target_lang": target_langs,
        "target_sentence": targets
    }
    return outputs

formatted_books = books.map(normalize_format_en_fr, batched=True)

tokenizers = {
    "eng_Latn": en_tokenizer,
    "fra_Latn": fr_tokenizer
}

max_seq_len = 512

def preprocess_function(examples):
    src_sentence_ids = [ tokenizers[source_lang](sentence)[:max_seq_len] for source_lang, sentence in zip(examples["source_lang"], examples["source_sentence"]) ]
    tgt_sentence_ids = [ tokenizers[target_lang](sentence)[:max_seq_len] for target_lang, sentence in zip(examples["target_lang"], examples["target_sentence"]) ]
    model_inputs = {
        "src_sentence_ids": src_sentence_ids,
        "tgt_sentence_ids": tgt_sentence_ids
    }
    return model_inputs

tokenized_books = formatted_books.map(preprocess_function, batched=True)
tokenized_books.remove_columns(['id', 'translation', 'source_lang', 'source_sentence', 'target_lang', 'target_sentence'])

from transformers import DefaultDataCollator
from fairseq2.models.sequence import SequenceBatch
from fairseq2.data import Collater
from sonar.inference_pipelines.utils import extract_sequence_batch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

class DataCollatorForSonarDistillation(DefaultDataCollator):
  def __init__(self, tokenizer, device="cuda", return_tensors='pt'):
    super().__init__(return_tensors)
    self.tokenizer = tokenizer
    self.device = device

  def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
     input_ids = Collater(self.tokenizer.vocab_info.pad_idx)(features["input_ids"])
     labels = Collater(self.tokenizer.vocab_info.pad_idx)(features["labels"])
     features["input_ids"] = extract_sequence_batch(input_ids, self.device)
     features["labels"] = extract_sequence_batch(labels, self.device)
     return features

data_collator = DataCollatorForSonarDistillation(tokenizer=tokenizer)

class SonarEmbeddingDistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.MSELoss(reduction="batchmean")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(self.device)
        self.teacher.eval()

    def compute_loss(self, student, inputs, return_outputs=False):
        print(inputs)
        student_source_output = self.student(inputs["src_sentence_ids"])
        student_source_embeddings = student_source_output.map(lambda x: x.sentence_embeddings.to(self.device))
        student_target_output = self.student(inputs["tgt_sentence_ids"])
        student_target_embeddings = student_source_output.map(lambda x: x.sentence_embeddings.to(self.device))

        with torch.no_grad():
          teacher_source_output = self.teacher(inputs["src_sentence_ids"])
          teacher_source_embeddings = teacher_source_output.map(lambda x: x.sentence_embeddings.to(self.device))

        distillation_loss = self.loss_function(teacher_source_embeddings, student_source_embeddings) + self.loss_function(teacher_source_embeddings, student_target_embeddings)

        return (distillation_loss, student_source_output, student_target_output) if return_outputs else distillation_loss

teacher_model = load_sonar_text_encoder_model("text_sonar_basic_encoder", device="cuda")

cfg = sonar_text_encoder_archs.get_config("basic")
cfg.num_encoder_layers = cfg.num_decoder_layers = 12
cfg.ffn_inner_dim = 4096
student_model = create_sonar_text_encoder_model(cfg)

training_args = TrainingArguments(
    output_dir="my-awesome-model",
    num_train_epochs=3,
    fp16=True,
    logging_dir=f"logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    push_to_hub=False
)

trainer = SonarEmbeddingDistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

