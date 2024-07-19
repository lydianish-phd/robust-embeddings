import os

from transformers import TrainingArguments, Trainer, DefaultDataCollator, EarlyStoppingCallback
from datasets import load_dataset, interleave_datasets

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

class DataCollatorForSonarDistillation(DefaultDataCollator):
  def __init__(self, tokenizer: NllbTokenizer, return_tensors: str = "pt"):
    super().__init__(return_tensors)
    self.padding_value = tokenizer.vocab_info.pad_idx

  def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
    src_sentence_ids = [ row["src_sentence_ids"].clone().detach() for row in features ]
    src_sentence_ids = pad_sequence(src_sentence_ids, batch_first=True, padding_value=self.padding_value)
    tgt_sentence_ids = [ row["tgt_sentence_ids"].clone().detach() for row in features ]
    tgt_sentence_ids = pad_sequence(tgt_sentence_ids, batch_first=True, padding_value=self.padding_value)

    batch = {
        "src_sentence_ids": src_sentence_ids,
        "src_seq_lens": (src_sentence_ids != self.padding_value).sum(-1).unsqueeze(-1),
        "src_batch_seq_len": torch.tensor([src_sentence_ids.shape[1]], dtype=torch.int).unsqueeze(0).repeat((src_sentence_ids.shape[0], 1)),
        "tgt_sentence_ids": tgt_sentence_ids,
        "tgt_seq_lens": (tgt_sentence_ids != self.padding_value).sum(-1).unsqueeze(-1),
        "tgt_batch_seq_len": torch.tensor([tgt_sentence_ids.shape[1]], dtype=torch.int).unsqueeze(0).repeat((tgt_sentence_ids.shape[0], 1))
    }
    return batch

class SonarDistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.loss_function = MSELoss(reduction="sum")

    def compute_loss(self, student, inputs, return_outputs=False):
        student_source_output = student(
            SequenceBatch(
                seqs=inputs["src_sentence_ids"],
                padding_mask=PaddingMask(inputs["src_seq_lens"].flatten(), inputs["src_batch_seq_len"].flatten()[0])
              )
        )
        student_target_output = student(
            SequenceBatch(
                seqs=inputs["tgt_sentence_ids"],
                padding_mask=PaddingMask(inputs["tgt_seq_lens"].flatten(), inputs["tgt_batch_seq_len"].flatten()[0]) 
              )
        )
        with torch.no_grad():
            teacher_target_output = self.teacher(
                SequenceBatch(
                    seqs=inputs["tgt_sentence_ids"],
                    padding_mask=PaddingMask(inputs["tgt_seq_lens"].flatten(), inputs["tgt_batch_seq_len"].flatten()[0])
                  )
            )
        distillation_loss = self.loss_function(teacher_target_output.sentence_embeddings, student_source_output.sentence_embeddings) + self.loss_function(teacher_target_output.sentence_embeddings, student_target_output.sentence_embeddings)

        student_source_output_dict = {
            "encoded_seqs": student_source_output.encoded_seqs,
            "sentence_embeddings": student_source_output.sentence_embeddings,
            "padding_mask": student_source_output.padding_mask
        }

        return (distillation_loss, student_source_output_dict) if return_outputs else distillation_loss

def tokenize_and_pad(tokenizer, sentence, max_length, pad_idx):
    tensor = tokenizer(sentence)
    padding_length = max_length - tensor.size(0)
    if padding_length <= 0:
        return tensor[:max_length]
    padding = torch.full(torch.Size([padding_length]), fill_value=pad_idx)
    padded_tensor = torch.cat((tensor, padding), dim=0)
    return padded_tensor[:max_length]


def tokenize_inputs(examples, tokenizers, max_seq_len, pad_idx):
    src_sentence_ids = [ tokenize_and_pad(tokenizers[source_lang],sentence,max_seq_len,pad_idx) for source_lang, sentence in zip(examples["source_lang"], examples["source_sentence"]) ]
    tgt_sentence_ids = [ tokenize_and_pad(tokenizers[target_lang],sentence,max_seq_len,pad_idx) for target_lang, sentence in zip(examples["target_lang"], examples["target_sentence"]) ]
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

SEED = 42

if __name__=="__main__":

    accelerator = Accelerator()

    print("Loading datasets...")

    bilingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/bilingual/concatenated")
    monolingual_data_dir = os.path.join(os.environ["DATASETS"], "rosonar/monolingual/concatenated")

    data_en_fr_files = {
        "train": f"{bilingual_data_dir}/eng-fra/train/train.eng_Latn-fra_Latn_chunks/train.eng_Latn-fra_Latn-*.jsonl",
        "valid": f"{bilingual_data_dir}/eng-fra/valid/valid.eng_Latn-fra_Latn_chunks/valid.eng_Latn-fra_Latn-*.jsonl"
    }
    data_en_fr = load_dataset("json", data_files=data_en_fr_files, streaming=True)
    data_en_fr = data_en_fr.shuffle(seed=SEED, buffer_size=10_000)

    data_fr_files = {
        "train": f"{monolingual_data_dir}/fra/train/train.fra_Latn-fra_Latn_chunks/train.fra_Latn-fra_Latn-*.jsonl",
        "valid": f"{monolingual_data_dir}/fra/valid/valid.fra_Latn-fra_Latn_chunks/valid.fra_Latn-fra_Latn-*.jsonl"
    }
    data_fr = load_dataset("json", data_files=data_fr_files, streaming=True)
    data_fr = data_fr.shuffle(seed=SEED, buffer_size=10_000)

    data_en_files = {
        "train": f"{monolingual_data_dir}/eng/train/train.eng_Latn-eng_Latn_chunks/train.eng_Latn-eng_Latn-*.jsonl",
        "valid": f"{monolingual_data_dir}/eng/valid/valid.eng_Latn-eng_Latn_chunks/valid.eng_Latn-eng_Latn-*.jsonl"
    }
    data_en = load_dataset("json", data_files=data_en_files, streaming=True)
    data_en = data_en.shuffle(seed=SEED, buffer_size=10_000)

    all_train_data = interleave_datasets([data_en_fr["train"], data_fr["train"], data_en["train"]], probabilities=[0.625, 0.25, 0.125], seed=SEED)
    all_valid_data = interleave_datasets([data_en_fr["valid"], data_fr["valid"], data_en["valid"]], probabilities=[0.625, 0.25, 0.125], seed=SEED)

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
    dadata_collatorta_fr = accelerator.prepare(data_collator)

    print("Loading teacher model...")

    teacher_model = accelerator.prepare(load_sonar_text_encoder_model("text_sonar_basic_encoder"))

    print("Instantiating student model...")

    student_config = get_student_model_config()
    student_model = create_sonar_text_encoder_model(student_config)

    print("Initializing student model...")

    nllb_checkpoint_path = os.path.join(os.environ["MODELS"], "nllb600m/nllb200densedst600mcheckpoint")
    nllb_checkpoint = torch.load(nllb_checkpoint_path)
    student_model_init = get_nllb_checkpoint_encoder(nllb_checkpoint, student_config)
    student_model.load_state_dict(student_model_init["model"])

    print("Training student model...")

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
        save_total_limit=5,
        push_to_hub=False,
        per_device_train_batch_size=16,
        remove_unused_columns=False,
        max_steps=2000,
        save_steps=100,
        logging_steps=100,
        label_names=['tgt_sentence_ids', 'tgt_seq_lens', 'tgt_batch_seq_len'],
        prediction_loss_only=True, 
        seed=SEED
    )

    trainer = SonarDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_valid_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

