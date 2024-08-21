import os
from transformers import DefaultDataCollator, Trainer
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.models.sequence import SequenceBatch, PaddingMask
from sonar.models.sonar_text.builder import sonar_text_encoder_archs, create_sonar_text_encoder_model
from sonar.models.sonar_text.loader import convert_sonar_text_encoder_checkpoint
from typing import Any, Dict, List
import torch
from torch.nn import MSELoss, Embedding
from torch.nn.utils.rnn import pad_sequence
from safetensors import safe_open

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
        print(self.teacher)
        self.loss_function = MSELoss(reduction="sum")

    def compute_loss(self, model, inputs, return_outputs=False):
        student_source_output = model(
            SequenceBatch(
                seqs=inputs["src_sentence_ids"],
                padding_mask=PaddingMask(inputs["src_seq_lens"].flatten(), inputs["src_batch_seq_len"].flatten()[0])
              )
        )
        student_target_output = model(
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

        outputs = {
            "student_source_embeddings": student_source_output.sentence_embeddings,
            "teacher_target_embeddings": teacher_target_output.sentence_embeddings
        }

        return (distillation_loss, outputs) if return_outputs else distillation_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            logits = outputs["student_source_embeddings"]
            labels = outputs["teacher_target_embeddings"]
           
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    loss = MSELoss(reduction="sum")(predictions, labels)
    return { "loss": loss }

def _tokenize_and_pad(tokenizer, sentence, max_length, pad_idx):
    tensor = tokenizer(sentence)
    padding_length = max_length - tensor.size(0)
    if padding_length <= 0:
        return tensor[:max_length]
    padding = torch.full(torch.Size([padding_length]), fill_value=pad_idx)
    padded_tensor = torch.cat((tensor, padding), dim=0)
    return padded_tensor[:max_length]

def tokenize_inputs(examples, tokenizers, max_seq_len, pad_idx):
    src_sentence_ids = [ _tokenize_and_pad(tokenizers[source_lang],sentence,max_seq_len,pad_idx) for source_lang, sentence in zip(examples["source_lang"], examples["source_sentence"]) ]
    tgt_sentence_ids = [ _tokenize_and_pad(tokenizers[target_lang],sentence,max_seq_len,pad_idx) for target_lang, sentence in zip(examples["target_lang"], examples["target_sentence"]) ]
    model_inputs = {
        "src_sentence_ids": src_sentence_ids,
        "tgt_sentence_ids": tgt_sentence_ids
    }
    return model_inputs

def _get_student_model_config():
    cfg = sonar_text_encoder_archs.get_config("basic")
    cfg.num_encoder_layers = cfg.num_decoder_layers = 12
    cfg.ffn_inner_dim = 4096
    return cfg

def _get_nllb_checkpoint_encoder(checkpoint_file, student_config):
    nllb_checkpoint = torch.load(checkpoint_file)
    nllb_checkpoint_encoder = {
        "state_dict": {},
        "embed_tokens": Embedding.from_pretrained(nllb_checkpoint["model"]["encoder.embed_tokens.weight"])
    }
    prefix = "encoder."
    for key, value in nllb_checkpoint['model'].items():
        if key.startswith(prefix):
            nllb_checkpoint_encoder["state_dict"][key[len(prefix):]] = value
    return convert_sonar_text_encoder_checkpoint(nllb_checkpoint_encoder, student_config)

def _get_rosonar_checkpoint_encoder(checkpoint_dir):
    checkpoint_file = f"{checkpoint_dir}/model.safetensors"
    encoder = { 
        "model": {}
    }
    with safe_open(checkpoint_file, framework="pt", device=0) as f:
        for k in f.keys():
            encoder["model"][k] = f.get_tensor(k)
    return encoder
        
def load_student_encoder_from_checkpoint(checkpoint_file_or_dir):
    student_config = _get_student_model_config()
    student_model = create_sonar_text_encoder_model(student_config)
    if os.path.isdir(checkpoint_file_or_dir):
        student_model_init = _get_rosonar_checkpoint_encoder(checkpoint_file_or_dir)
    else:
        student_model_init = _get_nllb_checkpoint_encoder(checkpoint_file_or_dir, student_config)
    student_model.load_state_dict(student_model_init["model"])
    return student_model
