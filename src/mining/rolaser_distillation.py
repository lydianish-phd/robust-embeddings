import os
from transformers import DefaultDataCollator, Trainer, XLMRobertaTokenizer
from typing import Any, Dict, List
import torch
from laser_encoders.models import SentenceEncoder
from laser_encoders.laser_tokenizer import (
    LaserTokenizer,
    NON_PRINT_CHARS
)
from rolaser_model import RoLaserModel
from torch.nn import MSELoss

class DataCollatorForRoLaserDistillation(DefaultDataCollator):
    def __init__(
        self, 
        teacher_tokenizer: LaserTokenizer, 
        student_tokenizer: XLMRobertaTokenizer, 
        max_length: int = 512,
        return_tensors: str = "pt"
    ):
        super().__init__(return_tensors)
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        src_sents = [ row["source_sentence"] for row in features ]
        tgt_sents = [ row["target_sentence"] for row in features ]

        teacher_tgt_pieces = self.teacher_tokenizer.tokenize_batch(tgt_sents)

        # preproc_src_sents = [ _preprocess_sentence(s, self.teacher_tokenizer) for s in src_sents  ]
        # preproc_tgt_sents = [ _preprocess_sentence(s, self.teacher_tokenizer) for s in tgt_sents  ]

        rt = return_tensors if return_tensors is not None else self.return_tensors
        student_src_ids_and_masks = self.student_tokenizer(src_sents, padding=True, max_length=self.max_length, truncation=True, return_tensors=rt)
        student_tgt_ids_and_masks = self.student_tokenizer(tgt_sents, padding=True, max_length=self.max_length, truncation=True, return_tensors=rt)

        return {
            "teacher_tgt_pieces": teacher_tgt_pieces,
            "student_src_ids_and_masks": student_src_ids_and_masks,
            "student_tgt_ids_and_masks": student_tgt_ids_and_masks
        }

class RoLaserDistillationTrainer(Trainer):
    def __init__(
        self, 
        teacher_model: SentenceEncoder,
        student_model: RoLaserModel,
        *args,
        **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.loss_function = MSELoss(reduction="sum")

    def compute_loss(self, model, inputs, return_outputs=False):
        student_source_output = model(**inputs["student_src_ids_and_masks"]).pooler_output
        student_target_output = model(**inputs["student_tgt_ids_and_masks"]).pooler_output
        
        with torch.no_grad():
            teacher_target_output = torch.tensor(self.teacher.encode_sentences(inputs["teacher_tgt_pieces"]))
        
        distillation_loss = self.loss_function(teacher_target_output.sentence_embeddings, student_source_output.sentence_embeddings) + self.loss_function(teacher_target_output.sentence_embeddings, student_target_output.sentence_embeddings)

        outputs = {
            "student_source_embeddings": student_source_output,
            "teacher_target_embeddings": teacher_target_output
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

def _preprocess_sentence(text, tokenizer):
    # Copied from laser_encoders.laser_tokenizer.LaserTokenizer.tokenize
    sentence_text = "".join([c if c not in NON_PRINT_CHARS else " " for c in text])
    if tokenizer.normalize_punct:
        sentence_text = tokenizer.moses_punct_normalizer.normalize(sentence_text)
    if tokenizer.descape:
        sentence_text = tokenizer.moses_detokenizer.unescape_xml(text=sentence_text)
    if tokenizer.lower_case:
        sentence_text = sentence_text.lower()
    return sentence_text

