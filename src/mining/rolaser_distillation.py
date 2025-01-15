from transformers import DefaultDataCollator, Trainer, XLMRobertaTokenizerFast
from typing import Any, Dict, List
import torch
from rolaser_sentence_encoder import RoLaserSentenceEncoder
from torch.nn import MSELoss

class DataCollatorForRoLaserDistillation(DefaultDataCollator):
    def __init__(self, student_tokenizer: XLMRobertaTokenizerFast, max_length: int = 514):
        super().__init__(return_tensors="pt")
        self.student_tokenizer = student_tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]], return_tensors="pt") -> Dict[str, Any]:
        student_src_ids_and_masks = self.student_tokenizer(
            [ row["source_sentence"] for row in features ], 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors=return_tensors
        )
        student_tgt_ids_and_masks = self.student_tokenizer(
            [ row["target_sentence"] for row in features ], 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors=return_tensors
        )
    
        batch = {
            "teacher_tgt_embeds": torch.tensor([ row["teacher_tgt_embeds"] for row in features ]),
            "student_src_ids": student_src_ids_and_masks["input_ids"],
            "student_src_masks": student_src_ids_and_masks["attention_mask"],
            "student_tgt_ids": student_tgt_ids_and_masks["input_ids"],
            "student_tgt_masks": student_tgt_ids_and_masks["attention_mask"]
        }
        return batch

class RoLaserDistillationTrainer(Trainer):
    def __init__(
        self, 
        student_model: RoLaserSentenceEncoder,
        single_mse_loss: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)

        self.loss_function = MSELoss(reduction="sum")
        self.single_mse_loss = single_mse_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = inputs["student_src_ids"].shape[0]
        student_ids = torch.cat([inputs["student_src_ids"], inputs["student_tgt_ids"]], dim=0)
        student_masks = torch.cat([inputs["student_src_masks"], inputs["student_tgt_masks"]], dim=0)
        student_outputs = model(
            {
                "input_ids": student_ids,
                "attention_mask": student_masks
            }
        )["sentence_embedding"]
        student_source_output, student_target_output = torch.split(student_outputs, batch_size, dim=0)
        
        if self.single_mse_loss:
            teacher_target_embeddings = torch.cat([inputs["teacher_tgt_embeds"], inputs["teacher_tgt_embeds"]], dim=0)
            distillation_loss = self.loss_function(teacher_target_embeddings, student_outputs)
        else:
            distillation_loss = (
                self.loss_function(inputs["teacher_tgt_embeds"], student_source_output) + 
                self.loss_function(inputs["teacher_tgt_embeds"], student_target_output)
            )

        outputs = {
            "student_source_embeddings": student_source_output,
            "teacher_target_embeddings": inputs["teacher_tgt_embeds"]
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
