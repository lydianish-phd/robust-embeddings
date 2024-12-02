from transformers import DefaultDataCollator, Trainer
from typing import Any, Dict, List
import torch
from rolaser_sentence_encoder import RoLaserSentenceEncoder
from torch.nn import MSELoss

class DataCollatorForRoLaserDistillation(DefaultDataCollator):
    def __init__(self):
        super().__init__(return_tensors="pt")

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        batch = {
            "teacher_tgt_embeds": torch.tensor([ row["teacher_tgt_embeds"] for row in features ]),
            "student_src_ids": torch.tensor([ row["student_src_ids"] for row in features ], dtype=torch.int),
            "student_src_masks": torch.tensor([ row["student_src_masks"] for row in features ], dtype=torch.int),
            "student_tgt_ids": torch.tensor([ row["student_tgt_ids"] for row in features ], dtype=torch.int),
            "student_tgt_masks": torch.tensor([ row["student_tgt_masks"] for row in features ], dtype=torch.int)
        }
        return batch

class RoLaserDistillationTrainer(Trainer):
    def __init__(
        self, 
        student_model: RoLaserSentenceEncoder,
        *args,
        **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)

        self.loss_function = MSELoss(reduction="sum")

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
