import os, argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizerFast
from rolaser_model import RoLaserModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import paired_cosine_distances
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", help="path to model directory", type=str)
    parser.add_argument("--src-file", help="path to source file", type=str)
    parser.add_argument("--tgt-file", help="path to target file", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizerFast.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
    model = RoLaserModel.from_pretrained(args.model_dir)
    model.eval()  

    with open(args.src_file) as f, open(args.tgt_file) as g:
        sources = f.readlines()
        targets = g.readlines()
    

    # Define a custom Dataset
    class TextDataset(Dataset):
        def __init__(self, src_texts, tgt_texts, tokenizer, max_length=512):
            assert len(src_texts) == len(tgt_texts), "Source and target texts must have the same length"
            self.src_texts = src_texts
            self.tgt_texts = tgt_texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.src_texts)

        def __getitem__(self, idx):
            src_inputs = self.tokenizer(
                self.src_texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            tgt_inputs = self.tokenizer(
                self.tgt_texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "src_input_ids": src_inputs["input_ids"],
                "src_attention_mask": src_inputs["attention_mask"],
                "tgt_input_ids": tgt_inputs["input_ids"],
                "tgt_attention_mask": tgt_inputs["attention_mask"],
            }

    # Create DataLoader for batching
    batch_size = 32
    dataset = TextDataset(sources, targets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model on the batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    src_embeddings = []
    tgt_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to the same device as the model
            src_input_ids = batch["src_input_ids"].squeeze(1).to(device)
            src_attention_mask = batch["src_attention_mask"].squeeze(1).to(device)
            tgt_input_ids = batch["tgt_input_ids"].squeeze(1).to(device)
            tgt_attention_mask = batch["tgt_attention_mask"].squeeze(1).to(device)
            
            # Forward pass
            src_outputs = model(input_ids=src_input_ids, attention_mask=src_attention_mask)
            tgt_outputs = model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask)
            
            # Collect predictions 
            src_embeddings.extend(src_outputs.pooler_output.cpu().numpy())
            tgt_embeddings.extend(tgt_outputs.pooler_output.cpu().numpy())

    # Print the results
    cos_dist = paired_cosine_distances(src_embeddings, tgt_embeddings)
    print("Average cosine distance:", cos_dist.mean())