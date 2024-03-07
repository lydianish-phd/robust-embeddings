import torch, argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="path to raw input file", type=str, default="/home/lnishimw/scratch/datasets/flores200/cleaned/dev/cleaned.eng_Latn.dev")
    args = parser.parse_args()

    device = "cuda"
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.input_file, "r") as f:
        data = f.read().split("\n")

    encodings = tokenizer(data, return_tensors="pt", padding="longest")
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    ppl = torch.exp(neg_log_likelihood)

    print("The perplexity of", args.input_file, "is", ppl)