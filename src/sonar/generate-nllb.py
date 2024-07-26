import os, argparse
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str)
    parser.add_argument("-s", "--src-lang", type=str)
    parser.add_argument("-t", "--tgt-lang", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("-m", "--model-name", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    output_file = os.path.join(args.output_dir, f"{file_name}.out")

    print("Loading translation model and tokenizer...")
    tokenizer = NllbTokenizer.from_pretrained(
        args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(torch.device("cuda"))
    
    print("Reading input sentences...")
    with open(args.input_file) as f:
        data = f.readlines()
    sentences = [line.strip() for line in data]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)

    print("Generating outputs...")
    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang),
        max_length=512
    )
    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    print("Writing output sentences...")
    with open(output_file, "w") as f:
        for output in outputs:
            f.write(output + "\n")



