import os, configargparse
from transformers import AutoModel, AutoTokenizer

FAIRSEQ_SPECIAL_TOKENS = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>"
]

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("-m", "--model-name", dest="model_name", help="name of pretrained model on HuggingFace", type=str, default="roberta-base")
    parser.add("-o", "--output-dir", dest="output_dir", help="path to directory to save plots", type=str, default="/home/lnishimw/scratch/models/checkpoints/")
    parser.add("--cvocab", help="save dictionary in cvocab format", default=False, action="store_true")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Saving model...")
    model = AutoModel.from_pretrained(args.model_name)
    model.save_pretrained(output_dir)
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(output_dir)

    if args.cvocab:
        print("Saving cvocab dictionary...")
        vocab_file = os.path.join(args.output_dir, f"{args.model_name}.cvocab")
        with open(vocab_file, "w") as f:
            for token in tokenizer.vocab:
                count = str(1)
                flag = "#fairseq:overwrite" if token in FAIRSEQ_SPECIAL_TOKENS else ""
                f.write(" ".join([token, count, flag, "\n"]))