import os, argparse, json
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", type=str)
    parser.add_argument("--sys-file", type=str)
    parser.add_argument("--ref-file", type=str)
    args = parser.parse_args()
    with open (args.src_file) as f:
        src_data = [ line.strip() for line in f.readlines() ]

    with open (args.sys_file) as f:
        sys_data = [ line.strip() for line in f.readlines() ]

    with open (args.ref_file) as f:
        ref_data = [ line.strip() for line in f.readlines() ]

    data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_data, sys_data, ref_data)]

    bleu_model = BLEU()
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    
    print("Computing BLEU and COMET scores...")
    scores = {
        "bleu": bleu_model.corpus_score(sys_data, [ref_data]).score,
        "comet": comet_model.predict(data, batch_size=32, gpus=1)[1]
    }

    output_file = f"{args.sys_file}.json"
    with open(output_file, 'w') as f:
        json.dump(scores, f)

