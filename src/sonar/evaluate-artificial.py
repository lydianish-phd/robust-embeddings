import os, argparse, json
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-c", "--corpus", type=str)
    parser.add_argument("-l", "--lang-pair", type=str)
    parser.add_argument("--src-dir", type=str)
    parser.add_argument("--src-file-name", type=str)
    parser.add_argument("--sys-file-name", type=str)
    parser.add_argument("--ref-file", type=str)
    parser.add_argument("-m", "--models", type=str, nargs="+")
    parser.add_argument("-s", "--seeds", type=int, nargs="+")
    parser.add_argument("-p", "--probas", type=float, nargs="+")
    args = parser.parse_args()

    print("Loading metric models...")
    bleu_model = BLEU()
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    
    print("Loading data...")

    with open (args.ref_file) as f:
        ref_data = [ line.strip() for line in f.readlines() ]

    print("Computing BLEU and COMET scores...")
    for seed in args.seeds:
        for proba in args.probas:
            src_file = os.path.join(args.src_dir, str(seed), str(proba), "ugc", args.src_file_name)
            for model in args.models:
                model_output_dir = os.path.join(args.input_dir, "outputs", model, args.corpus, args.lang_pair, str(seed), str(proba))
                sys_file = os.path.join(model_output_dir, args.sys_file_name)

                with open (src_file) as f:
                    src_data = [ line.strip() for line in f.readlines() ]
                with open (sys_file) as f:
                    sys_data = [ line.strip() for line in f.readlines() ]
                data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_data, sys_data, ref_data)]
        
                scores = {
                    "bleu": bleu_model.corpus_score(sys_data, [ref_data]).score,
                    "comet": comet_model.predict(data, batch_size=32, gpus=1)[1]
                }

                output_file = f"{sys_file}.json"
                with open(output_file, 'w') as f:
                    json.dump(scores, f)

