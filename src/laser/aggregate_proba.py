import pandas as pd
import os, argparse

def _name(m):
    if m == "xsimpp":
        return "xsim(++)"
    if m == "cosine_distance":
        return "cosdist"
    return m

def _file(m):
    return m + "_matrix.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment scores directory", type=str)
    parser.add_argument("-m", "--models", help="list of models", type=str, nargs="+", default=["LASER"])
    parser.add_argument("-c", "--corpus", type=str)
    parser.add_argument("--corpus-parts", help="name of corpus parts", type=str,  nargs="+", default= [ "dev", "devtest" ])
    parser.add_argument("--metrics", help="list of metrics", type=str, nargs="+", default=["cosine_distance", "xsim", "xsimpp"])
    parser.add_argument("--seeds", help="list of seeds", type=str, nargs="+", default=[ str(s) for s in range(100,110) ])
    parser.add_argument("--probas", help="list of probabilities", type=str, nargs="+", default=[ str(p/10) for p in range(0, 10) ])
    args = parser.parse_args()

    for corpus_part in args.corpus_parts:
        for metric in args.metrics:
            output_file = os.path.join(args.input_dir, f'{args.corpus}-{corpus_part}-{_file(metric)}')
            all_data = pd.DataFrame(columns=["dataset", "src-tgt", _name(metric), "model", "proba"])
            for model in args.models:
                for seed in args.seeds:
                    for proba in args.probas:
                        input_file = os.path.join(args.input_dir, model, args.corpus, seed, proba, corpus_part, _file(metric))
                        data = pd.read_csv(input_file)
                        data = data[data['src-tgt'] != "average"]
                        data["model"] = model
                        data["proba"] = proba
                        all_data = pd.concat([all_data, data[all_data.columns]], ignore_index=True)
            
            all_data.to_csv(output_file)
                


