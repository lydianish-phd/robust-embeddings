import pandas as pd
from scipy.stats import ttest_ind
import os, configargparse
import numpy as np

def _name(m):
    if m == "xsimpp":
        return "xsim(++)"
    if m == "cosine_distance":
        return "cosdist"
    return m

def _file(m):
    return m + "_matrix.csv"

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("-i", "--input-dir", dest="input_dir", help="path to ugc directory with outputs of different seeds", type=str)
    parser.add("-m", "--model", help="name of model to compare to LASER", type=str)
    parser.add("-c", "--corpus", help="name of corpus", type=str, default="flores200")
    args = parser.parse_args()

    seeds = [ str(s) for s in range(100,110) ]
    corpus_parts = [ "dev", "devtest" ]
    metrics = ["cosine_distance", "xsim", "xsimpp"]

    for metric in metrics:
        for corpus_part in corpus_parts:
            output_file = os.path.join(args.input_dir, args.model, args.corpus, corpus_part + '-' + _file(metric))
            all_data = pd.DataFrame(columns=["dataset", "src-tgt", _name(metric)])
            for seed in seeds:
                input_file = os.path.join(args.input_dir, args.model, args.corpus, seed, corpus_part, _file(metric))
                data = pd.read_csv(input_file)
                data = data[data["src-tgt"] != "average"]
                all_data = pd.concat([all_data, data[all_data.columns]], ignore_index=True)
            all_data.to_csv(output_file)

            gold_score_file = os.path.join(args.input_dir, "laser", args.corpus, corpus_part + '-' + _file(metric))
            gold_data = pd.read_csv(gold_score_file)

            average_output_file = os.path.join(args.input_dir, args.model, args.corpus, corpus_part + '-avg-' + _file(metric))
            average_data = pd.DataFrame(columns=["src-tgt", _name(metric), "p-value", "not significant", "significant", "very significant", "highly significant"])
            average_data[["src-tgt", _name(metric)]] = all_data[["src-tgt", _name(metric)]].groupby("src-tgt").mean().reset_index()
            p_values = [ ttest_ind(all_data[all_data["src-tgt"]==lang_pair][_name(metric)], gold_data[gold_data["src-tgt"]==lang_pair][_name(metric)])[1] for lang_pair in average_data["src-tgt"] ]
            average_data["p-value"] = p_values
            average_data["not significant"] = average_data["p-value"] >= 0.05
            average_data["significant"] = average_data["p-value"] < 0.05
            average_data["very significant"] = average_data["p-value"] < 0.01
            average_data["highly significant"] = average_data["p-value"] < 0.001
            average_data.to_csv(average_output_file)

            


