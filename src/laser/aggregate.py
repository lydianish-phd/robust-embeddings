import pandas as pd
from scipy.stats import ttest_1samp
import os, argparse
from utils import (
    _name,
    _file
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to ugc directory with outputs of different seeds", type=str, default="/home/lnishimw/scratch/datasets/flores200/cleaned/ugc")
    parser.add_argument("--metrics", help="list of metrics", type=str, nargs="+", default=["cosine_distance", "xsim", "xsimpp"])
    parser.add_argument("--seeds", help="list of seeds", type=str, nargs="+", default=[ str(s) for s in range(100,110) ])
    parser.add_argument("--corpus-parts", help="name of corpus parts", type=str,  nargs="+", default= [ "dev", "devtest" ])
    args = parser.parse_args()

    for metric in args.metrics:
        for corpus_part in args.corpus_parts:
            output_file = os.path.join(args.input_dir, corpus_part + '-' + _file(metric))
            all_data = pd.DataFrame(columns=["dataset", "src-tgt", _name(metric)])
            for seed in args.seeds:
                input_file = os.path.join(args.input_dir, seed, corpus_part, _file(metric))
                data = pd.read_csv(input_file)
                data = data[data['src-tgt'] != "average"]
                all_data = pd.concat([all_data, data[all_data.columns]], ignore_index=True)
            all_data.to_csv(output_file)

            average_output_file = os.path.join(args.input_dir, corpus_part + '-avg-' + _file(metric))
            average_data = pd.DataFrame(columns=["src-tgt", _name(metric), "p-value", "not significant", "significant", "very significant", "highly significant"])
            average_data[["src-tgt", _name(metric)]] = all_data[["src-tgt", _name(metric)]].groupby("src-tgt").mean().reset_index()
            p_values = [ ttest_1samp(all_data[all_data["src-tgt"]==lang_pair][_name(metric)], 0)[1] for lang_pair in average_data["src-tgt"] ]
            average_data["p-value"] = p_values
            average_data["not significant"] = average_data["p-value"] >= 0.05
            average_data["significant"] = average_data["p-value"] < 0.05
            average_data["very significant"] = average_data["p-value"] < 0.01
            average_data["highly significant"] = average_data["p-value"] < 0.001
            average_data.to_csv(average_output_file)

                


