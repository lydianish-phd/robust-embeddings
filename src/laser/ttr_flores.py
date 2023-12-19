import nltk, os, argparse
import pandas as pd
from aggregate import _name, _file
from scipy.stats import ttest_1samp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to input directory with tokenized files", type=str, required=True)
    args = parser.parse_args()

    corpus = "flores200"
    corpus_parts = ["dev", "devtest"]
    metric = "ttr"
    gold_file = "eng_Latn"

    print("Processing", corpus)
    for seed in range(100,110):    
        for corpus_part in corpus_parts:
            results = []
            files = [ file for file in os.scandir(os.path.join(args.input_dir, seed, corpus_part)) if file.name.endswith(f"{corpus_part}.tok") ]
            for file in files:
                print(file.name)
                with open(file.path, "r") as f:
                    doc = f.read().replace('\n',' ')

                tokens = doc.split()
                types=nltk.Counter(tokens)
                ttr = (len(types)/len(tokens))*100
                results.append({"file": file.name[:-4], "tokens": len(tokens), "types": len(types), "ttr": ttr})

            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(args.input_dir, seed, corpus_part, _file(metric)))
    
    print("Aggregating results...")
    for corpus_part in corpus_parts:
        output_file = os.path.join(args.input_dir, corpus_part + '-' + _file(metric))
        all_data = pd.DataFrame(columns=["file", "types", "tokens", "ttr"])
        for seed in range(100,110):
            input_file = os.path.join(args.input_dir, seed, corpus_part, _file(metric))
            data = pd.read_csv(input_file)
            all_data = pd.concat([all_data, data[all_data.columns]], ignore_index=True)
        all_data.to_csv(output_file)

        for metric in all_data.columns[1:]:
            average_output_file = os.path.join(args.input_dir, corpus_part + '-avg-' + _file(metric))
            average_data = pd.DataFrame(columns=["file", _name(metric), "p-value", "not significant", "significant", "very significant", "highly significant"])
            average_data[["file", _name(metric)]] = all_data[["file", _name(metric)]].groupby("file").mean().reset_index()
            target_mean = all_data[all_data["file"]==gold_file+"."+corpus_part][_name(metric)].mean()
            p_values = [ ttest_1samp(all_data[all_data["file"]==file][_name(metric)], target_mean)[1] for file in average_data["file"] ]
            average_data["p-value"] = p_values
            average_data["not significant"] = average_data["p-value"] >= 0.05
            average_data["significant"] = average_data["p-value"] < 0.05
            average_data["very significant"] = average_data["p-value"] < 0.01
            average_data["highly significant"] = average_data["p-value"] < 0.001
            average_data.to_csv(average_output_file)
