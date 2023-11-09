from evaluate import load
import pandas as pd
from aggregate import _name, _file
from scipy.stats import ttest_1samp
import os

cer = load("cer")
wer = load("wer")
bleu = load("bleu")

def read_file(path):
    with open(path, "r") as f:
        doc = f.read().strip()
    return doc.split("\n")

def compute_metrics(f1, f2):
    predictions = read_file(f1)
    references = read_file(f2)
    return{
        "cer": cer.compute(predictions=predictions, references=references) * 100,
        "wer": wer.compute(predictions=predictions, references=references) * 100,
        "bleu": bleu.compute(predictions=predictions, references=references)["bleu"] * 100
    }


# input_dir = "/home/lnishimw/scratch/datasets"

# corpus = "rocsmt"
# corpus_parts = ["test"]

# corpus = "multilexnorm2021"
# corpus_parts = ["train", "dev", "test"]

# metric = "cer-wer-bleu"

# for corpus_part in corpus_parts:
#     results = []
#     files = [ file for file in os.scandir(os.path.join(input_dir, corpus, "cleaned", corpus_part)) if file.name.endswith(corpus_part) ]
#     print("Processing", corpus_part)
#     scores = compute_metrics(files[0].path, files[1].path)
#     results.append(scores)
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(os.path.join(input_dir, corpus, "cleaned", corpus_part, _file(metric)))

input_dir = "/home/lnishimw/scratch/datasets/"

CORPUS = "flores200/cleaned/ugc"
corpus_parts = ["dev", "devtest"]
METRIC = "cer-wer-bleu"
gold_file = "cleaned.eng_Latn"

# for seed in range(100,110):    
#     corpus = CORPUS + "/" + str(seed)
#     for corpus_part in corpus_parts:
#         results = []
#         files = [ file for file in os.scandir(os.path.join(input_dir, corpus, corpus_part)) if file.name.endswith(corpus_part) and not file.name.startswith(gold_file) ]
#         gold_file_path = os.path.join(input_dir, corpus, corpus_part, gold_file + '.' + corpus_part)
#         for file in files:
#             print("Processing", file.name)
#             scores = compute_metrics(file.path, gold_file_path)
#             scores["file"] = file.name
#             results.append(scores)

#         results_df = pd.DataFrame(results)
#         results_df.to_csv(os.path.join(input_dir, corpus, corpus_part, _file(METRIC)))

for corpus_part in corpus_parts:
    output_file = os.path.join(input_dir, CORPUS, corpus_part + '-' + _file(METRIC))
    all_data = pd.DataFrame(columns=["file", "cer", "wer", "bleu"])
    for seed in range(100,110):
        corpus = CORPUS + "/" + str(seed)
        input_file = os.path.join(input_dir, corpus, corpus_part, _file(METRIC))
        data = pd.read_csv(input_file)
        all_data = pd.concat([all_data, data[all_data.columns]], ignore_index=True)
    all_data.to_csv(output_file)

    for metric in all_data.columns[1:]:
        average_output_file = os.path.join(input_dir, CORPUS, corpus_part + '-avg-' + _file(metric))
        average_data = pd.DataFrame(columns=["file", _name(metric), "p-value", "not significant", "significant", "very significant", "highly significant"])
        average_data[["file", _name(metric)]] = all_data[["file", _name(metric)]].groupby("file").mean().reset_index()
        target_mean = 100 if metric == "bleu" else 0
        p_values = [ ttest_1samp(all_data[all_data["file"]==file][_name(metric)], target_mean)[1] for file in average_data["file"] ]
        average_data["p-value"] = p_values
        average_data["not significant"] = average_data["p-value"] >= 0.05
        average_data["significant"] = average_data["p-value"] < 0.05
        average_data["very significant"] = average_data["p-value"] < 0.01
        average_data["highly significant"] = average_data["p-value"] < 0.001
        average_data.to_csv(average_output_file)
