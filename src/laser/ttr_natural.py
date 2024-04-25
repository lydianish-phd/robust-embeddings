import nltk, os, argparse
import pandas as pd
from aggregate import _file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to input directory with tokenized files", type=str)
    parser.add_argument("-c", "--corpus", help="name of corpus: multilexnorm2021 or rocsmt", type=str)
    args = parser.parse_args()

    if args.corpus == "multilexnorm2021":
        corpus_parts = ["train", "dev", "test"]
    elif args.corpus == "rocsmt":
        corpus_parts = ["test"]
    else:
        corpus_parts = [""]
    metric = "ttr"
    
    print("Processing", args.corpus)
    for corpus_part in corpus_parts:
        results = []
        files = [ file for file in os.scandir(os.path.join(args.input_dir, corpus_part)) if file.name.endswith(f"{corpus_part}.tok") ]
        for file in files:
            print(file.name)
            with open(file.path, "r") as f:
                doc = f.read().replace('\n',' ')

            tokens = doc.split()
            types=nltk.Counter(tokens)
            ttr = (len(types)/len(tokens))*100
            results.append({"file": file.name[:-4], "tokens": len(tokens), "types": len(types), "ttr": ttr})

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(args.input_dir, f"{corpus_part}-{_file(metric)}"))
