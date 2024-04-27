import nltk, os, argparse
import pandas as pd
from aggregate import _file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to input directory with tokenized files", type=str)
    parser.add_argument("-c", "--corpus-parts", help="list of corpus parts (train, dev, test)", type=str, nargs='+', default=[])
    args = parser.parse_args()

    metric = "ttr"
    
    print("Processing", args.corpus_parts)
    for corpus_part in args.corpus_parts:
        results = []
        files = [ file for file in os.scandir(os.path.join(args.input_dir, corpus_part)) if f"{corpus_part}.tok" in file.name ]
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
