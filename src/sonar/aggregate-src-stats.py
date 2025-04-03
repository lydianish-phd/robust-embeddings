import os, argparse, json
import pandas as pd
import numpy as np
from utils import (
    COLUMN_NAME_SEPARATOR,
    STATS_FILE_PREFIX,
    STATS_FILE_SUFFIX
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-files", type=str, required=True, help="Path to the input files", nargs="+")
    parser.add_argument("-o", "--output-dir", help="path to output directory", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "aggregated_src_stats.csv")
   
    print(f"Aggregating stats...")

    data = []
    
    for file in args.input_files:
        print(f"  - Processing {file}...")
        basedir, filename = os.path.split(file)
        stats_file = os.path.join(basedir, STATS_FILE_PREFIX + filename + STATS_FILE_SUFFIX)
        corpus_name = basedir.replace(f"{os.environ['DATASETS']}/", "").replace("/", COLUMN_NAME_SEPARATOR)
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
            stats["File"] = corpus_name + COLUMN_NAME_SEPARATOR + filename
            data.append(stats)
    
    df = pd.DataFrame(data)
    df = df[["File"] + [col for col in df.columns if col != "File"]]  # Ensure 'File' is the first column
    
    df.to_csv(output_file, index=False)
    print(f"Aggregated statistics saved to {output_file}")
