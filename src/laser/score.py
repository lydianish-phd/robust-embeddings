import os, configargparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.preprocessing import normalize
#from umap import UMAP
import pandas as pd
import re

seed = 0
dim = 1024
METRICS = ['checkpoint', 'model', 'cos', 'L1', 'L2']
SPACE_VIZ = ['PCA dim 1', 'PCA dim 2', 'UMAP dim 1', 'UMAP dim 2', 'T-SNE dim 1', 'T-SNE dim 2']


def read_embeddings(input_file, dim=1024, normalized=True):
    X = np.fromfile(input_file, dtype=np.float32, count=-1)
    X = np.resize(X, (X.shape[0] // dim, dim))
    if normalized:
        X = normalize(X)
    return X

def model_display_name(model):
    return model.replace("character", "c").replace("-student", "")

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("-i", "--input-dir", dest="input_dir", help="path to embeddings directory", type=str, default="/scratch/lnishimw/experiments/robust-embeddings/laser/experiment_025d/embeddings")
    parser.add("-o", "--output-dir", dest="output_dir", help="path to directory to save results", type=str, default="/scratch/lnishimw/experiments/robust-embeddings/laser/experiment_025d/scores")
    parser.add("-g", "--gold-file-name", dest="gold_file_name", help="file name of gold embeddings", type=str, default="cleaned.train.uncased.raw.ref.bin")
    parser.add("-t", "--teacher-model", dest="teacher_model", help="name of teacher model", type=str, default="laser")
    parser.add("-c", "--checkpoints", help="list of checkpoint names/numbers subdirectories. Leave empty to process all checkpoints.", nargs="+", type=str, default=[])
    args = parser.parse_args()
    
    checkpoints = args.checkpoints if args.checkpoints else [ f.name for f in os.scandir(args.input_dir) if f.is_dir() ]
    C = len(checkpoints)

    for checkpoint in checkpoints:
        print("Processing checkpoint_" + checkpoint)
        
        input_dir = os.path.join(args.input_dir, checkpoint)
        models = [ f.name for f in os.scandir(input_dir) if f.is_dir() ]
        model_names = [ model_display_name(m) for m in models ]
        embed_files = [ f.name for f in os.scandir(os.path.join(input_dir, args.teacher_model)) if f.name.endswith(".bin")]

        X_pca = PCA(n_components=2, random_state=seed)
#        X_umap = UMAP(n_components=2, init='random', random_state=seed)
        X_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=20, random_state=seed)

        print("Reading all embeddings and computing PCA, UMAP and distances wrt gold...")
        
        gold_output_dir =  os.path.join(args.output_dir, checkpoint, "gold")
        os.makedirs(gold_output_dir, exist_ok=True)

        X_gold = read_embeddings(os.path.join(input_dir, args.teacher_model, args.gold_file_name), normalized=True)
        N = X_gold.shape[0]
        M = len(models)
        data = {}
        for file in embed_files:
            all_embeddings = np.empty(shape=(N*(M+1), dim))
            all_embeddings[:N] = X_gold
            cos_data = np.zeros(shape=N*(M+1))
            l1_data = np.zeros(shape=N*(M+1))
            l2_data = np.zeros(shape=N*(M+1))
            for (i, model) in enumerate(models, start=1):
                all_embeddings[i*N:(i+1)*N] = read_embeddings(os.path.join(input_dir, model, file), normalized=True)
                cos_data[i*N:(i+1)*N] = paired_cosine_distances(all_embeddings[i*N:(i+1)*N], X_gold)
                l1_data[i*N:(i+1)*N] = paired_manhattan_distances(all_embeddings[i*N:(i+1)*N], X_gold)
                l2_data[i*N:(i+1)*N] = paired_euclidean_distances(all_embeddings[i*N:(i+1)*N], X_gold)
            data[file] = pd.DataFrame(columns=METRICS+SPACE_VIZ)
            data[file]['checkpoint'] = np.repeat(checkpoint, N*(M+1))
            data[file]['model'] = np.repeat(['laser-gold'] + model_names, N)
            data[file]['cos'] = cos_data
            data[file]['L1'] = l1_data
            data[file]['L2'] = l2_data
            data[file][['PCA dim 1', 'PCA dim 2']] = X_pca.fit_transform(all_embeddings)
 #           data[file][['UMAP dim 1', 'UMAP dim 2']] = X_umap.fit_transform(all_embeddings)
            data[file][['T-SNE dim 1', 'T-SNE dim 2']] = X_tsne.fit_transform(all_embeddings)

        for file in embed_files:    
            print("Saving results for", file)
            data[file].to_csv(os.path.join(gold_output_dir, file[:-4] + ".csv"))
            gold_checkpoint_data_file = os.path.join(args.output_dir, "gold-" + str(C) + "-chkpts-" + file[:-4] + ".csv")
            data[file].to_csv(gold_checkpoint_data_file, mode="a", header=not os.path.exists(gold_checkpoint_data_file))

        print("Reading all embeddings and computing PCA, UMAP and distances across standard-ugc file pairs...")
        
        paired_output_dir =  os.path.join(args.output_dir, checkpoint, "paired")
        os.makedirs(paired_output_dir, exist_ok=True)

        paired_files = [ (args.gold_file_name, f) for f in embed_files if f != args.gold_file_name ]
        paired_data = {}

        for file1, file2 in paired_files:
            cos_data = np.zeros(shape=N*M)
            l1_data = np.zeros(shape=N*M)
            l2_data = np.zeros(shape=N*M)
            for (i, model) in enumerate(models):
                embeddings1 = read_embeddings(os.path.join(input_dir, model, file1), normalized=True)
                embeddings2 = read_embeddings(os.path.join(input_dir, model, file2), normalized=True)
                cos_data[i*N:(i+1)*N] = paired_cosine_distances(embeddings1, embeddings2)
                l1_data[i*N:(i+1)*N] = paired_manhattan_distances(embeddings1, embeddings2)
                l2_data[i*N:(i+1)*N] = paired_euclidean_distances(embeddings1, embeddings2)
            paired_data[file2] = pd.DataFrame(columns=METRICS)
            paired_data[file2]['checkpoint'] = np.repeat(checkpoint, N*M)
            paired_data[file2]['model'] = np.repeat(model_names, N)
            paired_data[file2]['cos'] = cos_data
            paired_data[file2]['L1'] = l1_data
            paired_data[file2]['L2'] = l2_data

        for _, file in paired_files:
            print("Saving results for", file)
            paired_data[file].to_csv(os.path.join(paired_output_dir, file[:-4] + ".csv"))
            paired_checkpoint_data_file = os.path.join(args.output_dir, "paired-" + str(C) + "-chkpts-" + file[:-4] + ".csv")
            paired_data[file].to_csv(paired_checkpoint_data_file, mode="a", header=not os.path.exists(paired_checkpoint_data_file))
            
        print("Done...")
