import os, argparse
import numpy as np
import pandas as pd
from score import read_embeddings
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from scipy.stats import spearmanr

def plot_circles(ax, data, dim_reduction="PCA"):
    if dim_reduction == "UMAP":
        columns = ["UMAP dim 1", "UMAP dim 2"]
    elif dim_reduction == "t-SNE":
        columns = ["t-SNE dim 1", "t-SNE dim 2"]
    else:
        columns = ["PCA dim 1", "PCA dim 2"]
    laser_std_row = data.loc[((data['model'] == 'LASER') & (data['type'] == 'std'))]
    circle_center = laser_std_row[columns].to_numpy().flatten()
    for i, row in data.iterrows():
        point_b = row[columns].to_numpy()
        radius = paired_euclidean_distances(circle_center.reshape(1, -1), point_b.reshape(1, -1))[0]
        ax.add_patch(plt.Circle(circle_center, radius, alpha=0.2, color='xkcd:lightblue'))
    sns.scatterplot(data, x=columns[0], y=columns[1], hue="model", style="type", s=50, linewidth=0, ax=ax)

def plot_langs(ax, data, dim_reduction="PCA"):
    if dim_reduction == "UMAP":
        columns = ["UMAP dim 1", "UMAP dim 2"]
    elif dim_reduction == "t-SNE":
        columns = ["t-SNE dim 1", "t-SNE dim 2"]
    else:
        columns = ["PCA dim 1", "PCA dim 2"]
    for _, row in data.iterrows():
        ax.text(row[columns[0]] - 0.01, row[columns[1]] + 0.01, row["lang"], c="tab:blue")

def distance_preservation_correlation(X1, data1, X2=None, data2=None):
    if X2 is None:
        X2 = X1
    if data2 is None:
        data2 = data1
    dist_orig = np.square(euclidean_distances(X1, X2)).flatten()
    dist_pca = np.square(euclidean_distances(data1[["PCA dim 1","PCA dim 2"]], data2[["PCA dim 1","PCA dim 2"]])).flatten()
    # dist_tsne = np.square(euclidean_distances(data1[["t-SNE dim 1","t-SNE dim 2"]], data2[["t-SNE dim 1","t-SNE dim 2"]])).flatten()
    # dist_umap = np.square(euclidean_distances(data1[["UMAP dim 1","UMAP dim 2"]], data2[["UMAP dim 1","UMAP dim 2"]])).flatten()
    pca_r = spearmanr(dist_orig, dist_pca)
    # tsne_r = spearmanr(dist_orig, dist_tsne))
    # umap_r = spearmanr(dist_orig, dist_umap))
    return pca_r #, tsne_r, umap_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--evaldir", help="path to experiment eval directory", type=str)
    args = parser.parse_args()

    laser_embed_dir = os.path.join(args.evaldir, "embeddings", "laser", "rocsmt", "test")
    rolaser_embed_dir = os.path.join(args.evaldir, "embeddings", "roberta-maxpool", "rocsmt", "test")
    c_rolaser_embed_dir = os.path.join(args.evaldir, "embeddings", "c-roberta-maxpool", "rocsmt", "test")
    
    multilingual_files = {
        "en": os.path.join(laser_embed_dir, "norm.en.test.bin"),
        "cs": os.path.join(laser_embed_dir, "ref.cs.test.bin"),
        "de": os.path.join(laser_embed_dir, "ref.de.test.bin"),
        "fr": os.path.join(laser_embed_dir, "ref.fr.test.bin"),
        "ru": os.path.join(laser_embed_dir, "ref.ru.test.bin"),
        "uk": os.path.join(laser_embed_dir, "ref.uk.test.bin")
    }
    
    noisy_files = [
        {
            "file": os.path.join(laser_embed_dir, "raw.en.test.bin"),
            "model": "LASER",
            "type": "ugc"
        },
        {
            "file": os.path.join(rolaser_embed_dir, "raw.en.test.bin"),
            "model": "RoLASER",
            "type": "ugc"
        },
        {
            "file": os.path.join(rolaser_embed_dir, "norm.en.test.bin"),
            "model": "RoLASER",
            "type": "std"
        },
        {
            "file": os.path.join(c_rolaser_embed_dir, "raw.en.test.bin"),
            "model": "c-RoLASER",
            "type": "ugc"
        },
        {
            "file": os.path.join(c_rolaser_embed_dir, "norm.en.test.bin"),
            "model": "c-RoLASER",
            "type": "std"
        }
    ]

    examples = [
        {
            "id": 986,
            "ugc": "eye wud liek 2 aply 4 vilage idot",
            "std": "I would like to apply for village idiot."
        },
        {
            "id": 760,
            "ugc": "But tmrw im no longer putting up with it.",
            "std": "But tomorrow I’m no longer putting up with it."            
        }
    ]

    seed = 0
    X_multi_files = [ read_embeddings(file) for file in multilingual_files.values() ] 
    X_noisy_files = [ read_embeddings(file["file"]) for file in noisy_files ]
    X = np.concatenate(X_multi_files + X_noisy_files)

    n_sentences =  X_multi_files[0].shape[0]

    X_multi_models = np.repeat(["LASER"], n_sentences * len(X_multi_files))
    X_noisy_models = np.repeat([file["model"] for file in noisy_files], n_sentences)

    X_multi_langs = np.repeat(list(multilingual_files.keys()), n_sentences)
    X_noisy_langs = np.repeat(["en"], n_sentences * len(X_noisy_files))

    X_multi_types = np.repeat(["std"] + ["tra"]*(len(X_multi_files)-1), n_sentences)
    X_noisy_types = np.repeat([file["type"] for file in noisy_files], n_sentences)

    pca = PCA(n_components=2, random_state=seed)
    # umap = UMAP(n_components=2, init='random', random_state=seed)
    # tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=20, random_state=seed)

    data = pd.DataFrame()
    data[["PCA dim 1", "PCA dim 2"]] = pca.fit_transform(X)
    # data[["t-SNE dim 1", "t-SNE dim 2"]] = tsne.fit_transform(X)
    # data[["UMAP dim 1", "UMAP dim 2"]] = umap.fit_transform(X)
    data["model"] = np.concatenate([X_multi_models, X_noisy_models])
    data["lang"] = np.concatenate([X_multi_langs, X_noisy_langs])
    data["type"] = np.concatenate([X_multi_types, X_noisy_types])
    data["sentence"] = [ f"sent {i%n_sentences}" for i in range(X.shape[0]) ]

    print("Plotting sentences in reduced embedding space...")
    for i, example in enumerate(examples):
        subset = data[data["sentence"] == f"sent {example['id']}"]
        subset = subset[subset["model"] != "c-RoLASER"]
        subset = subset.loc[subset["type"].map({"std": 1, "ugc": 2, "tra": 3}).sort_values().index]
        subset_multi = subset[subset["type"] == "tra"]
        plt.clf()
        fig, ax = plt.subplots(figsize=(6,6))
        plot_circles(ax, subset)
        plot_langs(ax, subset_multi)
        ax.set_title(f"UGC: {example['ugc']}\n(STD: {example['std']})")
        ax.set_aspect("equal")
        ax.legend(loc='lower right')
        fig.tight_layout(pad=0.25)
        plt.savefig(os.path.join(args.evaldir, "scores", f"pca_distances_{i}.pdf"), format="pdf")
    
    print("Computing distance preservation correlation...")
    pca_r = distance_preservation_correlation(X, data)
    
    with open(os.path.join(args.evaldir, "scores", "pca_distances_correlation.txt"), "w") as f:
        f.write(f"PCA: {pca_r}\n")
