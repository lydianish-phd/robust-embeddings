import os, configargparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GOLD_FILES = [ 'train.uncased.raw.ref', 'test.uncased.raw.ref', 'eng_Latn.dev', 'eng_Latn.devtest' ]
METRICS = ['checkpoint', 'model', 'cos', 'L1', 'L2']
MAX_BOXPLOT_GROUPS = 4

def plot_lineplot(data,x,y,filename,src,tgt,hue=None):
    plt.clf()
    g = sns.lineplot(data, x=x, y=y, hue=hue, marker='o')
    if x == 'checkpoint':
        g.set_xticklabels([checkpoint_display_name(str(int(c))) for c in g.get_xticks()])
    plt.ylabel("avg " + y + "_distance(" + src + ", " + tgt + ")")
    plt.xlabel("steps")
    plt.savefig(filename)

def plot_lineplot_gold(data,x,y,filename,data_type,hue=None):
    return plot_lineplot(data,x,y,filename,r"$\it{model}$[" + data_type + "]", r"laser[std]",hue)

def plot_lineplot_paired(data,x,y,filename,hue=None):
    return plot_lineplot(data,x,y,filename,r"$\it{model}$[ugc]", r"$\it{model}$[std]",hue)

def plot_boxplot(data,x,y,filename,src,tgt,hue=None):
    plt.clf()
    g = sns.boxplot(data, x=x, y=y, hue=hue)
    if hue == 'checkpoint':
        for i, text in enumerate(g.legend_.texts):
            label = text.get_text()
            g.legend_.texts[i].set_text(checkpoint_display_name(label))
    plt.ylabel(y + "_distance(" + src + ", " + tgt + ")")
    plt.xlabel(None)
    plt.savefig(filename)

def plot_boxplot_gold(data,x,y,filename,data_type,hue=None):
    return plot_boxplot(data,x,y,filename,r"$\it{model}$[" + data_type + "]", r"laser[std]",hue)

def plot_boxplot_paired(data,x,y,filename,hue=None):
    return plot_boxplot(data,x,y,filename,r"$\it{model}$[ugc]", r"$\it{model}$[std]",hue)

def plot_scatterplot(data,x,y,hue,filename):
    plt.clf()
    _ = sns.scatterplot(data, x=x, y=y, hue=hue)
    plt.savefig(filename)
        
def checkpoint_display_name(checkpoint):
    if checkpoint.isdigit():
        num = float('{:.3g}'.format(int(checkpoint)))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'k', 'M', 'B', 'T'][magnitude])
    return checkpoint

def get_data_type(filename):
    if any([ f in filename for f in GOLD_FILES ]):
        return 'std'
    return 'ugc'

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("-i", "--input-dir", dest="input_dir", help="path to results directory", type=str, default="/scratch/lnishimw/experiments/robust-embeddings/laser/experiment_025d/scores")
    parser.add("-o", "--output-dir", dest="output_dir", help="path to directory to save plots", type=str, default="/scratch/lnishimw/experiments/robust-embeddings/laser/experiment_025d/plots")
    parser.add("-p", "--plot-types", dest="plot_types", help="types of plots", nargs="+", type=str, default=['gold-metrics', 'gold-space-viz', 'paired-metrics'])
    parser.add("-c", "--checkpoints", help="list of checkpoint names/numbers subdirectories. Leave empty to process all checkpoints.", nargs="+", type=str, default=[])
    args = parser.parse_args()
    
    checkpoints = args.checkpoints if args.checkpoints else [ f for f in os.scandir(args.input_dir) if f.is_dir() ]

    for checkpoint in checkpoints:
        print("Processing checkpoint_" + checkpoint.name)        
        input_dir = checkpoint.path
        if 'gold-metrics' in args.plot_types or 'gold-space-viz' in args.plot_types:
            gold_input_dir = os.path.join(input_dir, "gold")
            gold_output_dir = os.path.join(args.output_dir, checkpoint.name, "gold")
            os.makedirs(gold_output_dir, exist_ok=True)

            for file in os.scandir(gold_input_dir):    
                data = pd.read_csv(file.path)
                data_type = get_data_type(file.name)
                N = int(data.shape[0] / data['model'].unique().size)

                print("Saving plots for gold/" + file.name)

                if 'gold-metrics' in args.plot_types:
                    print("\t- cosine distance")
                    plot_boxplot_gold(data[N:], 'model', 'cos', os.path.join(gold_output_dir, "cos-" + file.name[:-4] + ".pdf"), data_type)
                    
                    print("\t- manhattan (l1) distance")
                    plot_boxplot_gold(data[N:], 'model', 'L1', os.path.join(gold_output_dir, "l1-" + file.name[:-4] + ".pdf"), data_type)

                    print("\t- euclidian (l2) distance")
                    plot_boxplot_gold(data[N:], 'model', 'L2', os.path.join(gold_output_dir, "l2-" + file.name[:-4] + ".pdf"), data_type)
                
                if 'gold-space-viz' in args.plot_types:
                    print("\t- PCA")
                    plot_scatterplot(data, 'PCA dim 1', 'PCA dim 2', 'model', os.path.join(gold_output_dir, "pca-" + file.name[:-4] + ".pdf"))

                    print("\t- UMAP")
                    plot_scatterplot(data, 'UMAP dim 1', 'UMAP dim 2', 'model', os.path.join(gold_output_dir, "umap-" + file.name[:-4] + ".pdf"))

                    print("\t- T-SNE")
                    plot_scatterplot(data, 'T-SNE dim 1', 'T-SNE dim 2', 'model', os.path.join(gold_output_dir, "tsne-" + file.name[:-4] + ".pdf"))

                plt.close('all')
        

        if 'paired-metrics' in args.plot_types:
            paired_input_dir = os.path.join(input_dir, "paired")
            paired_output_dir = os.path.join(args.output_dir, checkpoint.name, "paired")
            os.makedirs(paired_output_dir, exist_ok=True)

            for file in os.scandir(paired_input_dir):    
                data = pd.read_csv(file.path)
                N = int(data.shape[0] / data['model'].unique().size)

                print("Saving plots for paired/" + file.name)
                
                print("\t- cosine distance")
                plot_boxplot_paired(data, 'model', 'cos', os.path.join(paired_output_dir, "cos-" + file.name[:-4] + ".pdf"))
                
                print("\t- manhattan (l1) distance")
                plot_boxplot_paired(data, 'model', 'L1', os.path.join(paired_output_dir, "l1-" + file.name[:-4] + ".pdf"))

                print("\t- euclidian (l2) distance")
                plot_boxplot_paired(data, 'model', 'L2', os.path.join(paired_output_dir, "l2-" + file.name[:-4] + ".pdf"))

                plt.close('all')
        
    aggregated_checkpoint_files = [ f for f in os.scandir(args.input_dir) if not f.is_dir() ]           
    for file in aggregated_checkpoint_files:
        print("Saving plots for aggregated file", file.name)
        
        aggregated_checkpoint_data = pd.read_csv(file.path)
        data_type = get_data_type(file.name)
        n_checkpoints = aggregated_checkpoint_data['checkpoint'].unique().size

        if file.name.startswith('gold-'):
            aggregated_checkpoint_data.drop(aggregated_checkpoint_data[aggregated_checkpoint_data["model"] == "laser-gold"].index, inplace=True)
            print("\t- cosine distance")
            if n_checkpoints <= MAX_BOXPLOT_GROUPS:
                gold_checkpoint_data_file = os.path.join(args.output_dir, file.name[:-4].replace('gold', 'gold-cos') + ".pdf")
                plot_boxplot_gold(aggregated_checkpoint_data, 'model', 'cos', gold_checkpoint_data_file, data_type, hue='checkpoint')
            
            gold_average_data_file = os.path.join(args.output_dir, file.name[:-4].replace('gold', 'gold-cos-avg') + ".pdf")
            plot_lineplot_gold(aggregated_checkpoint_data, 'checkpoint', 'cos', gold_average_data_file, data_type, hue='model')
            print("\t- manhattan (l1) distance")
            if n_checkpoints <= MAX_BOXPLOT_GROUPS:
                gold_checkpoint_data_file = os.path.join(args.output_dir, file.name[:-4].replace('gold', 'gold-l1') + ".pdf")
                plot_boxplot_gold(aggregated_checkpoint_data, 'model', 'L1', gold_checkpoint_data_file, data_type, hue='checkpoint')
            
            gold_average_data_file = os.path.join(args.output_dir, file.name[:-4].replace('gold', 'gold-l1-avg') + ".pdf")
            plot_lineplot_gold(aggregated_checkpoint_data, 'checkpoint', 'L1', gold_average_data_file, data_type, hue='model')
            print("\t- euclidian (l2) distance")
            if n_checkpoints <= MAX_BOXPLOT_GROUPS:
                gold_checkpoint_data_file = os.path.join(args.output_dir, file.name[:-4].replace('gold', 'gold-l2') + ".pdf")
                plot_boxplot_gold(aggregated_checkpoint_data, 'model', 'L2', gold_checkpoint_data_file, data_type, hue='checkpoint')
            
            gold_average_data_file = os.path.join(args.output_dir, file.name[:-4].replace('gold', 'gold-l2-avg') + ".pdf")
            plot_lineplot_gold(aggregated_checkpoint_data, 'checkpoint', 'L2', gold_average_data_file, data_type, hue='model')

        if file.name.startswith('paired-'):
            print("\t- cosine distance")
            if n_checkpoints <= MAX_BOXPLOT_GROUPS:
                paired_checkpoint_data_file = os.path.join(args.output_dir, file.name[:-4].replace('paired', 'paired-cos') + ".pdf")
                plot_boxplot_paired(aggregated_checkpoint_data, 'model', 'cos', paired_checkpoint_data_file, hue='checkpoint')
            
            paired_average_data_file = os.path.join(args.output_dir, file.name[:-4].replace('paired', 'paired-cos-avg') + ".pdf")
            plot_lineplot_paired(aggregated_checkpoint_data, 'checkpoint', 'cos', paired_average_data_file, hue='model')
            print("\t- manhattan (l1) distance")
            if n_checkpoints <= MAX_BOXPLOT_GROUPS:
                paired_checkpoint_data_file = os.path.join(args.output_dir, file.name[:-4].replace('paired', 'paired-l1') + ".pdf")
                plot_boxplot_paired(aggregated_checkpoint_data, 'model', 'L1', paired_checkpoint_data_file, hue='checkpoint')
            
            paired_average_data_file = os.path.join(args.output_dir, file.name[:-4].replace('paired', 'paired-l1-avg') + ".pdf")
            plot_lineplot_paired(aggregated_checkpoint_data, 'checkpoint', 'L1', paired_average_data_file, hue='model')
            print("\t- euclidian (l2) distance")
            if n_checkpoints <= MAX_BOXPLOT_GROUPS:
                paired_checkpoint_data_file = os.path.join(args.output_dir, file.name[:-4].replace('paired', 'paired-l2') + ".pdf")
                plot_boxplot_paired(aggregated_checkpoint_data, 'model', 'L2', paired_checkpoint_data_file, hue='checkpoint')
            
            paired_average_data_file = os.path.join(args.output_dir, file.name[:-4].replace('paired', 'paired-l2-avg') + ".pdf")
            plot_lineplot_paired(aggregated_checkpoint_data, 'checkpoint', 'L2', paired_average_data_file, hue='model')
    print("Done...")
