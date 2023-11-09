import utils
import os, configargparse

if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add("-i", "--input-dir", dest="input_dir", help="path to directory to read error matrices", type=str, default="/home/lnishimw/scratch/experiments/robust-embeddings/laser/experiment_023/results")
    parser.add("-o", "--output-dir", dest="output_dir", help="path to directory to save plots", type=str, default="/home/lnishimw/scratch/experiments/robust-embeddings/laser/experiment_023/plots")
    parser.add("-p", "--pdf", help="save plots as pdf rather than png", default=False, action="store_true")
    parser.add("-n", "--nway", help="plot n-way xsim heatmaps", default=False, action="store_true")
    parser.add("-r", "--remove", help="space-separated list of transformations to remove. Available transformations: " + str(utils.NOISY_LANGS), nargs="+", type=str, default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.nway:
        print("N-way heatmaps...")
        for lang_set in [ utils.MULTI_LANG_SET, utils.NOISY_LANG_SET ]:
            utils.xsim_nway_heatmap(lang_set, args.input_dir, args.output_dir, args.pdf)

    langs = utils.NOISY_LANGS
    for perturb in args.remove:
        langs.remove(perturb)

    print("2-way plots...")
    utils.xsim_2way_plot(langs, utils.MULTI_LANG_SET, args.input_dir, args.output_dir, args.pdf)
    utils.xsim_2way_distribution_plots(langs, utils.MULTI_LANG_SET, args.input_dir, args.output_dir, args.pdf)

    print("Plots saved in", args.output_dir)