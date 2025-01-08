import os, argparse
from safetensors import safe_open
from rolaser_sentence_encoder import RoLaserSentenceEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-model", help="name or path of underlying language model", type=str, default="cardiffnlp/twitter-xlm-roberta-base")
    parser.add_argument("--checkpoint-path", help="path to model checkpoint", type=str, default="/home/lnishimw/scratch/experiments/robust-embeddings/mining/experiment_050o/models/checkpoint-620000")
    parser.add_argument("--output-path", help="path to save model", type=str)
    parser.add_argument("--push-to-hub", help="push model to hub", type=bool, action="store_true")
    args = parser.parse_args()

    # initialise model from underlying language model
    model = RoLaserSentenceEncoder(args.language_model)

    # load model.safetensors from checkpoint
    tensors = {}
    with safe_open(f"{args.checkpoint_path}/model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    # load model state_dict from checkpoint
    model.load_state_dict(tensors)

    # save model
    if not args.output_path:
        args.output_path = f"{os.path.dirname(args.checkpoint_path)}/RoLASER-v2"
    model.save(args.output_path)

    # push model to hub
    if args.push_to_hub:
        model.push_to_hub("lydianish/RoLASER-v2")
    