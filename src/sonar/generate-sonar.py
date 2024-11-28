import os, argparse
import torch
from sonar.inference_pipelines.text import TextToTextModelPipeline
from rosonar_distillation import load_student_encoder_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str)
    parser.add_argument("-s", "--src-lang", type=str)
    parser.add_argument("-t", "--tgt-lang", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("-m", "--model-dir", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    output_file = os.path.join(args.output_dir, f"{file_name}.out")

    print("Loading translation pipeline...")
    if args.model_dir:
        encoder = load_student_encoder_from_checkpoint(args.model_dir)
    else:
        encoder = "text_sonar_basic_encoder"
    translator = TextToTextModelPipeline(encoder=encoder,
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda")
    )

    print("Reading input sentences...")
    with open(args.input_file) as f:
        data = f.readlines()
    sentences = [line.strip() for line in data]

    print("Generating outputs...")
    outputs = translator.predict(sentences, 
        source_lang=args.src_lang,
        target_lang=args.tgt_lang,
        progress_bar=True,
        batch_size=32,
        max_seq_len=512
    )

    print("Writing output sentences...")
    with open(output_file, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output + "\n")



