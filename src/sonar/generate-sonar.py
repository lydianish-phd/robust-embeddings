import os, argparse
from sonar.inference_pipelines.text import TextToTextModelPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str)
    parser.add_argument("-s", "--src-lang", type=str)
    parser.add_argument("-t", "--tgt-lang", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    output_files = os.path.join(args.output_dir, f"{file_name}.out")

    print("Loading translation pipeline...")
    t2t_model = TextToTextModelPipeline(encoder="text_sonar_basic_encoder",
                                        decoder="text_sonar_basic_decoder",
                                        tokenizer="text_sonar_basic_encoder")

    with open(args.input_file) as f:
        data = f.read()
    sentences = data.strip().split("\n")

    print("Generating outputs...")
    outputs = t2t_model.predict(sentences, source_lang=args.src_lang, target_lang=args.tgt_lang, progress_bar=True, batch_size=8)

    with open(output_file, "w") as f:
        for output in outputs:
            f.write(output + "\n")



