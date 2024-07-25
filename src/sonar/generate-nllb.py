import os, argparse
from sonar.inference_pipelines.text import TextToTextModelPipeline
from sonar_distillation import load_student_encoder_from_checkpoint
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    TextTranslator    
)
from fairseq2.models.nllb import create_nllb_model, load_nllb_config, load_nllb_tokenizer
from fairseq2.generation import BeamSearchSeq2SeqGenerator, TextTranslator

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
    
    tokenizer = load_nllb_tokenizer("nllb-200")
    config = load_nllb_config("nllb-200_dense_distill_600m")
    model = create_nllb_model(config)
    generator = BeamSearchSeq2SeqGenerator(model)

    translator = TextTranslator(generator, tokenizer, source_lang=src_lang, target_lang=tgt_lang)
    
    if args.model_dir:
        encoder = load_student_encoder_from_checkpoint(args.model_dir)
    else:
        encoder = "text_sonar_basic_encoder"
    t2t_model = TextToTextModelPipeline(encoder=encoder,
                                        decoder="text_sonar_basic_decoder",
                                        tokenizer="text_sonar_basic_encoder")

    with open(args.input_file) as f:
        data = f.read()
    sentences = data.strip().split("\n")

    print("Generating outputs...")
    outputs = t2t_model.predict(sentences, 
        source_lang=args.src_lang,
        target_lang=args.tgt_lang,
        progress_bar=True,
        batch_size=32,
        max_seq_len=512
    )

    with open(output_file, "w") as f:
        for output in outputs:
            f.write(output + "\n")



