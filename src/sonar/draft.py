from sonar.inference_pipelines.text import TextToTextModelPipeline

input_file = "/gpfsscratch/rech/rnh/udc54vm/datasets/rocsmt/test/raw.en.test"
output_file = "/gpfsscratch/rech/rnh/udc54vm/experiments/robust-embeddings/sonar/experiment_047/outputs/sonar/rocsmt/en-fr/raw.en.test.out"
src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

print("Loading translation pipeline...")
t2t_model = TextToTextModelPipeline(encoder="text_sonar_basic_encoder",
                                        decoder="text_sonar_basic_decoder",
                                        tokenizer="text_sonar_basic_encoder")

with open(input_file) as f:
    data = f.read()
sentences = data.strip().split("\n")

print("Generating outputs...")
outputs = t2t_model.predict(sentences, source_lang=src_lang, target_lang=tgt_lang, progress_bar=True)

with open(output_file, "w") as f:
    for output in outputs:
        f.write(output + "\n")



