import os, argparse
from sonar.inference_pipelines.text import TextToTextModelPipeline
from fairseq2.models.nllb import (
    create_nllb_model, 
    load_nllb_config, 
    load_nllb_tokenizer,
    NllbTokenizer
)
from fairseq2.generation import BeamSearchSeq2SeqGenerator, TextTranslator
from fairseq2.models.transformer import TransformerModel
from typing import Union
from fairseq2.typing import Device
import torch

class NLLBTranslationPipeline(TextToTextModelPipeline):
    model: TransformerModel
    tokenizer: NllbTokenizer

    def __init__(
        self,
        model: Union[str, TransformerModel],
        tokenizer: Union[str, NllbTokenizer],
        device: Device = torch.device("cuda")
    ) -> None:
        """
        Args:
            model (Union[str, TransformerModel]): either card name or model object
            decoder (Union[str, NllbTokenizer]): either card name or model object
            device (device, optional): . Defaults to GPU.
        """
        super().__init__()
        if isinstance(model, str):
            config = load_nllb_config(model)
            model = create_nllb_model(config, dtype=torch.float32)

        if isinstance(tokenizer, str):
            tokenizer = load_nllb_tokenizer(tokenizer, progress=False)

        self.model = model.to(device).eval()   
        self.tokenizer = tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str)
    parser.add_argument("-s", "--src-lang", type=str)
    parser.add_argument("-t", "--tgt-lang", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("-m", "--model-name", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    output_file = os.path.join(args.output_dir, f"{file_name}.out")

    print("Loading translation pipeline...")
    translator = NLLBTranslationPipeline(model=args.model_name,
                                        tokenizer="nllb-200",
                                        device=torch.device("cuda")
    )
    
    with open(args.input_file) as f:
        data = f.readlines()
    sentences = [line.strip() for line in data]

    print("Generating outputs...")
    outputs = translator.predict(sentences, 
        source_lang=args.src_lang,
        target_lang=args.tgt_lang,
        progress_bar=True,
        batch_size=32
    )

    with open(output_file, "w") as f:
        for output in outputs:
            f.write(output + "\n")



