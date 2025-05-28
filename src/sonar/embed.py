import os, argparse
import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from rosonar_distillation import load_student_encoder_from_checkpoint
from utils import LANG_CODES, LANG_NAMES
            
def get_langs(langs):
    if langs:
        return sorted(langs.split(",")) if isinstance(langs, str) else sorted(langs)
    return []

def get_lang_code(lang):
    if lang in LANG_NAMES:
        return lang
    splits = lang.split(".")
    if splits[-1][:2] in LANG_CODES: # extension starts with the language code
        return LANG_CODES[splits[-1][:2]]
    if splits[0][:2] in LANG_CODES: # basename starts with the language code
        return LANG_CODES[splits[0][:2]]
    raise ValueError(f"Unknown language code: {lang}. Please use a valid language code.")

def embed_sentences(embedder, input_file, output_file, lang, fp16=False, batch_size=32):
    print("Reading input sentences...")
    with open(input_file) as f:
        data = f.readlines()
    sentences = [line.strip() for line in data]

    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = embedder.predict(sentences, 
        source_lang=get_lang_code(lang),
        progress_bar=True,
        batch_size=batch_size
    )

    print("Writing output embeddings...")
    embeddings = np.array(embeddings.cpu(), dtype=np.float16 if fp16 else np.float32)
    with open(output_file, "wb") as fout:
        embeddings.tofile(fout)
    
    assert (
        os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    ), f"Error encoding {input_file}"

    print(f"Embeddings saved to {output_file}")

def embed(embedder, data_dir, embed_dir, corpus, split, langs, tgt_aug_langs=[], fp16=False, batch_size=32, overwrite=False):
        for lang in langs:
            augjson = None
            fname = f"{lang}.{split}"
            input_dir = os.path.join(data_dir, corpus, split)
            infile = os.path.join(input_dir, fname)
            assert os.path.isfile(infile), f"{infile} does not exist"
            output_dir = os.path.join(embed_dir, corpus, split)
            os.makedirs(output_dir, exist_ok=True)
            outfile = os.path.join(output_dir, fname)
            if lang in tgt_aug_langs:
                fname = f"{lang}_augmented.{split}"
                fjname = f"{lang}_errtype.{split}.json"
                augment_dir = os.path.join(data_dir, corpus, f"{split}_augmented")
                augjson = os.path.join(augment_dir, fjname)
                auginfile = os.path.join(augment_dir, fname)
                assert os.path.isfile(augjson), f"{augjson} does not exist"
                assert os.path.isfile(auginfile), f"{auginfile} does not exist"
                combined_infile = os.path.join(input_dir, f"combined_{lang}")
                with open(combined_infile, "w") as newfile:
                    for f in [infile, auginfile]:
                        with open(f) as fin:
                            newfile.write(fin.read())
                infile = combined_infile
            if overwrite or not os.path.isfile(outfile) or os.path.getsize(outfile) == 0:
                embed_sentences(embedder, infile, outfile, lang, fp16, batch_size)
            else:
                print(f"Skipping {outfile}, already exists and overwrite is False")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", type=str)
    parser.add_argument("--data-dir", type=str, required=True, help="Base directory for data")
    parser.add_argument("--embed-dir", type=str, required=True, help="Directory to save embeddings")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus name")
    parser.add_argument("--split", type=str, default="test", help="Data split (default: test)")
    parser.add_argument("--src-langs", type=str, required=True, help="Source languages to process, comma-separated")
    parser.add_argument("--tgt-langs", type=str, required=True, help="Target languages to process, comma-separated")
    parser.add_argument("--tgt-aug-langs", type=str, nargs="*", default=[], help="Target augmented languages, comma-separated")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision for embeddings")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings if they exist")
    args = parser.parse_args()

    print("Loading embedding pipeline...")
    if args.model_dir:
        encoder = load_student_encoder_from_checkpoint(args.model_dir, init="rosonar")
    else:
        encoder = "text_sonar_basic_encoder"
    embedder = TextToEmbeddingModelPipeline(encoder=encoder,
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda")
    )
    src_langs = get_langs(args.src_langs)
    tgt_langs = get_langs(args.tgt_langs)
    tgt_aug_langs = get_langs(args.tgt_aug_langs) 

    print(f"Source languages: {src_langs}")
    embed(embedder,
        args.data_dir, 
        args.embed_dir, 
        args.corpus, 
        args.split, 
        src_langs, 
        fp16=args.fp16,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )
    print(f"Target languages: {tgt_langs}")
    embed(embedder,
        args.data_dir, 
        args.embed_dir, 
        args.corpus, 
        args.split, 
        tgt_langs, 
        tgt_aug_langs=tgt_aug_langs,
        fp16=args.fp16,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )