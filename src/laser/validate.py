import torch, os, configargparse
import numpy as np
import pandas as pd
from embed import embed_sentences
from score import read_embeddings
from sklearn.metrics import mean_squared_error
from pathlib import Path

if __name__ == "__main__":
	parser = configargparse.ArgParser()
	parser.add("--teacher-model", dest="teacher_model", help="name of teacher model", type=str, default="laser")
	parser.add("--teacher-model-path", dest="teacher_model_path", help="path to teacher model", type=str, default="/home/lnishimw/scratch/LASER/models/laser2.pt")
	parser.add("--teacher-vocab", dest="teacher_vocab", help="path of teacher model vocabulary", type=str, default="/home/lnishimw/scratch/LASER/models/laser2.cvocab")
	parser.add("--teacher-tok", dest="teacher_tok", help="teacher tokenizer", type=str, default="spm")
	parser.add("--student-model", dest="student_model", help="name of student model", type=str)
	parser.add("--student-model-dir", dest="student_model_dir", help="path to student model checkpoints directory", type=str)
	parser.add("--student-vocab", dest="student_vocab", help="path of student model vocabulary", type=str)
	parser.add("--student-tok", dest="student_tok", help="student tokenizer", type=str)
	parser.add("--ugc-file", dest="ugc_file", help="name of UGC data file", type=str)
	parser.add("--std-file", dest="std_file", help="name of standard data file", type=str)
	parser.add("--output-dir", dest="output_dir", help="path to directory to save embeddings and results", type=str)
	args = parser.parse_args()
	
	src_embed_dir = os.path.join(args.output_dir, "embeddings", args.student_model)
	tgt_embed_dir = os.path.join(args.output_dir, "embeddings", args.teacher_model)
	Path(src_embed_dir).mkdir(parents=True, exist_ok=True)
	Path(tgt_embed_dir).mkdir(parents=True, exist_ok=True)
	#os.makedirs(src_embed_dir, exist_ok=True)
	#os.makedirs(tgt_embed_dir, exist_ok=True)

	ugc_filename = os.path.basename(args.ugc_file)
	std_filename = os.path.basename(args.std_file)
	tgt_embed_file = os.path.join(tgt_embed_dir, std_filename + ".bin")

	src_score_dir = os.path.join(args.output_dir, "scores", args.student_model)
	Path(src_score_dir).mkdir(parents=True, exist_ok=True)
	#os.makedirs(src_score_dir, exist_ok=True)
	output_score_file = os.path.join(src_score_dir, "train_valid.csv")

	if not os.path.exists(tgt_embed_file):
		embed_sentences(
		ifname=args.std_file,
		encoder_path=args.teacher_model_path,
		custom_tokenizer=args.teacher_tok,
		custom_vocab_file=args.teacher_vocab,
		verbose=True,
		output=tgt_embed_file
		)

	results = []

	for f in os.scandir(args.student_model_dir):
		print("Validating", f.name)

		src_model_file = f.path
		model = torch.load(src_model_file)
		epoch = int(model["extra_state"]["train_iterator"]["epoch"])
		steps = int(model["optimizer_history"][-1]["num_updates"])
		del model

		src_ugc_embed_file = os.path.join(src_embed_dir, '_'.join([str(epoch), str(steps), ugc_filename + ".bin"]))
		src_std_embed_file = os.path.join(src_embed_dir, '_'.join([str(epoch), str(steps), std_filename + ".bin"]))

		if not os.path.exists(src_std_embed_file):
			embed_sentences(
			ifname=args.std_file,
			encoder_path=src_model_file,
			custom_tokenizer=args.student_tok,
			custom_vocab_file=args.student_vocab,
			verbose=True,
			output=src_std_embed_file
			)

		if not os.path.exists(src_ugc_embed_file):
			embed_sentences(
			ifname=args.ugc_file,
			encoder_path=src_model_file,
			custom_tokenizer=args.student_tok,
			custom_vocab_file=args.student_vocab,
			verbose=True,
			output=src_ugc_embed_file
			)

		X_gold = read_embeddings(tgt_embed_file, normalized=False)
		X_std = read_embeddings(src_std_embed_file, normalized=False)
		X_ugc = read_embeddings(src_ugc_embed_file, normalized=False)

		loss_std_gold = mean_squared_error(X_gold, X_std)
		loss_ugc_gold = mean_squared_error(X_gold, X_ugc)

		valid_distil_loss = loss_std_gold + loss_ugc_gold

		results.append({
		"epoch": epoch, 
		"steps": steps, 
		"loss_std_gold": loss_std_gold,
		"loss_ugc_gold": loss_ugc_gold,
		"valid_distil_loss": valid_distil_loss
		})

	results_df = pd.DataFrame(results)
	print("Writing results of all checkpoints...")
	results_df.to_csv(output_score_file)
