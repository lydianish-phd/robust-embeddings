# Robust Sentence Embeddings for User-Generated Content (UGC)

This repository contains the code and experiments for my PhD work on **robust sentence embeddings for user-generated content (UGC)**, focusing on aligning standard and non-standard language in a shared semantic space.  
It covers **Experiment V (RoLASER)** and provides the foundations for **Experiment VI (RoSONAR)** from my dissertation.

---

## 📁 Repository Structure

The repository is organised by model and experiment, with a clear separation between source code and SLURM job scripts:

```bash
├── src/
│   ├── rolaser/   # Preprocessing, training, and evaluation code for RoLASER
│   └── rosonar/   # Preprocessing, training, and evaluation code for RoSONAR
├── slurm/
│   ├── rolaser/   # SLURM scripts for RoLASER training and experiments
│   └── rosonar/   # SLURM scripts for RoSONAR training and experiments
├── .gitignore
└── README.md
```
---

## 🔍 Motivation

Most sentence encoders are trained on clean, standard text and degrade sharply when applied to UGC such as social media content, which exhibits:
- spelling and grammatical errors,
- slang, acronyms, and abbreviations,
- expressive typography (emojis, repetitions, leetspeak),
- tokenisation-breaking character-level perturbations.

This work tackles robustness **at the sentence level**, framing UGC robustness as a **bitext alignment problem in embedding space**:
> *How close are the embeddings of a standard sentence and its non-standard counterpart?*

---

## 🧠 Core Idea

We propose training **UGC-robust sentence encoders** using:
- **Knowledge distillation** (teacher–student training),
- **Synthetic UGC generation** from standard text,
- **Embedding alignment losses** that minimise the distance between standard and non-standard variants.

Rather than normalising text explicitly, the model learns to **abstract away surface-level variation**.

---

## 🧪 Experiment V — RoLASER

### RoLASER: Robust LASER-style Sentence Embeddings

**RoLASER** is a Transformer-based student encoder trained to map non-standard English sentences close to their standard equivalents in the **LASER embedding space**.

🔗 **RoLASER Demo GitHub repository:** https://github.com/lydianish-phd/RoLASER

> Note: The separate RoLASER GitHub repo linked above is the official demo released with the paper and is intended for demonstration purposes, while this repository contains the full research code used in the thesis.

**Key features:**
- Teacher: frozen **LASER** encoder
- Students:
  - **RoLASER** (token-level, RoBERTa-style)
  - **cRoLASER** (character-aware variant)
- Training objective: **MSE loss between teacher and student embeddings**
- Data: standard English paired with **synthetically generated UGC**

**Main findings:**
- Substantially improves robustness to both **synthetic and natural UGC**
- Handles tokenisation-breaking perturbations better than LASER
- Maintains (or slightly improves) performance on standard text
- Token-level models outperform character-aware models in this setting

---

## 🔬 Evaluation

Robustness is evaluated as a **sentence alignment task**, using:
- Average cosine distance
- **xSIM** and **xSIM++** (bitext mining proxy metrics)

Datasets include:
- **MultiLexNorm** (natural UGC)
- **ROCS-MT** (UGC + standardised variants)
- **FLORES** with controlled synthetic UGC perturbations

Downstream evaluations include:
- Sentence (pair) classification
- Semantic Textual Similarity (STS)

---

## 🧩 Synthetic UGC Generation

UGC is generated from standard text using probabilistic combinations of:
- abbreviations & slang,
- misspellings & typos,
- leetspeak,
- whitespace and segmentation noise,
- contractions and expansions.

This enables controlled robustness analysis while highlighting the **gap between synthetic and natural UGC**.

---

## 🔁 Experiment VI — RoSONAR (context)

This repository also provides the conceptual and experimental groundwork for **RoSONAR**, which extends the approach to **machine translation** by:
- training a robust sentence encoder aligned to **SONAR**,
- pairing it with a frozen multilingual SONAR decoder,
- evaluating robustness transfer across languages.

---

## 🛠️ Implementation

- Frameworks: **Fairseq**, **PyTorch**
- Architectures: Transformer encoders (token-level & character-aware)
- Training:
  - multi-GPU distributed training
  - synthetic data augmentation
  - large-scale embedding distillation

---

## 📄 Publication

If you use the RoLASER model or ideas from this work, please cite the following paper:

```bibtex
@inproceedings{nishimwe-etal-2024-making-sentence,
    title = "Making Sentence Embeddings Robust to User-Generated Content",
    author = "Nishimwe, Lydia  and
      Sagot, Beno{\^\i}t  and
      Bawden, Rachel",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.958",
    pages = "10984--10998"
}
```
---

## 👤 Author

**Lydia Nishimwe**  
PhD in Machine Translation & NLP  
Focus: UGC robustness, sentence embeddings, multilingual NLP

🔗 Personal GitHub: https://github.com/lydianish  
🔗 PhD organisation: https://github.com/lydianish-phd

---

## ⚠️ Notes & Limitations

- Synthetic UGC does not fully capture real-world UGC distributions
- Robust embeddings do not automatically guarantee robust MT
- Domain adaptation remains a key challenge

This repository should be viewed as a **research artefact** supporting the dissertation rather than a polished end-user library.
