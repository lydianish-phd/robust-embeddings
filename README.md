# Robust Sentence Embeddings for User-Generated Content (UGC)

This repository contains the code and experiments for my PhD work on **robust sentence embeddings for user-generated content (UGC)**, focusing on aligning standard and non-standard language in a shared semantic space.  
It covers **Experiment V (RoLASER)** and **Experiment VI (RoSONAR)** from my dissertation.

---

## 📑 Table of Contents

1. [📁 Repository Structure](#-repository-structure)
2. [🔍 Motivation](#-motivation)
3. [🧠 Core Idea](#-core-idea)
4. [🧪 Experiments](#-experiments)
   - [🧩 Synthetic UGC Generation](#-synthetic-ugc-generation)
   - [🧩 RoLASER](#-rolaser)
     - [⚙️ Experimental Setup](#experimental-setup)
     - [🔬 Evaluation & Findings](#-evaluation--findings)
   - [🧩 RoSONAR](#-rosonar)
     - [⚙️ Experimental Setup](#-experimental-setup-1)
     - [🔬 Evaluation & Findings](#-evaluation--findings-1)
5. [📄 Publication](#-publication)
6. [👤 Author](#-author)
7. [⚠️ Notes & Limitations](#-notes--limitations)

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

## 🧪 Experiments

### 🧩 Synthetic UGC Generation

Due to the scarcity of natural user-generated content (UGC) data, we artificially generate non-standard English sentences from standard text. This allows us to **train and evaluate models on a wide range of UGC phenomena** without relying solely on limited real-world datasets.

We apply a set of 12 probabilistic transformations to standard sentences, including:

- **Abbreviations, acronyms, and slang** 
- **Contractions and expansions**  
- **Misspellings and typos**
- **Visual and segmentation perturbations**

We also use a **mix_all** transformation, which randomly applies a subset of these perturbations in shuffled order, simulating realistic UGC variation.  

These synthetic datasets enable controlled experimentation, allowing us to **measure model robustness by UGC phenomenon type** and address the lack of large-scale annotated non-standard text.

### 🧩 RoLASER

**RoLASER** is a Transformer-based student encoder trained to map non-standard English sentences close to their standard equivalents in the **LASER embedding space**.

🔗 **RoLASER Demo GitHub repository:** https://github.com/lydianish-phd/RoLASER

> Note: The separate RoLASER GitHub repo linked above is the official demo released with the paper and is intended for demonstration purposes, while this repository contains the full research code used in the thesis.


#### Experimental Setup

**Variants:**

| Model    | Input type      | Architecture                 |
| -------- | --------------- | ---------------------------- |
| RoLASER  | Token-level     | RoBERTa-style Transformer    |
| cRoLASER | Character-level | CNN + Transformer (CharacterBERT) |

**Key points:**

- Teacher: frozen LASER encoder
- Training: MSE loss between teacher and student embeddings
- Data: 2M standard sentences from OSCAR, augmented with synthetic UGC phenomena
- Pooling: max-pooling (works better than CLS/mean for sentence-level alignment)
- Framework: Fairseq, multi-GPU training


#### 🔬 Evaluation & Findings

**Metrics:** cosine distance, xSIM / xSIM++  
**Datasets:** MultiLexNorm, ROCS-MT (natural UGC), FLORES artificial UGC  
**Downstream tasks:**

1. **Sentence classification** (e.g., TweetSentimentExtraction)
2. **Sentence pair classification** (e.g., paraphrase detection)
3. **Semantic Textual Similarity (STS)**

**Main findings:**

- RoLASER substantially improves robustness to **synthetic and natural UGC**
- Handles tokenisation-breaking perturbations better than LASER
- Maintains (or slightly improves) performance on standard text
- Token-level RoLASER outperforms character-aware c-RoLASER in most settings
- Character-level models are internally robust but struggle to map outputs to LASER space

### 🧩 RoSONAR

**RoSONAR** extends the RoLASER approach to **machine translation**, training a bilingual English–French sentence encoder aligned with SONAR and paired with a frozen multilingual SONAR decoder.

#### ⚙️ Experimental Setup

- **Teacher:** Multilingual SONAR encoder  
- **Student:** Smaller bilingual encoder trained on:
  - Parallel English↔French data (NLLB)
  - Monolingual English and French data (OSCAR)
  - Synthetic English UGC (19 transformation types)
- **Objective:** Minimise MSE between teacher and student embeddings for standard and non-standard sentences

**Architecture & Training:**

- 12 Transformer layers (half of SONAR), 16 attention heads, hidden size 1024, FFN 8192  
- Tokeniser: SentencePiece, vocab size 256k  
- Encoder parameters: 514M; combined with SONAR decoder: 1.643B  
- Optimisation: AdamW, LR 7e-3, BF16 mixed precision, 16 H100 GPUs, effective batch size 1M tokens

#### 🔬 Evaluation & Findings

- Evaluated on standard, synthetic, and natural UGC datasets (MultiLexNorm, ROCS-MT, FLORES)  
- Compared models: RoSONAR, RoSONAR-std (trained only on standard data), SONAR, NLLB  

**Main results:**

1. RoSONAR vs RoSONAR-std: nearly identical (<0.5 COMET points), showing limited impact of synthetic UGC at this scale  
2. Compared to baselines: both RoSONAR variants slightly underperform SONAR but remain close to NLLB; some gains on highly non-standard French (PFS-MB)  
3. Robustness to perturbations: large models degrade slowly under synthetic UGC; RoSONAR similar to SONAR with limited extra gains  
4. Cross-lingual transfer: robustness from English synthetic UGC does not reliably transfer to French  
5. Implication: encoder-level robustness alone is insufficient; natural UGC and domain adaptation are crucial

**Key takeaway:** Large-scale encoders already have inherent robustness to surface-level non-standardness. Synthetic UGC objectives show diminishing returns at scale; in-domain natural UGC is necessary for meaningful MT robustness.

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
