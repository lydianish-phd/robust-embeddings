SCORE_FILE_SUFFIX = ".out.json"
STATS_FILE_PREFIX = "stats."
STATS_FILE_SUFFIX = ".json"
GPT_NORM_FILE_PREFIX = "gpt."

ROCSMT_CORPUS_NAME = "rocsmt"
FLORES_CORPUS_NAME = "flores200"

ROCSMT_NORM_FILE_NAME = "norm.en.test"
ROCSMT_RAW_FILE_NAME = "raw.en.test"
FLORES_FILE_NAME = "eng_Latn.devtest"

COLUMN_NAME_SEPARATOR = "__"

AVERAGE_ROCSMT_NORM_COLUMN= COLUMN_NAME_SEPARATOR.join(["avg", "rocsmt", ROCSMT_NORM_FILE_NAME])
AVERAGE_ROCSMT_RAW_COLUMN= COLUMN_NAME_SEPARATOR.join(["avg", "rocsmt", ROCSMT_RAW_FILE_NAME])

MODEL_NAMES = {
    "nllb1b": "NLLB-1.3B",
    "nllb600m": "NLLB-600M",
    "sonar": "SONAR",
    "rosonar": "RoSONAR",
    "rosonar_std": "RoSONAR w/o UGC",
    "nllb600m_ft_nat_en_fr": "NLLB-600M + FT en-fr",
    "nllb600m_ft_nat_fr_en": "NLLB-600M + FT fr-en",
    "nllb600m_ft_syn_en_fr": "NLLB-600M + FT en-fr (syn)",
    "nllb600m_ft_nat_en_fr_bal": "NLLB-600M + FT en-fr_bal",
    "nllb600m_ft_nat_fr_en_bal": "NLLB-600M + FT fr-en_bal",
    "nllb600m_ft_syn_en_fr_bal": "NLLB-600M + FT en-fr_bal (syn)",
    "nllb600m_ft_syn_plus_en_fr": "NLLB-600M + FT en-fr (syn+)",
    "nllb600m_ft_syn_plus_en_fr_bal": "NLLB-600M + FT en-fr_bal (syn+)",
    "gpt4omini": "GPT-4o-mini",
}

METRIC_NAMES = {
    "bleu": "BLEU",
    "comet": "COMET",
    "chrf2": "chrF++",
}

STATS =  [
    "lines",
    "fertility",
    "types",
    "tokens",
    "ttr",
    "urls",
    "usernames",
    "hashtags",
    "urls_per_line",
    "usernames_per_line",
    "hashtags_per_line",
    "average_sentence_length",
    "stddev_sentence_length",
    "mattr",
    "hdd",
]


LANG_NAMES = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ces_Latn": "Czech",
    "rus_Cyrl": "Russian",
    "ukr_Cyrl": "Ukrainian",
    "jap_Japn": "Japanese"
}

ROUND_DECIMALS = 2

MULTILINGUAL_COLUMNS = [
    "model",
    "flores200__eng_Latn-ces_Latn__eng_Latn.devtest",
    "rocsmt__eng_Latn-ces_Latn__norm.en.test",
    "rocsmt__eng_Latn-ces_Latn__raw.en.test",
    "delta__flores200__eng_Latn-ces_Latn",
    "delta__rocsmt__eng_Latn-ces_Latn",
    "flores200__eng_Latn-deu_Latn__eng_Latn.devtest",
    "rocsmt__eng_Latn-deu_Latn__norm.en.test",
    "rocsmt__eng_Latn-deu_Latn__raw.en.test",
    "delta__flores200__eng_Latn-deu_Latn",
    "delta__rocsmt__eng_Latn-deu_Latn",
    "flores200__eng_Latn-fra_Latn__eng_Latn.devtest",
    "rocsmt__eng_Latn-fra_Latn__norm.en.test",
    "rocsmt__eng_Latn-fra_Latn__raw.en.test",
    "delta__flores200__eng_Latn-fra_Latn",
    "delta__rocsmt__eng_Latn-fra_Latn",
    "flores200__eng_Latn-rus_Cyrl__eng_Latn.devtest",
    "rocsmt__eng_Latn-rus_Cyrl__norm.en.test",
    "rocsmt__eng_Latn-rus_Cyrl__raw.en.test",
    "delta__flores200__eng_Latn-rus_Cyrl",
    "delta__rocsmt__eng_Latn-rus_Cyrl",
    "flores200__eng_Latn-ukr_Cyrl__eng_Latn.devtest",
    "rocsmt__eng_Latn-ukr_Cyrl__norm.en.test",
    "rocsmt__eng_Latn-ukr_Cyrl__raw.en.test",
    "delta__flores200__eng_Latn-ukr_Cyrl",
    "delta__rocsmt__eng_Latn-ukr_Cyrl",
    "avg__flores200__eng_Latn.devtest",
    "avg__rocsmt__norm.en.test",
    "avg__rocsmt__raw.en.test",
    "delta__avg__flores200",
    "delta__avg__rocsmt"
]

