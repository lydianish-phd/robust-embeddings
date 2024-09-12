SCORE_FILE_SUFFIX = ".out.json"

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
    "rosonar": "RoSONAR"
}

METRIC_NAMES = {
    "bleu": "BLEU",
    "comet": "COMET",
}

BLEU_ROUND_DECIMALS = 2
COMET_ROUND_DECIMALS = 2

MULTILINGUAL_COLUMNS = [
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