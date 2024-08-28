SCORE_FILE_SUFFIX = ".out.json"

ROCSMT_NORM_FILE_NAME = "norm.en.test"
ROCSMT_RAW_FILE_NAME = "raw.en.test"

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