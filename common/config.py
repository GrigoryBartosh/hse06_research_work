import os

__all__ = ["PATH"]

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "..")

PATH = {
    "DATASETS": {
        "COCO": {
            "TRAIN": {
                "IMAGES_DIR": os.path.join(ROOT_DIR, "data", "datasets", "coco", "images", "train2014"),
                "CAPTIONS": os.path.join(ROOT_DIR, "data", "datasets", "coco", "annotations", "captions_train2014.json")
            },
            "VAL": {
                "IMAGES_DIR": os.path.join(ROOT_DIR, "data", "datasets", "coco", "images", "val2014"),
                "CAPTIONS": os.path.join(ROOT_DIR, "data", "datasets", "coco", "annotations", "captions_val2014.json")
            },
            "VOCAB": os.path.join(ROOT_DIR, "data", "datasets", "coco", "vocab.pickle")
        }
    },
    "TF_LOGS": os.path.join(ROOT_DIR, "data", "logs_tf"),
    "MODELS": {
        "UNIMODEL_DIR": os.path.join(ROOT_DIR, "data", "models", "unimodel"),
        "WELDON_CLASSIF_PRETRAINED": os.path.join(ROOT_DIR, "data", "models", "pretrained_classif_152_2400.pth.tar"),
        "WORD2VEC_PRETRAINED": os.path.join(ROOT_DIR, "data", "models", "GoogleNews-vectors-negative300.bin"),
        "TRANSFORMER_PRETRAINED": {
            "PATH": os.path.join(ROOT_DIR, "data", "models", "transformer", "model"),
            "PATH_NAMES": os.path.join(ROOT_DIR, "data", "models", "transformer"),
            "ENCODER_PATH": os.path.join(ROOT_DIR, "data", "models", "transformer", "model", "encoder_bpe_40000.json"),
            "BPE_PATH": os.path.join(ROOT_DIR, "data", "models", "transformer", "model", "vocab_40000.bpe")
        }
    }
}
