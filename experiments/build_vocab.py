import nltk
from collections import Counter

from pycocotools.coco import COCO

from common.config import PATH
from common.text_utils import tokenize, Vocabulary

THRESHOLD = 2


def get_words(path, threshold):
    coco = COCO(path)
    counter = Counter()
    ids = coco.anns.keys()
    for id in ids:
        caption = str(coco.anns[id]['caption'])
        tokens = tokenize(caption)
        counter.update(tokens)

    return [word for word, cnt in counter.items() if cnt >= threshold]

if __name__ == '__main__':
    words = get_words(PATH["DATASETS"]["COCO"]["TRAIN"]["CAPTIONS"], THRESHOLD)

    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)

    vocab.save(PATH["DATASETS"]["COCO"]["VOCAB"])
