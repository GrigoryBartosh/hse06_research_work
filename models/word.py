import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

from common.config import PATH

__all__ = ["Word2Vec"]


class WordEmbedding(nn.Module):
    def __init__(self, weights=None, num_embeddings=None, embedding_dim=300, freeze=True):
        super().__init__()

        if weights is not None:
            weights = torch.FloatTensor(weights)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.requires_grad = not freeze
        
    def forward(self, input):
        return self.embedding(input)

    def set_requires_grad(self, requires_grad):
        self.embedding.weight.requires_grad = requires_grad


class Word2Vec(WordEmbedding):
    def __init__(self, vocab, freeze=True):
        w2v = KeyedVectors.load_word2vec_format(PATH["MODELS"]["WORD2VEC_PRETRAINED"], binary=True)
        self.emb_size = w2v.vector_size

        weights = []
        for i in range(len(vocab)):
            w = vocab[i]
            v = w2v[w] if w in w2v else np.zeros(self.emb_size)
            weights.append(v)
        weights = np.array(weights)

        super().__init__(weights, freeze)
