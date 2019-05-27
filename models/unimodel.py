import torch
import torch.nn as nn
import torchvision

from models.image import ResNetWeldonEmbedding
from models.word import Word2Vec
from models.text import LSTMEmbedding, SRUEmbedding, TransformerEmbedding

__all__ = ["UniModelLSTM", "UniModelSRU", "UniModelTransformer"]


class AUniModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.emb_size = args["emb_size"]

    def get_text_embedding(self, caps, lengths=None):
        return self.text_model(caps, lengths=lengths)

    def get_image_embedding(self, imgs, x_caps=None):
        return self.image_model(imgs, caps=x_caps)

    def norm_vector(self, v):
        return v / torch.norm(v, p=2, dim=-1, keepdim=True).expand_as(v)

    def forward(self, imgs, caps, lengths):
        if caps is not None:
            x_caps = self.get_text_embedding(caps, lengths=lengths)
            x_caps = self.norm_vector(x_caps)
        else:
            x_caps = None

        if imgs is not None:
            x_imgs, x_imgs_att, x_g = self.get_image_embedding(imgs, x_caps=x_caps)
            x_imgs = self.norm_vector(x_imgs)
            x_imgs_att = self.norm_vector(x_imgs_att)
        else:
            x_imgs = None
            x_imgs_att = None
            x_g = None

        return x_imgs, x_imgs_att, x_caps, x_g


class UniModelLSTM(AUniModel):
    def __init__(self, vocab, args):
        super().__init__(args)

        self.image_model = ResNetWeldonEmbedding(
            emb_size=self.emb_size,
            attention_k=args["image"]["attention_k"],
            pretrained=args["image"]["pretrained"],
            fc_dropout=args["image"]["fc_dropout"],
            resnet_freeze=args["image"]["resnet_freeze"])

        self.w2v = Word2Vec(
            vocab, 
            freeze=args["word"]["freeze"])

        self.lstm = LSTMEmbedding(
            self.w2v.emb_size, 
            self.emb_size, 
            num_layers=args["text"]["num_layers"], 
            dropout=args["text"]["dropout"], 
            bidirectional=args["text"]["bidirectional"])

    def get_text_embedding(self, caps, lengths=None):
        x_caps = self.w2v(caps)
        x_caps = self.lstm(x_caps, lengths=lengths)
        return x_caps


class UniModelSRU(AUniModel):
    def __init__(self, vocab, args):
        super().__init__(args)

        self.image_model = ResNetWeldonEmbedding(
            emb_size=self.emb_size,
            attention_k=args["image"]["attention_k"],
            pretrained=args["image"]["pretrained"],
            fc_dropout=args["image"]["fc_dropout"],
            resnet_freeze=args["image"]["resnet_freeze"])

        self.w2v = Word2Vec(
            vocab, 
            freeze=args["word"]["freeze"])

        self.sru = LSTMEmbedding(
            self.w2v.emb_size, 
            self.emb_size, 
            num_layers=args["text"]["num_layers"], 
            dropout=args["text"]["dropout"])

    def get_text_embedding(self, caps, lengths=None):
        x_caps = self.w2v(caps)
        x_caps = self.sru(x_caps, lengths=lengths)
        return x_caps


class UniModelTransformer(AUniModel):
    def __init__(self, args):
        super().__init__(args)

        self.image_model = ResNetWeldonEmbedding(
            emb_size=self.emb_size,
            attention_k=args["image"]["attention_k"],
            pretrained=args["image"]["pretrained"],
            fc_dropout=args["image"]["fc_dropout"],
            resnet_freeze=args["image"]["resnet_freeze"])

        self.text_model = TransformerEmbedding(
            emb_size=self.emb_size,
            pretrained=args["text"]["pretrained"],
            fc_dropout=args["text"]["fc_dropout"],
            transformer_freeze=args["text"]["transformer_freeze"])
