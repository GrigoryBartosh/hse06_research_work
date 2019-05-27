import torch
import torch.nn as nn

from common.config import PATH

from models.resnet import WeldonPooling, ResNetWeldon

__all__ = ["ResNetWeldonEmbedding"]


class ResNetWeldonEmbedding(nn.Module):
    def __init__(self, emb_size=2400, attention_k=180, pretrained=True, fc_dropout=0.5, resnet_freeze=True):
        super().__init__()

        self.attention_k = attention_k

        model_weldon2 = ResNetWeldon(
            emb_size=emb_size, 
            pretrained=pretrained, 
            weldon_pretrained_path=PATH["MODELS"]["WELDON_CLASSIF_PRETRAINED"])

        self.base_layer = nn.Sequential(*list(model_weldon2.children())[:-2])

        self.wldPool = WeldonPooling(15)

        self.fc = nn.Linear(emb_size, emb_size, bias=True)
        self.fc_dropout = nn.Dropout(p=fc_dropout)

        self.set_resnet_requires_grad(not resnet_freeze)

    def get_heat_map(self, g, caps):
        k = self.attention_k
        caps_cnt = caps.size(0)
        caps, indexes = torch.sort(caps, dim=1, descending=True)
        caps, indexes = caps[:, :k], indexes[:, :k]

        g = g.contiguous()
        imgs_cnt, emb_size, h, w = g.size()
        g = g.permute(1, 0, 2, 3).contiguous().view(emb_size, -1)
        g_ = torch.mm(self.fc.weight, g) #TODO A -> affine
        emb_size = self.fc.weight.size(0)
        g_ = g_.contiguous().view(emb_size, imgs_cnt, h, w).permute(1, 2, 3, 0)

        g_ = g_.view(imgs_cnt, 1, h, w, emb_size).expand(imgs_cnt, caps_cnt, h, w, emb_size)

        caps = caps.view(1, caps_cnt, 1, 1, k).expand(imgs_cnt, caps_cnt, h, w, k)
        indexes = indexes.view(1, caps_cnt, 1, 1, k).expand(imgs_cnt, caps_cnt, h, w, k)

        g_ = torch.gather(g_, dim=4, index=indexes)
        heat_maps = torch.sum(torch.abs(g_ * caps), dim=4)
        heat_maps /= torch.sum(heat_maps, dim=[2, 3]).view(imgs_cnt, caps_cnt, 1, 1).expand(imgs_cnt, caps_cnt, h, w)

        return heat_maps

    def attention_embedding(self, g, caps=None): # TODO use only max heat map element
        if caps is None:
            return None

        caps_cnt = caps.size(0)
        imgs_cnt, emb_size, h, w = g.size()

        heat_maps = self.get_heat_map(g, caps)
        heat_maps = heat_maps.view(imgs_cnt, caps_cnt, 1, h, w).expand(imgs_cnt, caps_cnt, emb_size, h, w)

        g = g.contiguous()
        g = g.view(imgs_cnt, 1, emb_size, h, w).expand(imgs_cnt, caps_cnt, emb_size, h, w)

        x = torch.sum(g * heat_maps, dim=[3, 4])
        x = self.fc(self.fc_dropout(x))

        return x

    def forward(self, imgs, caps=None):
        g = self.base_layer(imgs)
        h = self.wldPool(g)
        h = h.view(h.size(0), -1)
        x = self.fc(self.fc_dropout(h))

        x_att = self.attention_embedding(g, caps)

        return x, x_att, g

    def set_resnet_requires_grad(self, requires_grad):
        for param in self.base_layer.parameters():
            param.requires_grad = requires_grad

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
