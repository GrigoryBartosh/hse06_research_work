import torch
import torch.nn as nn

__all__ = ["ContrastiveLoss", "HardNegativeContrastiveLoss"]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, use_caps=True, use_imgs=True):
        super().__init__()
        self.margin = margin
        self.use_caps = use_caps
        self.use_imgs = use_imgs

    def get_scores_diag(self, imgs, caps):
        if len(imgs.size()) == 2:
            scores = torch.mm(imgs, caps.t())
        else:
            batch_size, emb_size = caps.size()
            caps = caps.view(1, batch_size, emb_size).expand_as(imgs)
            scores = torch.sum(imgs * caps, dim=2)

        diag = scores.diag()

        return scores, diag 

    def forward(self, imgs, caps):
        scores, diag = self.get_scores_diag(imgs, caps)

        cost_c = torch.clamp(self.margin - diag.expand_as(scores) + scores, min=0)
        cost_i = torch.clamp(self.margin - diag.view(-1, 1).expand_as(scores) + scores, min=0)

        diag_c = torch.diag(cost_c.diag())
        diag_i = torch.diag(cost_i.diag())

        cost_c = torch.mean(cost_c - diag_c)
        cost_i = torch.mean(cost_i - diag_i)

        cost = 0
        if self.use_caps:
            cost = cost + cost_c
        if self.use_imgs:
            cost = cost + cost_i

        return cost


class HardNegativeContrastiveLoss(ContrastiveLoss):
    def __init__(self, nmax=1, margin=0.2, use_caps=True, use_imgs=True):
        super().__init__(margin, use_caps, use_imgs)
        self.nmax = nmax

    def get_negs(self, imgs, caps):
        scores, diag = super().get_scores_diag(imgs, caps)

        scores = (scores - 2 * torch.diag(scores.diag()))

        sorted_cap, _ = torch.sort(scores, dim=0, descending=True)
        sorted_img, _ = torch.sort(scores, dim=1, descending=True)

        max_c = sorted_cap[:self.nmax, :]
        max_i = sorted_img[:, :self.nmax]

        neg_cap = self.margin - diag.view(1, -1).expand_as(max_c) + max_c
        neg_img = self.margin - diag.view(-1, 1).expand_as(max_i) + max_i
        neg_cap = torch.clamp(neg_cap, min=0)
        neg_img = torch.clamp(neg_img, min=0)
        neg_cap = torch.sum(neg_cap, dim=0)
        neg_img = torch.sum(neg_img, dim=1)

        return neg_cap, neg_img

    def forward(self, imgs, caps):
        neg_cap, neg_img = self.get_negs(imgs, caps)

        neg_cap = torch.mean(neg_cap)
        neg_img = torch.mean(neg_img)

        loss = 0
        if self.use_caps:
            loss = loss + neg_cap
        if self.use_imgs:
            loss = loss + neg_img

        return loss
