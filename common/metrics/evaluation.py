import numpy as np

import torch

__all__ = ["Recall", "RecallMiddleHead"]


class Recall:
    def __init__(self, batch_size=1000, ks=[1, 5, 10], imgs_dupls=1):
        self.batch_size = batch_size
        self.ks = ks
        self.imgs_dupls = imgs_dupls
        self.batch_items = self.batch_size * self.imgs_dupls

        self.imgs_enc = []
        self.caps_enc = []

        self.res = []

    def cosine_sim(self, imgs, caps):
        imgs_norm = np.linalg.norm(imgs, axis=1)
        caps_norm = np.linalg.norm(caps, axis=1)

        scores = np.dot(imgs, caps.T)

        norms = np.dot(np.expand_dims(imgs_norm, 1),
                       np.expand_dims(caps_norm.T, 1).T)

        scores = (scores / norms)

        return scores

    def recall_by_ranks(self, ranks):
        recall_search = list()
        for k in self.ks:
            recall_search.append(
                len(np.where(ranks < k)[0]) / ranks.shape[0])
        return recall_search

    def recall_for_all(self, imgs_enc, caps_enc):
        imgs_dupls = self.imgs_dupls

        scores = self.cosine_sim(imgs_enc[::imgs_dupls], caps_enc)

        ranks = np.array([np.nonzero(np.in1d(row, np.arange(x * imgs_dupls, x * imgs_dupls + imgs_dupls, 1)))[0][0]
                          for x, row in enumerate(np.argsort(scores, axis=1)[:, ::-1])])
        recall_caps_search = self.recall_by_ranks(ranks)

        ranks = np.array([np.nonzero(row == x // imgs_dupls)[0][0]
                          for x, row in enumerate(np.argsort(scores.T, axis=1)[:, ::-1])])
        recall_imgs_search = self.recall_by_ranks(ranks)

        return recall_caps_search, recall_imgs_search

    def add(self, imgs, caps):
        self.imgs_enc += list(imgs)
        self.caps_enc += list(caps)

        if len(self.imgs_enc) >= self.batch_items:
            imgs, self.imgs_enc = self.imgs_enc[:self.batch_items], self.imgs_enc[self.batch_items:]
            caps, self.caps_enc = self.caps_enc[:self.batch_items], self.caps_enc[self.batch_items:]
            imgs, caps = np.stack(imgs, axis=0), np.stack(caps, axis=0)
            self.res.append(self.recall_for_all(imgs, caps))

    def get_res(self):
        res = self.res
        if len(res) > 0:
            return [np.mean([x[i] for x in res], axis=0) for i in range(len(res[0]))]
        else:
            return [[None] * len(ks)] * 2


class RecallMiddleHead(Recall):
    def __init__(self, model, batch_size=1000, ks=[1, 5, 10], imgs_dupls=1, sim_batch_size_imgs=10, sim_batch_size_caps=100):
        super().__init__(batch_size=batch_size, ks=ks, imgs_dupls=imgs_dupls)

        self.model = model
        self.sim_batch_size_imgs = sim_batch_size_imgs
        self.sim_batch_size_caps = sim_batch_size_caps

    def cosine_sim(self, g, caps):
        model = self.model
        device = next(model.parameters()).device

        imgs_cnt = g.shape[0]
        caps_cnt = caps.shape[0]
        scores = np.zeros((imgs_cnt, caps_cnt))

        emb_size = caps.shape[1]
        bs_imgs = self.sim_batch_size_imgs
        bs_caps = self.sim_batch_size_caps

        with torch.no_grad():
            for i in range(0, imgs_cnt, bs_imgs):
                g_tensor = torch.tensor(g[i:i + bs_imgs]).to(device, non_blocking=True)

                for j in range(0, caps_cnt, bs_caps):
                    caps_b = caps[j:j + bs_caps]
                    caps_tensor = torch.tensor(caps_b).to(device, non_blocking=True)

                    imgs = model.image_model.attention_embedding(g_tensor, caps_tensor)
                    imgs = imgs.cpu().data.numpy()

                    l_scores = np.sum(imgs * caps_b, axis=2)
                    l_scores /= np.linalg.norm(imgs, axis=2)
                    caps_b = np.reshape(caps_b, (1, bs_caps, emb_size))
                    caps_b = np.repeat(caps_b, bs_imgs, axis=0)
                    l_scores /= np.linalg.norm(caps_b, axis=2)

                    scores[i:i + bs_imgs, j:j + bs_caps] = l_scores

        return scores
