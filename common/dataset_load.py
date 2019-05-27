import os

import numpy as np

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

from common.config import PATH
from common.text_utils import TextEncoder

__all__ = ["get_loader"]


class DatasetCOCO(data.Dataset):
    def __init__(self, imgs_dir, captions_path, transform=None):
        self.imgs_dir = imgs_dir
        self.coco = COCO(captions_path)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

        image_id_to_ids = {}
        for i in self.ids:
            image_id = self.coco.anns[i]['image_id']
            if image_id not in image_id_to_ids:
                image_id_to_ids[image_id] = []
            image_id_to_ids[image_id].append(i)

        image_ids_gen = filter(lambda i: len(image_id_to_ids[i]) >= 5, image_id_to_ids.keys())
        self.ids = []
        for image_id in image_ids_gen:
            self.ids += image_id_to_ids[image_id][:5]

        self.text_encoder = TextEncoder()
        self.vocab_size = self.text_encoder.get_vocab_size()

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.imgs_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption = self.text_encoder.encode([caption])[0]
        l, vs = len(caption), self.vocab_size
        caption = np.array([caption, list(range(vs, vs + l))])
        caption = caption.transpose()

        return image, caption

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = list(map(len, captions))
    shape = np.zeros(captions[0].ndim + 1, dtype=np.int32)
    shape[0] = len(captions)
    shape[1:] = captions[0].shape
    shape[1] = max(lengths)
    captions_ts = np.zeros(shape, dtype=np.int32)
    for i, cap in enumerate(captions):
        end = lengths[i]
        captions_ts[i, :end] = cap[:end]

    lengths = torch.LongTensor(lengths)
    captions_ts = torch.LongTensor(captions_ts)

    return images, captions_ts, lengths

def get_loader(dataset_name="COCO", sset="VAL", transform=None, batch_size=5, shuffle=False, num_workers=4):
    if transform is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        if sset == "TRAIN":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((350, 350)),
                transforms.ToTensor(),
                normalize,
            ])

    if dataset_name == "COCO":
        dataset = DatasetCOCO(
            imgs_dir=PATH["DATASETS"][dataset_name][sset]["IMAGES_DIR"],
            captions_path=PATH["DATASETS"][dataset_name][sset]["CAPTIONS"],
            transform=transform
        )

    return data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
