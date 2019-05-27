from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

import torch
import torchvision.transforms as transforms

from common.config import PATH
from common.model_save_load import load_unimodel_state
from common.text_utils import TextEncoder

from models.unimodel import UniModelTransformer as UniModel

args = {
    "model_name": "model",
    "model": {
        "emb_size": 2400,
        "image": {
            "attention_k": 180,
            "pretrained": True,
            "resnet_freeze": False,
            "fc_dropout": 0.5
        },
        "text": {
            "pretrained": True,
            "transformer_freeze": False,
            "fc_dropout": 0.1
        }
    }
}

IMG_PATH = "img.jpg"
TEXT = u"a man"

def load_img(path):
    img = Image.open(path).convert('RGB')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.Resize(500),
        transforms.ToTensor(),
        normalize,
    ])

    img = transform(img).data.numpy()
    return img

def encode_text(text_encoder, caption):
    caption = text_encoder.encode([caption])[0]
    l, vs = len(caption), text_encoder.get_vocab_size()
    caption = np.array([caption, list(range(vs, vs + l))])
    caption = caption.transpose()
    return caption, l

def get_heat_map(model, img, cap, length):
    with torch.no_grad():
        img = torch.tensor([img]).to(device, non_blocking=True)
        cap = torch.tensor([cap]).to(device, non_blocking=True)
        length = torch.tensor([length]).to(device, non_blocking=True)

        _, _, output_cap, output_g = model(img, cap, length)
        heat_map = model.image_model.get_heat_map(output_g, output_cap)

        heat_map = heat_map.cpu().data.numpy()

    return heat_map[0][0]

def show_image(img):
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    plt.axis("off")
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    text_encoder = TextEncoder()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UniModel(args["model"])
    load_unimodel_state(model, args["model_name"], device)
    model.to(device)
    model.eval()

    img = image = load_img(IMG_PATH)
    cap, length = encode_text(text_encoder, TEXT)

    heat_map = get_heat_map(model, img, cap, length)
    show_image(heat_map)
