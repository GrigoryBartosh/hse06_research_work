import numpy as np

import torch
import torch.optim as optim

from common.config import PATH
from common.dataset_load import get_loader
from common.logging import TensorBoard
from common.model_save_load import save_unimodel, load_unimodel_state

from models.unimodel import UniModelTransformer as UniModel

from common.metrics.loss import ContrastiveLoss, HardNegativeContrastiveLoss
from common.metrics.evaluation import Recall, RecallMiddleHead

args = {
    "model_name": "model",
    "dataset_name": "COCO",
    "model": {
        "emb_size": 2400,
        "image": {
            "attention_k": 180,
            "pretrained": True,
            "resnet_freeze": True,
            "fc_dropout": 0.5
        },
        "text": {
            "pretrained": True,
            "transformer_freeze": True,
            "fc_dropout": 0.1
        }
    },
    "train": {
        "epochs": 120,
        "batch_size": 32,
        "lr": 0.001,
        "loss_margin": 0.2,
        "use_caps": True,
        "use_imgs": True,
        "use_base_head": True,
        "use_middle_head": False
    },
    "recall": {
        "do_it": True,
        "batch_size": 1000,
        "ks": [1, 5, 10],
        "imgs_dupls": 5,
        "sim_batch_size_imgs": 10,
        "sim_batch_size_caps": 100
    }
}


def train(model, data_loader, criterion, optimizer, device, num_epochs=50):
    tensor_board = TensorBoard()

    model_name = args["model_name"]
    save_unimodel(model, model_name)

    for epoch in range(num_epochs):
        for phase in list(data_loader.keys()):
            if phase == 'train':
                model.train()
            else:
                model.eval()
                recall = Recall(
                    batch_size=args["recall"]["batch_size"],
                    ks=args["recall"]["ks"],
                    imgs_dupls=args["recall"]["imgs_dupls"])
                recall_middle_head = RecallMiddleHead(
                    model,
                    batch_size=args["recall"]["batch_size"],
                    ks=args["recall"]["ks"],
                    imgs_dupls=args["recall"]["imgs_dupls"],
                    sim_batch_size_imgs=args["recall"]["sim_batch_size_imgs"],
                    sim_batch_size_caps=args["recall"]["sim_batch_size_caps"])

            running_loss_base, running_loss_att, running_loss = 0.0, 0.0, 0.0
            for i, (imgs, caps, lengths) in enumerate(data_loader[phase]):
                imgs = imgs.to(device, non_blocking=True)
                caps = caps.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output_imgs, output_imgs_att, output_caps, output_g = model(imgs, caps, lengths)
                    loss_base = criterion(output_imgs, output_caps)
                    loss_att = criterion(output_imgs_att, output_caps)
                    loss = 0
                    if (args["train"]["use_base_head"]):
                        loss += loss_base
                    if (args["train"]["use_middle_head"]):
                        loss += loss_att

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    elif args["recall"]["do_it"]:
                        imgs_enc = output_imgs.cpu().data.numpy()
                        g_enc = output_g.cpu().data.numpy()
                        caps_enc = output_caps.cpu().data.numpy()
                        recall.add(imgs_enc, caps_enc)
                        recall_middle_head.add(g_enc, caps_enc)

                running_loss_base += loss_base.item() * imgs.size(0)
                running_loss_att += loss_att.item() * imgs.size(0)
                running_loss += loss.item() * imgs.size(0)
                tensor_board.add_to_graph("00_" + phase + "_base_loss", loss_base.item())
                tensor_board.add_to_graph("01_" + phase + "_att_loss", loss_att.item())
                tensor_board.add_to_graph("02_" + phase + "_loss", loss.item())

            epoch_loss_base = running_loss_base / len(data_loader[phase].dataset)
            epoch_loss_att = running_loss_att / len(data_loader[phase].dataset)
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            tensor_board.add_to_graph("10_" + phase + "_base_epoch_loss", epoch_loss_base)
            tensor_board.add_to_graph("11_" + phase + "_att_epoch_loss", epoch_loss_att)
            tensor_board.add_to_graph("12_" + phase + "_epoch_loss", epoch_loss)

            if phase == 'val':
                save_unimodel(model, model_name)
                if args["recall"]["do_it"]:
                    recall_res = recall.get_res()
                    recall_middle_head_res = recall_middle_head.get_res()
                    for i, k in enumerate(args["recall"]["ks"]):
                        tensor_board.add_to_graph("2{}_recall_search_text_by_image_k={}".format(i, k), recall_res[0][i])
                        tensor_board.add_to_graph("3{}_recall_search_image_by_text_k={}".format(i, k), recall_res[1][i])
                        tensor_board.add_to_graph("4{}_recall_middle_head_search_text_by_image_k={}".format(i, k), recall_middle_head_res[0][i])
                        tensor_board.add_to_graph("5{}_recall_middle_head_search_image_by_text_k={}".format(i, k), recall_middle_head_res[1][i])

    return model

if __name__ == '__main__':
    dataset_name = args["dataset_name"]

    batch_size = args["train"]["batch_size"]
    data_loader = {
        "train": get_loader(
            dataset_name=dataset_name, 
            sset="TRAIN", 
            batch_size=batch_size,
            shuffle=True),
        "val": get_loader(
            dataset_name=dataset_name, 
            sset="VAL", 
            batch_size=batch_size, 
            shuffle=False)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UniModel(args["model"])
    #load_unimodel_state(model, args["model_name"], device)
    model.to(device)

    criterion = ContrastiveLoss(
        margin=args["train"]["loss_margin"],
        use_caps=args["train"]["use_caps"],
        use_imgs=args["train"]["use_imgs"])

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args["train"]["lr"])

    model = train(
        model, 
        data_loader, 
        criterion,
        optimizer, 
        device, 
        num_epochs=args["train"]["epochs"])
