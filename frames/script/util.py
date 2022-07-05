import os
import functools
import operator

import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

seedling_list = (os.listdir('data'))
# ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
# 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
num2seedling_dic = dict(zip([i for i in range(12)], seedling_list))


def load_model(model_name: str, tuned: bool = False) -> models:
    """Load model from Pytorch pre-trained models, adjust its output layer into
    12 classes and freeze some low-feature layers"""
    # resnet
    if model_name in ["resnet50", "resnet101", "resnet152"]:
        model = getattr(models, model_name)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=2048, out_features=12, bias=True)
        for layer in ['layer3', 'layer4']:
            for param in getattr(model, layer).parameters():
                param.requires_grad = True
    # inception_v3
    elif model_name in ['inception_v3']:
        model = getattr(models, model_name)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=2048, out_features=12, bias=True)
        for layer in ['Mixed_5b', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
                      'Mixed_7a', 'Mixed_7b', 'Mixed_7c']:
            for param in getattr(model, layer).parameters():
                param.requires_grad = True
        model.aux_logits = False
    # resnext
    elif model_name in ['resnext50', 'resnext101']:
        if model_name == 'resnext50':
            model_name = 'resnext50_32x4d'
        elif model_name == 'resnext101':
            model_name = 'resnext101_32x8d'
        model = getattr(models, model_name)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=2048, out_features=12, bias=True)
        for layer in ['layer3', 'layer4']:
            for param in getattr(model, layer).parameters():
                param.requires_grad = True
    # densenet
    elif model_name in ['densenet121', 'densenet161']:
        model = getattr(models, model_name)(pretrained=True)
        if model_name == 'densenet121':
            model.classifier = nn.Linear(
                in_features=1024, out_features=12, bias=True)
        else:
            model.classifier = nn.Linear(
                in_features=2208, out_features=12, bias=True)
        for param in model.parameters():
            param.requires_grad = False
        for layer in ['denseblock3', 'denseblock4']:
            for param in getattr(model.features, layer).parameters():
                param.requires_grad = True
    else:
        raise NotImplementedError
    model.name = model_name

    if not tuned:
        return model
    else:
        for param in model.parameters():
            param.requires_grad = False
        path = './tuned-models/%s.pt' % model_name
        model.load_state_dict(torch.load(path))
        return model


def imshow(inp, c: list = None) -> None:
    """show figure from 3-dim vector"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(16, 4))
    if c is not None:
        title = [num2seedling_dic[int(x)] for x in c]
        plt.title(title)
    plt.imshow(inp)


def segmentation(im, s: int = 35) -> Image:
    """remove background of the figure"""
    im = np.array(im)
    image_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([60 - s, 100, 50])
    upper_hsv = np.array([60 + s, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    output = cv2.bitwise_and(im, im, mask=mask)

    image_blurred = cv2.GaussianBlur(output, (5, 5), 0)
    image_sharp = cv2.addWeighted(output, 1.5, image_blurred, -0.5, 0)
    return Image.fromarray(image_sharp)


def plot_cm(model: torch.nn.Module, dataloader: torch.utils.data.dataloader) -> None:
    """print confusion matrix"""
    y_pred = []
    y_true = []
    device = torch.device('cuda')
    model.eval()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        y_pred.append(preds.tolist())
        y_true.append(labels.tolist())
    # reduce the nest lists created by the mini-batch to list
    y_pred = functools.reduce(operator.iconcat, y_pred, [])
    y_true = functools.reduce(operator.iconcat, y_true, [])
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=tuple(seedling_list))
    _, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.title(model.name, fontsize=16)
    plt.show()


def plot_sta(sta: dict, interval: int = None, title: str = None) -> None:
    """plot the statistics for acc/loss of train/dev"""
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    epochs = len(sta['train']['epoch_acc'])
    max_d_acc = max(sta['dev']['epoch_acc'])
    max_d_acc_index = sta['dev']['epoch_acc'].index(max_d_acc) + 1

    # plot acc
    ax = plt.subplot(121)
    t_acc = sta['train']['epoch_acc']
    d_acc = sta['dev']['epoch_acc']
    tline, = plt.plot(np.append(np.roll(t_acc, 1),
                                t_acc[epochs - 1]), color='g')
    dline, = plt.plot(np.append(np.roll(d_acc, 1),
                                d_acc[epochs - 1]), linestyle=":", color='r')
    plt.grid(color="k", linestyle=":")
    plt.legend((tline, dline), ('train', 'dev'))
    plt.ylabel('acc')
    plt.xlabel('iterations')
    ax.set_xlim(1, epochs)
    if interval is not None:
        dim = np.arange(1, epochs + 1, interval)
        plt.xticks(dim)
    plt.scatter(max_d_acc_index, max_d_acc, s=40, color='black')

    # plot loss
    t_loss = sta['train']['epoch_loss']
    d_loss = sta['dev']['epoch_loss']
    ax = plt.subplot(122)
    tlline, = plt.plot(np.append(np.roll(t_loss, 1),
                                 t_loss[epochs - 1]), color='g')
    dlline, = plt.plot(np.append(np.roll(d_loss, 1),
                                 d_loss[epochs - 1]), linestyle=":", color='r')
    plt.grid(color="k", linestyle=":")
    plt.legend((tlline, dlline), ('train', 'dev'))
    plt.ylabel('loss')
    plt.xlabel('iterations')
    ax.set_xlim(1, epochs)
    if interval is not None:
        dim = np.arange(1, epochs + 1, interval)
        plt.xticks(dim)

    plt.show()

    print("max train acc: " + str(max(t_acc)))
    print("max dev acc: " + str(max_d_acc) +
          " at epoch " + str(max_d_acc_index))
    print("corresponding train acc: " + str(t_acc[max_d_acc_index - 1]))
