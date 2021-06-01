import torch
import torch.nn as nn
import numpy as np


def pgd_attack_reverse(model, images, labels, eps=1.0, alpha=0.1, iters=20, half=False, double=False):
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def pgd_attack_reverse_binary(model, images, labels, eps=1.0, alpha=0.1, iters=20, half=False, double=False,
                              verbose=False):
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.BCEWithLogitsLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        # print(outputs, labels, cost)
        cost.backward()
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    if verbose:
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels)
        print(cost)
    return images
