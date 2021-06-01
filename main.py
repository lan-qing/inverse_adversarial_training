import os
import random
import functools

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from utils import *
from model import *
from visualization import to_img

if __name__ == '__main__':
    seed = 55
    print("Use random seed ", seed)
    signature = "test"
    rootpath = f"results/{signature}_seed{seed}/"
    if not os.path.isdir(rootpath):
        os.mkdir(rootpath)

    set_seed(seed)

    NTrainPointsMNIST = 60000
    batch_size = 100
    log_interval = 1

    transform_train = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    trainset = datasets.MNIST(root='/slstore/tianfeng/data', train=True, download=True, transform=transform_train)
    valset = datasets.MNIST(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                          num_workers=3)
    # to_img(torch.reshape(trainset[0][0], [28, 28]).cpu().numpy(), "example.png")
    # exit()
    net = LenetWrapper(N=NTrainPointsMNIST)
    # net = MLPWrapper(N=NTrainPointsMNIST)
    for epoch in range(2000):
        lr = 0.01
        net.fit_inverse(trainloader, lr=lr, epoch=epoch)

        print("generating...")
        img = net.generate_from_noise()
        to_img(torch.reshape(img, [32, 32]).cpu().numpy(), f"{rootpath}res{epoch}.png")

    net.save(rootpath + "lenet_50_inverse.pt")

    # net.load(rootpath + "lenet_50.pt")
    # img = net.generate_from_noise()
    # to_img(torch.reshape(img, [32, 32]).cpu().numpy(), "tmp8.png")
