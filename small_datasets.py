import random
import os
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models.resnet import resnet18
from torchvision.datasets import CIFAR100
from torchvision.transforms import *


def main():
    device = "cuda:0"
    model = resnet18(num_classes=100)
    model = model.to(device)

    norm_stats = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    train_transforms = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(*norm_stats),
    ])
    test_transforms = Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize(*norm_stats),
    ])
    train_dataset = CIFAR100('data/', train=True, transform=train_transforms, download=True)
    test_dataset = CIFAR100('data/', train=True, transform=test_transforms, download=True)
    new_size = 1.0
    if new_size < 1.0:
        class_size = int(len(train_dataset) * new_size / 100)
        idx = np.concatenate([np.argwhere(np.array(train_dataset.targets) == class_idx).flatten()[:class_size] for class_idx in range(100)])
        train_dataset = torch.utils.data.Subset(train_dataset, idx)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
    test_dataloder = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)
    # critertion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    epochs = 50

    # for epoch in range(epochs):
    #     model.train()
    #     for inp, target in train_dataloader:
    #         optimizer.zero_grad()
    #         inp = inp.to(device)
    #         target = target.to(device)

    #         y_pred = model(inp)
    #         loss = critertion(y_pred, target).sum()
    #         loss.backward()
    #         optimizer.step()

    #     test_acc = eval(model, test_dataloder, device)
    #     print(f'epoch {epoch} finished, test acc = {test_acc}')

    # torch.save(model.state_dict(), 'weights.pth')

    teacher_model = resnet18(num_classes=100)
    teacher_model.load_state_dict(torch.load('weights.pth'))
    teacher_model = teacher_model.to(device)
    print(f'test acc after loading = {eval(teacher_model, test_dataloder, device)}')
    teacher_model = teacher_model.eval()

    student_model = resnet18(num_classes=100)
    student_model = student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=0.0001)
    T = 20
    alpha = 0.8

    for epoch in range(epochs):
        student_model.train()
        for inp, target in train_dataloader:
            optimizer.zero_grad()
            inp = inp.to(device)
            target = target.to(device)

            y_pred = student_model(inp)
            loss_cross_entropy = F.cross_entropy(y_pred, target)

            with torch.no_grad():
                teacher_pred = teacher_model(inp)
                # teacher_pred = F.softmax(teacher_pred / T, dim=1)
            # y_pred_soft = F.softmax(y_pred / T, dim=1)
            # loss_distilation = F.kl_div(y_pred_soft, teacher_pred, size_average=False)
            loss_distilation = 1 - F.cosine_similarity(y_pred, teacher_pred).mean()

            loss = alpha * loss_cross_entropy + (1-alpha) * loss_distilation
            loss.backward()
            optimizer.step()

        test_acc = eval(student_model, test_dataloder, device)
        print(f'student training epoch {epoch} finished, test acc = {test_acc}')


def eval(model, test_dataloader, device):
    acc = 0.0
    n = 0

    model.eval()
    with torch.no_grad():
        for inp, target in test_dataloader:
            inp = inp.to(device)
            target = target.to(device)
            y_pred = model(inp)
            y_pred = torch.argmax(y_pred, dim=1)
            correct = torch.sum(y_pred == target).item()
            acc += correct
            n += y_pred.shape[0]

    acc = acc / n
    return acc


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # change to true for faster convergence


if __name__ == '__main__':
    main()
