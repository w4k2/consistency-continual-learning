import torch
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Lambda

from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10, SplitTinyImageNet
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.core50.core50 import CORe50Dataset
from avalanche.benchmarks.scenarios.new_classes import NCScenario

from data.notmnist import NOTMNIST


def get_data(dataset_name, n_experiences, seed, image_size):
    benchmark = None
    if dataset_name == 'cifar10':
        norm_stats = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = SplitCIFAR10(n_experiences=n_experiences,
                                 train_transform=train_transforms,
                                 eval_transform=eval_transforms,
                                 seed=seed,
                                 return_task_id=True
                                 )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'cifar100':
        norm_stats = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = SplitCIFAR100(n_experiences=n_experiences,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms,
                                  seed=seed,
                                  return_task_id=True
                                  )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'mnist':
        norm_stats = (0.1307,), (0.3081,)
        train_transforms, eval_transforms = get_mnist_transforms(norm_stats, image_size)
        benchmark = SplitMNIST(n_experiences=n_experiences,
                               train_transform=train_transforms,
                               eval_transform=eval_transforms,
                               seed=seed
                               )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'permutation-mnist':
        norm_stats = (0.1307,), (0.3081,)
        train_transforms, eval_transforms = get_mnist_transforms(norm_stats, image_size)
        benchmark = PermutedMNIST(n_experiences=n_experiences,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms,
                                  seed=seed
                                  )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'tiny-imagenet':
        norm_stats = (0.4443, 0.4395, 0.4250), (0.3138, 0.3181, 0.3182)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = SplitTinyImageNet(n_experiences=n_experiences,
                                      train_transform=train_transforms,
                                      eval_transform=eval_transforms,
                                      seed=seed,
                                      return_task_id=True,
                                      )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'cifar10-mnist-fashion-mnist':
        benchmark = get_multidataset_benchmark(('cifar', 'mnist', 'fmnist'), image_size, seed)
        classes_per_task = [10, 10, 10]
    elif dataset_name == 'mnist-fashion-mnist-cifar10':
        benchmark = get_multidataset_benchmark(('mnist', 'fmnist', 'cifar'), image_size, seed)
        classes_per_task = [10, 10, 10]
    elif dataset_name == 'fashion-mnist-cifar10-mnist':
        benchmark = get_multidataset_benchmark(('fmnist', 'cifar', 'mnist'), image_size, seed)
        classes_per_task = [10, 10, 10]
    elif dataset_name == '5-datasets':
        benchmark = get_multidataset_benchmark(('svhn', 'cifar', 'mnist', 'fmnist', 'notmnist'), image_size, seed)
        classes_per_task = [10, 10, 10, 10, 10]
    elif dataset_name == 'cores50':
        norm_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)

        dataset_root = default_dataset_location('core50')
        # Download the dataset and initialize filelists
        train_dataset = CORe50Dataset(root=dataset_root, train=True, transform=train_transforms, download=True, mini=False)
        test_dataset = CORe50Dataset(root=dataset_root, train=False, transform=eval_transforms, download=True, mini=False)
        benchmark = NCScenario(train_dataset, test_dataset,
                               n_experiences=n_experiences, task_labels=True, shuffle=True, seed=seed,
                               class_ids_from_zero_in_each_exp=True)

        classes_per_task = [len(exp.classes_in_this_experience) for exp in benchmark.train_stream]

    return benchmark, classes_per_task


def get_transforms(norm_stats, image_size):
    train_list = [
        # Resize((image_size, image_size)),
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(*norm_stats),
    ]
    train_transforms = Compose(train_list)
    eval_list = [
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(*norm_stats),
    ]
    eval_transforms = Compose(eval_list)
    return train_transforms, eval_transforms


def get_mnist_transforms(norm_stats, image_size):
    transform_list = [
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(*norm_stats),
        Lambda(lambda x: torch.cat([x, x, x], dim=0))
    ]
    train_transforms = Compose(transform_list)
    eval_transforms = Compose(transform_list)
    return train_transforms, eval_transforms


def get_multidataset_benchmark(order, image_size, seed):
    cifar10_norm_stats = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    mnist_norm_stats = (0.1307,), (0.3081,)
    fmnist_norm_stats = (0.2860,), (0.3530,)

    cifar10_train_transforms, cifar10_eval_transforms = get_transforms(cifar10_norm_stats, image_size)
    mnist_train_transforms, mnist_eval_transforms = get_mnist_transforms(mnist_norm_stats, image_size)
    fmnist_train_transforms, fmnist_eval_transforms = get_mnist_transforms(fmnist_norm_stats, image_size)
    svhn_train_transforms, svhn_eval_transforms = get_mnist_transforms(mnist_norm_stats, image_size, stack_channels=False)
    notmnist_train_transforms, notmnist_eval_transforms = get_mnist_transforms(mnist_norm_stats, image_size)

    train_datasets = []
    test_datasets = []

    for dataset_name in order:
        if dataset_name == 'cifar':
            train_datasets.append(CIFAR10('./data/datasets', train=True, transform=cifar10_train_transforms, download=True))
            test_datasets.append(CIFAR10('./data/datasets', train=False, transform=cifar10_eval_transforms, download=True))
        elif dataset_name == 'mnist':
            train_datasets.append(MNIST('./data/datasets', train=True, transform=mnist_train_transforms, download=True))
            test_datasets.append(MNIST('./data/datasets', train=False, transform=mnist_eval_transforms, download=True))
        elif dataset_name == 'fmnist':
            train_datasets.append(FashionMNIST('./data/datasets', train=True, transform=fmnist_train_transforms, download=True))
            test_datasets.append(FashionMNIST('./data/datasets', train=False, transform=fmnist_eval_transforms, download=True))
        elif dataset_name == 'svhn':
            train_svhn = SVHN('./data/datasets', split='train', transform=svhn_train_transforms, download=True)
            train_svhn.targets = train_svhn.labels
            train_datasets.append(train_svhn)
            test_svhn = SVHN('./data/datasets', split='test', transform=svhn_eval_transforms, download=True)
            test_svhn.targets = test_svhn.labels
            test_datasets.append(test_svhn)
        elif dataset_name == 'notmnist':
            train_datasets.append(NOTMNIST('./data/datasets', train=True, transforms=notmnist_train_transforms, seed=seed))
            test_datasets.append(NOTMNIST('./data/datasets', train=False, transforms=notmnist_eval_transforms, seed=seed))
        else:
            raise ValueError("Invalid dataset name")

    benchmark = dataset_benchmark(train_datasets, test_datasets)
    new_train_stream = []
    for i, exp in enumerate(benchmark.train_stream):
        new_dataset = exp.dataset
        new_dataset.targets_task_labels = [i for _ in range(len(new_dataset.targets_task_labels))]
        exp.dataset = new_dataset
        new_train_stream.append(exp)
    benchmark.train_stream = new_train_stream

    new_test_stream = []
    for i, exp in enumerate(benchmark.test_stream):
        new_dataset = exp.dataset
        new_dataset.targets_task_labels = [i for _ in range(len(new_dataset.targets_task_labels))]
        exp.dataset = new_dataset
        new_test_stream.append(exp)
    benchmark.test_stream = new_test_stream

    return benchmark
