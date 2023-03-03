import argparse
import distutils.util
import random
import os
import numpy as np
import torch


import data
import methods
import resnet


def main():
    args = parse_args()
    run_experiment(args)


def run_experiment(args):
    seed_everything(args.seed)

    benchmark, classes_per_task = data.get_data(args.dataset, args.n_experiences, args.seed, args.image_size)
    num_classes = sum(classes_per_task)

    model = resnet.resnet18(num_classes=num_classes)
    torch.save(model.state_dict(), f'weights/resnet_before_training.pth')
    cl_strategy, mlf_logger = methods.get_strategy(args, benchmark, model)

    results = []
    for i, experience in enumerate(benchmark.train_stream):
        if i >= args.train_on_experiences:
            break

        cl_strategy.train(experience, num_workers=args.num_workers)

        selected_tasks = [benchmark.test_stream[j] for j in range(0, i+1)]
        eval_results = cl_strategy.eval(selected_tasks, num_workers=20)
        results.append(eval_results)

        forgetting = get_forgetting(eval_results)
        if forgetting is not None and forgetting > args.forgetting_stopping_threshold:
            print(f'Stopping training after task {i} due to large forgetting: {forgetting}')
            break

        torch.save(model.state_dict(), f'weights/resnet_after_{i}.pth')

    avrg_acc = 0.0
    for j in range(args.train_on_experiences):
        acc_name = 'Top1_Acc_Stream/eval_phase/test_stream/Task{:03d}'.format(j)
        task_acc = eval_results[acc_name]
        avrg_acc += task_acc

    avrg_acc /= args.train_on_experiences
    print('acc = ', avrg_acc)

    if mlf_logger:
        mlf_logger.log_model(cl_strategy.model)
        mlf_logger.log_avrg_accuracy()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--experiment', default='Default', help='mlflow experiment name')
    parser.add_argument('--nested_run', action='store_true', help='create nested run in mlflow')
    parser.add_argument('--debug', action='store_true', help='if true, execute only one iteration in training epoch')
    parser.add_argument('--interactive_logger', default=False, type=distutils.util.strtobool, help='if True use interactive logger with tqdm for printing in console')

    parser.add_argument('--method', default='replay', choices=('baseline', 'cumulative', 'extended_consistency',
                                                               'agem', 'replay', 'consistency',
                                                               'lwf', 'mir', 'hat', 'cat'))
    parser.add_argument('--base_model', default='resnet18', choices=('resnet18', 'reduced_resnet18', 'simpleMLP'))
    parser.add_argument('--dataset', default='cifar100', choices=('cifar100', 'cifar10', 'mnist', 'permutation-mnist', 'tiny-imagenet',
                        'cifar10-mnist-fashion-mnist', 'mnist-fashion-mnist-cifar10', 'fashion-mnist-cifar10-mnist', '5-datasets', 'cores50'))
    parser.add_argument('--n_experiences', default=20, type=int)
    parser.add_argument('--train_on_experiences', default=20, type=int)
    parser.add_argument('--forgetting_stopping_threshold', default=0.5, type=float)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--image_size', default=32, type=int)

    parser.add_argument('--mem_size', default=500, type=int)
    parser.add_argument('--regularisation_type', default='L1', choices=('L1', 'L2', 'Linf', 'MSE', 'KD', 'cosine'))
    parser.add_argument('--alpha', default=0.2, type=float, help='parameter for consistency regularisation reponsible for consistency regularizer term for predictions')
    parser.add_argument('--beta', default=1.0, type=float, help='parameter for consistency regularisation reponsible for cross entropy term for previous tasks')
    parser.add_argument('--use_layer', default=-1, type=int, help='what activations to use')
    parser.add_argument('--T', default=5, type=int, help='temparature softmax scaling for disitilation')

    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # change to true for faster convergence


def get_forgetting(results):
    forgetting = None
    try:
        forgetting = results['StreamForgetting/eval_phase/test_stream']
    except KeyError:
        pass
    return forgetting


if __name__ == "__main__":
    main()
