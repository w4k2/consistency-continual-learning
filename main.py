import argparse
import distutils.util

import data
import methods

from torchvision.models import resnet


def main():
    args = parse_args()

    benchmark, classes_per_task = data.get_data(args.dataset, args.n_experiences, args.seed, args.image_size)
    num_classes = sum(classes_per_task)

    model = resnet.resnet18(num_classes=num_classes)

    cl_strategy = methods.get_strategy(args, benchmark, model)

    results = []
    for i, experience in enumerate(benchmark.train_stream):
        cl_strategy.train(experience, num_workers=args.num_workers)

        selected_tasks = [benchmark.test_stream[j] for j in range(0, i+1)]
        eval_results = cl_strategy.eval(selected_tasks, num_workers=20)
        results.append(eval_results)

    avrg_acc = 0.0
    for j in range(len(benchmark.train_stream)):
        acc_name = 'Top1_Acc_Stream/eval_phase/test_stream/Task{:03d}'.format(j)
        task_acc = eval_results[acc_name]
        avrg_acc += task_acc

    avrg_acc /= len(benchmark.train_stream)
    print('acc = ', avrg_acc)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--experiment', default='Default', help='mlflow experiment name')
    parser.add_argument('--nested_run', action='store_true', help='create nested run in mlflow')
    parser.add_argument('--debug', action='store_true', help='if true, execute only one iteration in training epoch')
    parser.add_argument('--interactive_logger', default=False, type=distutils.util.strtobool, help='if True use interactive logger with tqdm for printing in console')

    parser.add_argument('--method', default='replay', choices=('baseline', 'cumulative',
                                                               'agem', 'replay', 'consistency',
                                                               'lwf', 'mir', 'hat', 'cat'))
    parser.add_argument('--base_model', default='resnet18', choices=('resnet18', 'reduced_resnet18', 'simpleMLP'))
    parser.add_argument('--dataset', default='cifar100', choices=('cifar100', 'cifar10', 'mnist', 'permutation-mnist', 'tiny-imagenet',
                        'cifar10-mnist-fashion-mnist', 'mnist-fashion-mnist-cifar10', 'fashion-mnist-cifar10-mnist', '5-datasets', 'cores50'))
    parser.add_argument('--n_experiences', default=20, type=int)

    def train_exp_type(value):
        try:
            return int(value)
        except:
            if value == 'all':
                return value
            else:
                raise ValueError("Invalid train_on_experiences, should be int or 'all'")
    parser.add_argument('--train_on_experiences', default='all', type=train_exp_type)
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
    parser.add_argument('--regularisation_type', default='L1', choices=('L1', 'L2', 'Linf', 'MSE'))
    parser.add_argument('--alpha', default=1.0, type=float, help='parameter for consistency regularisation reponsible for cross entropy term for previous tasks')
    parser.add_argument('--beta', default=1.0, type=float, help='parameter for consistency regularisation reponsible for  consistency regularizer term for predictions')

    args = parser.parse_args()
    if args.train_on_experiences == 'all':
        args.train_on_experiences = args.n_experiences
    return args


if __name__ == "__main__":
    main()
