from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

from .replay import ReplayModified


def get_strategy(args, benchmark, model):
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger],
        benchmark=benchmark
    )
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = CrossEntropyLoss()

    if args.method == 'naive':
        cl_strategy = Naive(
            model, optimizer, criterion,
            train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
            train_epochs=args.epochs, evaluator=eval_plugin,
            device=args.device)
    elif args.method == 'replay':
        cl_strategy = ReplayModified(model, optimizer, criterion, mem_size=args.mem_size,
                                     train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                     device=args.device, train_epochs=args.n_epochs,
                                     evaluator=eval_plugin, eval_every=-1)

    return cl_strategy
