import utils

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

from .replay import ReplayModified
from .consistency_regularisation import ConsistencyRegularistaion
from .extended_cr import ExtendedConsistencyRegularistaion


def get_strategy(args, benchmark, model):
    loggers = list()
    if args.interactive_logger:
        loggers.append(InteractiveLogger())
    else:
        loggers.append(TextLogger())
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
        benchmark=benchmark
    )

    plugins = list()
    if args.debug:
        plugins.append(utils.DebugingPlugin())

    optimizer = SGD(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    if args.method == 'naive':
        cl_strategy = Naive(
            model, optimizer, criterion,
            train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
            train_epochs=args.epochs, evaluator=eval_plugin,
            device=args.device, plugins=plugins)
    elif args.method == 'replay':
        cl_strategy = ReplayModified(model, optimizer, criterion, mem_size=args.mem_size,
                                     train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                     device=args.device, train_epochs=args.n_epochs,
                                     evaluator=eval_plugin, eval_every=-1, plugins=plugins)
    elif args.method == 'consistency':
        cl_strategy = ConsistencyRegularistaion(model, optimizer, criterion, mem_size=args.mem_size,
                                                regularisation_type=args.regularisation_type, alpha=args.alpha, beta=args.beta,
                                                train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                                device=args.device, train_epochs=args.n_epochs,
                                                evaluator=eval_plugin, eval_every=-1, plugins=plugins)
    elif args.method == 'extended_consistency':
        cl_strategy = ExtendedConsistencyRegularistaion(model, optimizer, criterion, mem_size=args.mem_size,
                                                        regularisation_type=args.regularisation_type, alpha=args.alpha, beta=args.beta,
                                                        train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                                        device=args.device, train_epochs=args.n_epochs,
                                                        evaluator=eval_plugin, eval_every=-1, plugins=plugins)

    return cl_strategy
