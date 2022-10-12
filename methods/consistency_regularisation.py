import torch
import torch.nn as nn

from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator

from .utils import dataset_subset, collate, RehersalSampler, MutliDataset


class MemoryBufferWithPredictions:
    def __init__(self, dataset, predictions) -> None:
        assert len(dataset) == len(predictions)
        self.dataset = dataset
        self.preditctions = predictions

    def __getitem__(self, i):
        x, y, t = self.dataset[i]
        y_hat = self.preditctions[i]
        sample_with_pred = (x, y, y_hat, t)
        return sample_with_pred

    def __len__(self) -> int:
        return len(self.dataset)


class ConsistencyPlugin(SupervisedPlugin):
    """ Consistency regularistation based on https://arxiv.org/pdf/2207.04998.pdf

    args:
        regularisation (str) - type of regularisation
    """

    def __init__(self, regularisation, mem_size: int = 200, alpha=1.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_size = mem_size
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        self.regularisation = regularisation
        self.datasets_buffer = []
        self.memory_dataloder = []

    def before_backward(self, strategy, **kwargs):
        for x_m, y_m, z_m, _ in self.memory_dataloder:
            B = len(x_m)
            x_m = x_m.to(strategy.device)
            y_m = y_m.to(strategy.device)
            z_m = z_m.to(strategy.device)
            y_hat = strategy.model(x_m)
            L_er = self.alpha / B * self.cross_entropy(y_hat, y_m)
            strategy.loss += L_er
            L_cr = self.compute_regularistaion(y_hat, z_m)
            strategy.loss += self.beta * L_cr

    def compute_regularistaion(self, z_hat, z):
        if self.regularisation == 'L1':
            reg = torch.norm(z_hat - z, dim=1, p=1)
            reg = reg.mean()
        else:
            raise ValueError(f"Invalid regularistation type, got: {self.regularisation}")
        return reg

    def after_training_exp(self, strategy, num_workers=10, **kwargs):
        pred = self.get_predictions(strategy, num_workers)
        dataset = MemoryBufferWithPredictions(strategy.experience.dataset, pred)
        self.datasets_buffer.append(dataset)

        new_size = self.mem_size // len(self.datasets_buffer)
        for i in range(len(self.datasets_buffer)):
            self.datasets_buffer[i] = dataset_subset(self.datasets_buffer[i], new_size)

        dataset_list = list(self.datasets_buffer)
        concat_dataset = MutliDataset(dataset_list)
        sampler = RehersalSampler(dataset_sizes=[len(dataset) for dataset in dataset_list],
                                  dataset_samplers=[RandomSampler(dataset) for dataset in dataset_list],
                                  batch_size=strategy.train_mb_size,
                                  drop_last=False)
        self.memory_dataloder = DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate
        )

    def get_predictions(self, strategy, num_workers):
        dataset = strategy.adapted_dataset
        dataloader = DataLoader(dataset, batch_size=strategy.eval_mb_size, shuffle=False, num_workers=num_workers)
        predictions = []
        with torch.no_grad():
            for x, _, _ in dataloader:
                x = x.to(strategy.device)
                y_hat = strategy.model(x)
                y_hat = y_hat.to("cpu")
                predictions.append(y_hat)
        predictions = torch.cat(predictions, dim=0)
        return predictions


class ConsistencyRegularistaion(SupervisedTemplate):
    """ Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer, criterion,
                 mem_size: int = 200, regularisation_type='L1',
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None,
                 evaluator=default_evaluator, eval_every=-1):
        plugin = ConsistencyPlugin(regularisation_type, mem_size)
        if plugins is None:
            plugins = [plugin]
        else:
            plugins.append(plugin)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
