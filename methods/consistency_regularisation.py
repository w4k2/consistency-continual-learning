import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator

from .utils import dataset_subset, collate


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
        self.cross_entropy = nn.CrossEntropyLoss()
        self.regularisation = regularisation
        self.datasets_buffer = []
        self.memory_dataloder = []
        self.memory_dataloder_iter = iter(self.memory_dataloder)

    def before_forward(self, strategy, **kwargs):
        if len(self.memory_dataloder) > 0:
            self.current_batch_size = len(strategy.mb_x)

            x_m, self.y_m, _ = self.sample_from_memory(strategy)
            strategy.mbatch[0] = torch.cat((strategy.mb_x, x_m), dim=0)
            self.mem_batch_size = len(x_m)
            x_m, _, self.z_m = self.sample_from_memory(strategy)
            strategy.mbatch[0] = torch.cat((strategy.mb_x, x_m), dim=0)

    def sample_from_memory(self, strategy):
        try:
            x_m, y_m, z_m, _ = next(self.memory_dataloder_iter)
        except StopIteration:
            self.memory_dataloder_iter = iter(self.memory_dataloder)
            x_m, y_m, z_m, _ = next(self.memory_dataloder_iter)
        x_m = x_m.to(strategy.device)
        y_m = y_m.to(strategy.device)
        z_m = z_m.to(strategy.device)
        return x_m, y_m, z_m

    def after_forward(self, strategy, **kwargs):
        if len(self.memory_dataloder) > 0:
            self.y_hat = strategy.mb_output[self.current_batch_size:self.current_batch_size+self.mem_batch_size]
            self.y_hat_dist = strategy.mb_output[self.current_batch_size+self.mem_batch_size:]
            strategy.mb_output = strategy.mb_output[:self.current_batch_size]

    def before_backward(self, strategy, **kwargs):
        if len(self.memory_dataloder) > 0:
            L_er = self.beta * self.cross_entropy(self.y_hat, self.y_m)  # for some reason alpha and beta are switched comparing paper and implementation, here we follow original implementation
            strategy.loss = L_er
            L_cr = self.compute_regularistaion(self.y_hat_dist, self.z_m)
            strategy.loss += self.alpha * L_cr

    def compute_regularistaion(self, z_hat, z):
        if self.regularisation == 'L1':
            reg = torch.pairwise_distance(z_hat, z, p=1).mean()
        elif self.regularisation == 'L2':
            reg = torch.pairwise_distance(z_hat, z, p=2).mean()
        elif self.regularisation == 'Linf':
            reg = torch.pairwise_distance(z_hat, z, p=float('inf')).mean()
        elif self.regularisation == 'MSE':
            reg = F.mse_loss(z_hat, z)
        else:
            raise ValueError(f"Invalid regularistation type, got: {self.regularisation}")
        return reg

    def after_training_exp(self, strategy, num_workers=10, **kwargs):
        new_size = self.mem_size // (len(self.datasets_buffer) + 1)
        rest = self.mem_size % (len(self.datasets_buffer) + 1)
        for i in range(len(self.datasets_buffer)):
            self.datasets_buffer[i] = dataset_subset(self.datasets_buffer[i], new_size)

        dataset = dataset_subset(strategy.experience.dataset, new_size + rest)
        pred = self.get_predictions(dataset, strategy, num_workers)
        self.datasets_buffer.append(MemoryBufferWithPredictions(dataset, pred))

        concat_dataset = ConcatDataset(self.datasets_buffer)
        assert len(concat_dataset) == self.mem_size
        self.memory_dataloder = DataLoader(
            concat_dataset,
            batch_size=strategy.train_mb_size,
            num_workers=num_workers,
            collate_fn=collate,
            shuffle=True
        )
        self.memory_dataloder_iter = iter(self.memory_dataloder)

    def get_predictions(self, dataset, strategy, num_workers):
        batch_size = strategy.eval_mb_size
        if len(dataset) % batch_size == 1:  # make sure that batch norm will work
            batch_size += 1
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
                 alpha=1.0, beta=1.0,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None,
                 evaluator=default_evaluator, eval_every=-1):
        plugin = ConsistencyPlugin(regularisation_type, mem_size, alpha=alpha, beta=beta)
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
