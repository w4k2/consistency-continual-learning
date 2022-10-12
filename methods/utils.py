import torch


def dataset_subset(dataset, new_size):
    indices = torch.randperm(len(dataset))[:new_size]
    subset = None
    if type(dataset) == torch.utils.data.Subset:
        dataset.indices = [dataset.indices[i] for i in indices]
        subset = dataset
    else:
        subset = torch.utils.data.Subset(dataset, indices)
    return subset


def collate(mbatches):
    batch = []
    for i in range(len(mbatches[0])):
        elems = list()
        for sample in mbatches:
            elem = sample[i]
            if type(elem) != torch.Tensor and type(elem) == int:
                elem = torch.Tensor([elem]).to(torch.long)
            elems.append(elem)
        t = torch.stack(elems, dim=0)
        if i >= 1:
            t.squeeze_()
        batch.append(t)
    return batch


class MutliDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.len = sum(len(dataset) for dataset in datasets)

    def __getitem__(self, index):
        dataset_index, sample_index = index
        sample = self.datasets[dataset_index][sample_index]
        return sample

    def __len__(self):
        return self.len


class RehersalSampler():
    def __init__(self, dataset_sizes, dataset_samplers, batch_size, drop_last, oversample_small_tasks=False):
        self.dataset_sizes = dataset_sizes
        self.dataset_active = [True for _ in dataset_sizes]
        self.dataset_samplers = dataset_samplers
        self.dataset_samplers_iter = [iter(sampler) for sampler in dataset_samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.oversample_small_tasks = oversample_small_tasks

        self.len = None
        if self.drop_last:
            self.len = sum(self.dataset_sizes) // self.batch_size
        else:
            self.len = (sum(self.dataset_sizes) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.dataset_active = [True for _ in self.dataset_sizes]
        self.dataset_samplers_iter = [iter(sampler) for sampler in self.dataset_samplers]

        batch = []
        num_generated = 0
        i = -1
        while num_generated < len(self):
            i += 1
            i = i % len(self.dataset_sizes)
            if not any(self.dataset_active):
                break
            if not self.dataset_active[i]:
                continue
            try:
                j = next(self.dataset_samplers_iter[i])
            except StopIteration:
                if self.oversample_small_tasks:
                    self.dataset_samplers_iter[i] = iter(self.dataset_samplers[i])
                    j = next(self.dataset_samplers_iter[i])
                else:
                    self.dataset_active[i] = False
                    continue
            idx = (i, j)
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                num_generated += 1
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return self.len
