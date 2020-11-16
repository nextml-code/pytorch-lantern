from functools import partial
import textwrap
from tqdm import tqdm
import pandas as pd
import torch
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import datastream

import wildfire
from wildfire import ModuleCompose


def test_mnist():

    device = torch.device("cpu")
    model = ModuleCompose(
        nn.Conv2d(1, 4, 3, 1),
        partial(F.max_pool2d, kernel_size=2),
        partial(torch.flatten, start_dim=1),
        F.relu,
        nn.Linear(676, 10),
        partial(F.log_softmax, dim=1),
    ).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=5e-3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    gradient_dataset = datastream.Dataset.from_subscriptable(
        datasets.MNIST('data', train=True, download=True, transform=transform)
    )
    early_stopping_dataset = datastream.Dataset.from_subscriptable(
        datasets.MNIST('data', train=False, transform=transform)
    )

    gradient_data_loader = (
        datastream.Datastream(gradient_dataset)
        .take(16 * 4)
        .data_loader(batch_size=16)
    )
    early_stopping_data_loader = (
        datastream.Datastream(early_stopping_dataset)
        .take(16 * 4)
        .data_loader(batch_size=16)
    )
    evaluate_data_loaders = dict(
        evaluate_gradient=gradient_data_loader,
        evaluate_early_stopping=early_stopping_data_loader,
    )

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    early_stopping = wildfire.EarlyStopping()
    # gradient_metrics = metrics.gradient_metrics()

    class Metrics:
        def __init__(self, name, tensorboard_logger, **metrics):
            self.name = name
            self.tensorboard_logger = tensorboard_logger
            self.metrics = metrics

        def __getitem__(self, names):
            return Metrics(
                name=self.name,
                tensorboard_logger=self.tensorboard_logger,
                **{name: self.metrics[name] for name in names},
            )

        def update_(self, *args, **kwargs):
            self.metrics = {
                name: metric.reduce(*args, **kwargs)
                for name, metric in self.metrics.items()
            }
            return self

        def compute(self):
            return {
                name: metric.compute()
                for name, metric in self.metrics.items()
            }

        def log_(self):
            # TODO
            return self

        def table(self):
            return '\n'.join([
                f'{self.name}:',
                textwrap.indent(
                    (
                        pd.Series(self.compute())
                        .to_string(name=True, dtype=False, index=True)
                    ),
                    prefix='  ',
                ),
            ])

        def print(self):
            print(self.table())
            return self


    gradient_metrics = Metrics(
        name='gradient',
        loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
        tensorboard_logger=tensorboard_logger,
    )
    from time import sleep

    for epoch in wildfire.Epochs(2):

        with wildfire.module_train(model):
            for examples, targets in wildfire.ProgressBar(
                gradient_data_loader, metrics=gradient_metrics[['loss']]
            ):
                predictions = model(examples)
                loss = F.nll_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                (
                    gradient_metrics
                    .update_(examples, predictions.detach(), loss.detach())
                    .log_()
                )
                sleep(1)

        gradient_metrics.print()
        

        with wildfire.module_eval(model), torch.no_grad():
            for name, data_loader in evaluate_data_loaders.items():
                evaluate_metrics = Metrics(
                    name=name,
                    loss=wildfire.MapMetric(lambda examples, predictions, loss: loss),
                    tensorboard_logger=tensorboard_logger,
                )

                for examples, targets in tqdm(data_loader, desc=name, leave=False):
                    predictions = model(examples)
                    loss = F.nll_loss(predictions, targets)
                    sleep(0.5)

                    evaluate_metrics.update_(
                        examples, predictions, loss
                    )
                evaluate_metrics.log_().print()

        # early_stopping = early_stopping.score(tensorboard_logger)
        # if early_stopping.scores_since_improvement == 0:
        #     torch.save(train_state, 'model_checkpoint.pt')
        # elif early_stopping.scores_since_improvement > patience:
        #     break


if __name__ == '__main__':
    test_mnist()
