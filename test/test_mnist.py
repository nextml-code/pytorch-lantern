from functools import partial
from time import sleep
from tqdm import tqdm
import torch
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import datastream

import lantern
from lantern import ModuleCompose


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
        datasets.MNIST('data', train=True, transform=transform, download=True)
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
    early_stopping = lantern.EarlyStopping()
    gradient_metrics = lantern.Metrics(
        name='gradient',
        tensorboard_logger=tensorboard_logger,
        metrics=dict(
            loss=lantern.MapMetric(lambda examples, predictions, loss: loss),
        ),
    )

    for epoch in lantern.Epochs(2):

        with lantern.module_train(model):
            for examples, targets in lantern.ProgressBar(
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
                sleep(0.5)
        gradient_metrics.print()

        evaluate_metrics = {
            name: lantern.Metrics(
                name=name,
                tensorboard_logger=tensorboard_logger,
                metrics=dict(
                    loss=lantern.MapMetric(
                        lambda examples, predictions, loss: loss
                    ),
                ),
            )
            for name in evaluate_data_loaders.keys()
        }

        with lantern.module_eval(model), torch.no_grad():
            for name, data_loader in evaluate_data_loaders.items():
                for examples, targets in tqdm(
                    data_loader, desc=name, leave=False
                ):
                    predictions = model(examples)
                    loss = F.nll_loss(predictions, targets)
                    evaluate_metrics[name].update_(
                        examples, predictions, loss
                    )
                evaluate_metrics[name].log_().print()

        early_stopping = early_stopping.score(
            -evaluate_metrics['evaluate_early_stopping']['loss'].compute()
        )
        if early_stopping.scores_since_improvement == 0:
            torch.save(model.state_dict(), 'model.pt')
            torch.save(optimizer.state_dict(), 'optimizer.pt')
        elif early_stopping.scores_since_improvement > 5:
            break
        early_stopping.print()


if __name__ == '__main__':
    test_mnist()
