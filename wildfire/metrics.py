import textwrap
import pandas as pd


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
