import textwrap
import pandas as pd


class MetricsView:
    def __init__(self, metrics_container, names):
        self.metrics_container = metrics_container
        self.names = names

    @property
    def name(self):
        return self.metrics_container.name

    def compute(self):
        return {
            name: self.metrics_container.metrics[name].compute() for name in self.names
        }


class Metrics:
    def __init__(self, name, tensorboard_logger, metrics, n_logs=0):
        self.name = name
        self.tensorboard_logger = tensorboard_logger
        self.metrics = metrics
        self.n_logs = n_logs

    def __getitem__(self, name_or_names):
        if type(name_or_names) == str:
            return self.metrics[name_or_names]
        else:
            return MetricsView(self, name_or_names)

    def update_(self, *args, **kwargs):
        self.metrics = {
            name: metric.reduce(*args, **kwargs)
            for name, metric in self.metrics.items()
        }
        return self

    def compute(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def log_(self, step=None):
        self.n_logs += 1
        if step is None:
            step = self.n_logs

        for name, metric in self.metrics.items():
            self.tensorboard_logger.add_scalar(
                f"{self.name}/{name}",
                metric.compute(),
                step,
            )
        return self

    def table(self):
        return "\n".join(
            [
                f"{self.name}:",
                textwrap.indent(
                    (
                        pd.Series(self.compute()).to_string(
                            name=True, dtype=False, index=True
                        )
                    ),
                    prefix="  ",
                ),
            ]
        )

    def print(self):
        print(self.table())
        return self
