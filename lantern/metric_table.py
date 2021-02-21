import textwrap
import pandas as pd
from lantern import FunctionalBase, Metric
from typing import Dict


class MetricTable(FunctionalBase):
    name: str
    metrics: Dict[str, Metric]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name, metrics):
        super().__init__(
            name=name,
            metrics=metrics,
        )

    def compute(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

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

    def __str__(self):
        return self.table()
