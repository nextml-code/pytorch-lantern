import textwrap
from typing import Any, Dict, Union

import pandas as pd

from lantern import FunctionalBase


class MetricTable(FunctionalBase, arbitrary_types_allowed=True):
    name: str
    metrics: Dict[str, Any]

    def __init__(self, name, metrics):
        super().__init__(
            name=name,
            metrics=metrics,
        )

    def compute(self):
        log_dict = dict()
        for metric_name, metric in self.metrics.items():
            metric_value = metric.compute()
            if isinstance(metric_value, dict):
                log_dict.update(**metric_value)
            else:
                log_dict[metric_name] = metric_value
        return log_dict

    def table(self):
        return "\n".join(
            [
                f"{self.name}:",
                textwrap.indent(
                    (
                        pd.Series(self.compute(), dtype=object).to_string(
                            name=True, dtype=False, index=True
                        )
                    ),
                    prefix="  ",
                ),
            ]
        )

    def __str__(self):
        return self.table()
