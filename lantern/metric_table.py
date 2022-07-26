import textwrap
import pandas as pd
from lantern import FunctionalBase
from typing import Dict, Union, Any

# from wire_damage.tools import MapMetric, ReduceMetric, AggregateMetric


class MetricTable(FunctionalBase):
    name: str
    metrics: Dict[str, Any]
    # metrics: Dict[str, Union[MapMetric, ReduceMetric, AggregateMetric]]

    class Config:
        arbitrary_types_allowed = True

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
