from tqdm import tqdm
from typing import Dict, Optional
from lantern.metric import Metric


def ProgressBar(data_loader, name, metrics: Optional[Dict[str, Metric]] = None):
    """Simple progress bar with metrics"""
    if metrics is None:
        for item in tqdm(data_loader, desc=name, leave=False):
            yield item
    else:
        with tqdm(data_loader, desc=name, leave=False) as tqdm_:
            for item in tqdm_:
                yield item

                log_dict = dict()
                for metric_name, metric in metrics.items():
                    metric_value = metric.compute()
                    if isinstance(metric_value, dict):
                        log_dict.update(**metric_value)
                    else:
                        log_dict[metric_name] = metric_value
                tqdm_.set_postfix(log_dict)
