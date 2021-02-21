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
                tqdm_.set_postfix(
                    {name: metric.compute() for name, metric in metrics.items()}
                )
