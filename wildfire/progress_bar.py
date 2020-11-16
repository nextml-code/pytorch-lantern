from tqdm import tqdm


def ProgressBar(data_loader, metrics=None):
    if metrics is None:
        return tqdm(data_loader, leave=False)
    else:
        with tqdm(data_loader, desc=metrics.name, leave=False) as tqdm_:
            for item in tqdm_:
                yield item
                tqdm_.set_postfix(metrics.compute())
