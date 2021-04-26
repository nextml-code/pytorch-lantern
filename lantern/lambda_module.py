from torch import nn


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def test_lambda_module():
    import torch

    model = nn.Sequential(Lambda(lambda x: x * 2))
    assert (torch.tensor([4, 8]) == model(torch.tensor([2, 4]))).all()
