from pydantic import BaseModel


class FunctionalBase(BaseModel):
    def map(self, fn, *args, **kwargs):
        return fn(self, *args, **kwargs)

    def replace(self, **kwargs):
        new_dict = self.model_dump()
        new_dict.update(**kwargs)
        return type(self)(**new_dict)

    @classmethod
    def setattr(cls, name, value):
        if hasattr(cls, name):
            raise ValueError(f"Attribute {name} already exists")
        setattr(cls, name, value)
        return value

    @classmethod
    def method(cls, fn):
        return cls.setattr(fn.__name__, fn)

    @classmethod
    def property(cls, fn):
        return cls.setattr(fn.__name__, property(fn))

    @classmethod
    def staticmethod(cls, fn):
        return cls.setattr(fn.__name__, staticmethod(fn))

    @classmethod
    def classmethod(cls, fn):
        return cls.setattr(fn.__name__, classmethod(fn))


def test_replace_same_device():
    import torch

    from .tensor import Tensor

    class A(FunctionalBase):
        x: Tensor
        y: int

    a = A(x=torch.tensor([1, 2, 3]).to("meta"), y=2)
    b = a.replace(y=2)

    assert b.x.device == a.x.device
