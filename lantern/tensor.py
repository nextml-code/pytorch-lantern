import torch


class Tensor(torch.Tensor):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, data):
        if isinstance(data, cls):
            return torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            return torch.as_tensor(data)

    @classmethod
    def ndim(cls, ndim):
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritTensor

    @classmethod
    def short(cls, dims):
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.ndim != len(dims):
                    raise ValueError(
                        f"Unexpected number of dims {data.ndim} for {dims}"
                    )
                return data

        return InheritTensor

    @classmethod
    def shape(cls, *sizes):
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                for data_size, size in zip(data.shape, sizes):
                    if size != -1 and data_size != size:
                        raise ValueError(f"Expected size {size}, got {data_size}")
                return data

        return InheritTensor

    @classmethod
    def between(cls, geq, leq):
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                data_min = data.min()
                if data_min < geq:
                    raise ValueError(f"Expected min value {geq}, got {data_min}")

                data_max = data.min()
                if data_max > leq:
                    raise ValueError(f"Expected max value {leq}, got {data_max}")
                return data

        return InheritTensor


def test_base_model():
    from pydantic import BaseModel

    class Test(BaseModel):
        tensor: Tensor.short("nchw")

    Test(tensor=torch.ones(10, 3, 32, 32))


def test_validate():
    from pytest import raises

    with raises(ValueError):
        Tensor.ndim(4).validate(torch.ones(3, 4, 5))


def test_conversion():
    from pydantic import BaseModel
    import numpy as np

    class Test(BaseModel):
        numbers: Tensor.short("N")
        numbers2: Tensor.short("N")

    Test(
        numbers=[1.1, 2.1, 3.1],
        numbers2=np.array([1.1, 2.1, 3.1]),
    )


def test_chaining():
    from pytest import raises

    with raises(ValueError):
        Tensor.ndim(4).short("NCH").validate(torch.ones(3, 4, 5))

    with raises(ValueError):
        Tensor.short("NCH").ndim(4).validate(torch.ones(3, 4, 5))
