import torch


class Tensor(torch.Tensor):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, data, values, field, config):
        return data

    @classmethod
    def ndim(cls, ndim):
        class InheritTensor(Tensor):
            @classmethod
            def validate(cls, data):
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritTensor

    @classmethod
    def short(cls, dims):
        class InheritTensor(Tensor):
            @classmethod
            def validate(cls, data):
                if data.ndim != len(dims):
                    raise ValueError(
                        f"Unexpected number of dims {data.ndim} for {dims}"
                    )
                return data

        return InheritTensor

    @classmethod
    def shape(cls, *sizes):
        class InheritTensor(Tensor):
            @classmethod
            def validate(cls, data):
                for data_size, size in zip(data.shape, sizes):
                    if size != -1 and data_size != size:
                        raise ValueError(f"Expected size {size}, got {data_size}")
                return data

        return InheritTensor

    @classmethod
    def between(cls, geq, leq):
        class InheritTensor(Tensor):
            @classmethod
            def validate(cls, data):
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
