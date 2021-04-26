import numpy as np
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
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
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
    def dims(cls, dims):
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

    @classmethod
    def device(cls, device):
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                return super().validate(data).to(device)

        return InheritTensor

    @classmethod
    def cpu(cls):
        return cls.device(torch.device("cpu"))

    @classmethod
    def cuda(cls):
        return cls.device(torch.device("cuda"))

    @classmethod
    def dtype(cls, dtype):
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                new_data = data.type(dtype)
                if not torch.allclose(data.float(), new_data.float(), equal_nan=True):
                    raise ValueError(f"Was unable to cast from {data.dtype} to {dtype}")
                return new_data

        return InheritTensor

    @classmethod
    def float(cls):
        return cls.dtype(torch.float32)

    @classmethod
    def half(cls):
        return cls.dtype(torch.float16)

    @classmethod
    def double(cls):
        return cls.dtype(torch.float64)

    @classmethod
    def int(cls):
        return cls.dtype(torch.int32)

    @classmethod
    def long(cls):
        return cls.dtype(torch.int64)

    @classmethod
    def short(cls):
        return cls.dtype(torch.int16)

    @classmethod
    def uint8(cls):
        return cls.dtype(torch.uint8)


def test_base_model():
    from pydantic import BaseModel

    class Test(BaseModel):
        tensor: Tensor.dims("NCHW")

    Test(tensor=torch.ones(10, 3, 32, 32))


def test_validate():
    from pytest import raises

    with raises(ValueError):
        Tensor.ndim(4).validate(torch.ones(3, 4, 5))


def test_conversion():
    from pydantic import BaseModel
    import numpy as np

    class Test(BaseModel):
        numbers: Tensor.dims("N")
        numbers2: Tensor.dims("N")

    Test(
        numbers=[1.1, 2.1, 3.1],
        numbers2=np.array([1.1, 2.1, 3.1]),
    )


def test_chaining():
    from pytest import raises

    with raises(ValueError):
        Tensor.ndim(4).dims("NCH").validate(torch.ones(3, 4, 5))

    with raises(ValueError):
        Tensor.dims("NCH").ndim(4).validate(torch.ones(3, 4, 5))


def test_dtype():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Tensor.uint8()

    Test(numbers=[1, 2, 3])

    with raises(ValueError):
        Test(numbers=[1.5, 2.2, 3.2])


def test_device():
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Tensor.float().cpu()

    Test(numbers=[1, 2, 3])


def test_from_numpy():
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Tensor

    numbers = np.array([1, 2, 3])
    torch_numbers = Test(numbers=numbers).numbers

    assert type(torch_numbers) == torch.Tensor
    assert np.allclose(torch_numbers.numpy(), numbers)
