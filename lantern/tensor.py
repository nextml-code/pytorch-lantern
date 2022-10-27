from __future__ import annotations
import numpy as np
import torch


class Tensor(torch.Tensor):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, data) -> torch.Tensor:
        if isinstance(data, cls):
            return torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            return torch.as_tensor(data)

    @classmethod
    def ndim(cls, ndim) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritTensor

    @classmethod
    def dims(cls, dims) -> Tensor:
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
    def shape(cls, *sizes) -> Tensor:
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
    def between(cls, ge, le) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.min() < ge:
                    raise ValueError(
                        f"Expected greater than or equal to {ge}, got {data.min()}"
                    )

                if data.max() > le:
                    raise ValueError(
                        f"Expected less than or equal to {le}, got {data.max()}"
                    )
                return data

        return InheritTensor

    @classmethod
    def ge(cls, ge) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.min() < ge:
                    raise ValueError(
                        f"Expected greater than or equal to {ge}, got {data.min()}"
                    )

        return InheritTensor

    @classmethod
    def le(cls, le) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if data.max() > le:
                    raise ValueError(
                        f"Expected less than or equal to {le}, got {data.max()}"
                    )
                return data

        return InheritTensor

    @classmethod
    def gt(cls, gt) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if data.min() <= gt:
                    raise ValueError(f"Expected greater than {gt}, got {data.min()}")

        return InheritTensor

    @classmethod
    def lt(cls, lt) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if data.max() >= lt:
                    raise ValueError(f"Expected less than {lt}, got {data.max()}")
                return data

        return InheritTensor

    @classmethod
    def ne(cls, ne) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if (data == ne).any():
                    raise ValueError(f"Unexpected value {ne}")
                return data

        return InheritTensor

    @classmethod
    def device(cls, device) -> Tensor:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                return super().validate(data).to(device)

        return InheritTensor

    @classmethod
    def cpu(cls) -> Tensor:
        return cls.device(torch.device("cpu"))

    @classmethod
    def cuda(cls) -> Tensor:
        return cls.device(torch.device("cuda"))

    @classmethod
    def dtype(cls, dtype) -> Tensor:
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
    def float(cls) -> Tensor:
        return cls.dtype(torch.float32)

    @classmethod
    def float32(cls) -> Tensor:
        return cls.dtype(torch.float32)

    @classmethod
    def half(cls) -> Tensor:
        return cls.dtype(torch.float16)

    @classmethod
    def float16(cls):
        return cls.dtype(torch.float16)

    @classmethod
    def double(cls) -> Tensor:
        return cls.dtype(torch.float64)

    @classmethod
    def float64(cls) -> Tensor:
        return cls.dtype(torch.float64)

    @classmethod
    def int(cls) -> Tensor:
        return cls.dtype(torch.int32)

    @classmethod
    def int32(cls) -> Tensor:
        return cls.dtype(torch.int32)

    @classmethod
    def long(cls) -> Tensor:
        return cls.dtype(torch.int64)

    @classmethod
    def int64(cls) -> Tensor:
        return cls.dtype(torch.int64)

    @classmethod
    def short(cls) -> Tensor:
        return cls.dtype(torch.int16)

    @classmethod
    def int16(cls) -> Tensor:
        return cls.dtype(torch.int16)

    @classmethod
    def byte(cls) -> Tensor:
        return cls.dtype(torch.uint8)

    @classmethod
    def uint8(cls) -> Tensor:
        return cls.dtype(torch.uint8)

    @classmethod
    def bool(cls) -> Tensor:
        return cls.dtype(torch.bool)


def test_base_model():
    from pydantic import BaseModel

    class Test(BaseModel):
        tensor: Tensor.dims("NCHW").float()

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

    class TestBool(BaseModel):
        flags: Tensor.bool()

    TestBool(flags=[True, False, True])

    with raises(ValueError):
        TestBool(numbers=[1.5, 2.2, 3.2])


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


def test_ge():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Tensor.ge(0)

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[-1.5, 2.2, 3.2])


def test_ne():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Tensor.ne(1)

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[1, 2.2, 3.2])
