from __future__ import annotations
import numpy as np
import torch


class Numpy(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, data) -> np.ndarray:
        if isinstance(data, cls):
            return data.view(np.ndarray)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.numpy()
        else:
            return np.array(data)

    @classmethod
    def ndim(cls, ndim) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritNumpy

    @classmethod
    def dims(cls, dims) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.ndim != len(dims):
                    raise ValueError(
                        f"Unexpected number of dims {data.ndim} for {dims}"
                    )
                return data

        return InheritNumpy

    @classmethod
    def shape(cls, *sizes) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                for data_size, size in zip(data.shape, sizes):
                    if size != -1 and data_size != size:
                        raise ValueError(f"Expected size {size}, got {data_size}")
                return data

        return InheritNumpy

    @classmethod
    def between(cls, ge, le) -> Numpy:
        class InheritNumpy(cls):
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

        return InheritNumpy

    @classmethod
    def ge(cls, ge) -> Numpy:
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
    def le(cls, le) -> Numpy:
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
    def gt(cls, gt) -> Numpy:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if data.min() <= gt:
                    raise ValueError(f"Expected greater than {gt}, got {data.min()}")

        return InheritTensor

    @classmethod
    def lt(cls, lt) -> Numpy:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if data.max() >= lt:
                    raise ValueError(f"Expected less than {lt}, got {data.max()}")
                return data

        return InheritTensor

    @classmethod
    def ne(cls, ne) -> Numpy:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)

                if (data == ne).any():
                    raise ValueError(f"Unexpected value {ne}")
                return data

        return InheritTensor

    @classmethod
    def dtype(cls, dtype) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                new_data = data.astype(dtype)
                if not np.allclose(data, new_data, equal_nan=True):
                    raise ValueError(f"Was unable to cast from {data.dtype} to {dtype}")
                return new_data

        return InheritNumpy

    @classmethod
    def float(cls) -> Numpy:
        return cls.dtype(np.float32)

    @classmethod
    def float32(cls) -> Numpy:
        return cls.dtype(np.float32)

    @classmethod
    def half(cls) -> Numpy:
        return cls.dtype(np.float16)

    @classmethod
    def float16(cls):
        return cls.dtype(np.float16)

    @classmethod
    def double(cls) -> Numpy:
        return cls.dtype(np.float64)

    @classmethod
    def float64(cls) -> Numpy:
        return cls.dtype(np.float64)

    @classmethod
    def int(cls) -> Numpy:
        return cls.dtype(np.int32)

    @classmethod
    def int32(cls) -> Numpy:
        return cls.dtype(np.int32)

    @classmethod
    def long(cls) -> Numpy:
        return cls.dtype(np.int64)

    @classmethod
    def int64(cls) -> Numpy:
        return cls.dtype(np.int64)

    @classmethod
    def short(cls) -> Numpy:
        return cls.dtype(np.int16)

    @classmethod
    def int16(cls) -> Numpy:
        return cls.dtype(np.int16)

    @classmethod
    def byte(cls) -> Numpy:
        return cls.dtype(np.uint8)

    @classmethod
    def uint8(cls) -> Numpy:
        return cls.dtype(np.uint8)

    @classmethod
    def bool(cls) -> Numpy:
        return cls.dtype(bool)


def test_base_model():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        images: Numpy.dims("NCHW")

    Test(images=np.ones((10, 3, 32, 32)))

    with raises(ValueError):
        Test(images=np.ones((10, 3, 32)))


def test_validate():
    from pytest import raises

    with raises(ValueError):
        Numpy.ndim(4).validate(np.ones((3, 4, 5)))


def test_conversion():
    from pydantic import BaseModel
    import torch

    class Test(BaseModel):
        numbers: Numpy.dims("N")

    Test(numbers=[1.1, 2.1, 3.1])
    Test(numbers=torch.tensor([1.1, 2.1, 3.1]))


def test_chaining():
    from pytest import raises

    with raises(ValueError):
        Numpy.ndim(4).dims("NCH").validate(np.ones((3, 4, 5)))

    with raises(ValueError):
        Numpy.dims("NCH").ndim(4).validate(np.ones((3, 4, 5)))


def test_dtype():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Numpy.uint8()

    Test(numbers=[1, 2, 3])

    with raises(ValueError):
        Test(numbers=[1.5, 2.2, 3.2])

    class TestBool(BaseModel):
        flags: Numpy.bool()

    TestBool(flags=[True, False, True])

    with raises(ValueError):
        TestBool(numbers=[1.5, 2.2, 3.2])


def test_from_torch():
    import torch
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Numpy

    numbers = torch.tensor([1, 2, 3])
    numpy_numbers = Test(numbers=numbers).numbers

    assert type(numpy_numbers) == np.ndarray
    assert np.allclose(torch.from_numpy(numpy_numbers), numbers)


def test_between():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Numpy.between(1, 3.5)

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[-1.5, 2.2, 3.2])


def test_gt():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Numpy.gt(1)

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[1, 2.2, 3.2])
