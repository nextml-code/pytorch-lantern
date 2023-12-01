from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated


def validate_from_list(values: List) -> np.ndarray:
    return np.array(values)


def validate_from_torch(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


class Numpy:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        from_list_schema = core_schema.chain_schema(
            [
                core_schema.list_schema(),
                core_schema.no_info_plain_validator_function(validate_from_list),
            ]
        )

        from_torch_schema = core_schema.chain_schema(
            [
                core_schema.is_instance_schema(torch.Tensor),
                core_schema.no_info_plain_validator_function(validate_from_torch),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.is_instance_schema(np.ndarray),
                            from_list_schema,
                            from_torch_schema,
                        ]
                    ),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.list_schema())

    @classmethod
    def validate(cls, data, config=None, field=None):
        return data

    @classmethod
    def ndim(cls, ndim) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data, config=None, field=None):
                data = super().validate(data)
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritNumpy

    @classmethod
    def dims(cls, dims) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data, config=None, field=None):
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
            def validate(cls, data, config=None, field=None):
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
            def validate(cls, data, config=None, field=None):
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
            def validate(cls, data, config=None, field=None):
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
            def validate(cls, data, config=None, field=None):
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
            def validate(cls, data, config=None, field=None):
                data = super().validate(data)

                if data.min() <= gt:
                    raise ValueError(f"Expected greater than {gt}, got {data.min()}")

        return InheritTensor

    @classmethod
    def lt(cls, lt) -> Numpy:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data, config=None, field=None):
                data = super().validate(data)

                if data.max() >= lt:
                    raise ValueError(f"Expected less than {lt}, got {data.max()}")
                return data

        return InheritTensor

    @classmethod
    def ne(cls, ne) -> Numpy:
        class InheritTensor(cls):
            @classmethod
            def validate(cls, data, config=None, field=None):
                data = super().validate(data)

                if (data == ne).any():
                    raise ValueError(f"Unexpected value {ne}")
                return data

        return InheritTensor

    @classmethod
    def dtype(cls, dtype) -> Numpy:
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data, config=None, field=None):
                data = super().validate(data)
                if data.dtype == dtype:
                    return data
                else:
                    new_data = data.astype(dtype)
                    if not np.allclose(data, new_data, equal_nan=True):
                        raise ValueError(
                            f"Was unable to cast from {data.dtype} to {dtype}"
                        )
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
        images: Annotated[np.ndarray, Numpy.dims("NCHW")]

    with raises(ValueError):
        Test(images=np.ones((10, 3, 32)))


def test_validate():
    from pytest import raises

    with raises(ValueError):
        Numpy.ndim(4).validate(np.ones((3, 4, 5)))


def test_conversion():
    import torch
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Annotated[np.ndarray, Numpy.dims("N")]

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
        numbers: Annotated[np.ndarray, Numpy.uint8()]

    Test(numbers=[1, 2, 3])

    with raises(ValueError):
        Test(numbers=[1.5, 2.2, 3.2])

    class TestBool(BaseModel):
        flags: Annotated[np.ndarray, Numpy.bool()]

    TestBool(flags=[True, False, True])

    with raises(ValueError):
        TestBool(numbers=[1.5, 2.2, 3.2])


def test_from_torch():
    import torch
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Annotated[np.ndarray, Numpy]

    numbers = torch.tensor([1, 2, 3])
    numpy_numbers = Test(numbers=numbers).numbers

    assert type(numpy_numbers) == np.ndarray
    assert np.allclose(torch.from_numpy(numpy_numbers), numbers)


def test_between():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Annotated[np.ndarray, Numpy.between(1, 3.5)]

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[-1.5, 2.2, 3.2])


def test_gt():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Annotated[np.ndarray, Numpy.gt(1)]

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[1, 2.2, 3.2])
