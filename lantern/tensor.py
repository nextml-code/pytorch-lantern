from __future__ import annotations

from typing import Any, List, Type, TypeVar

import numpy as np
import torch
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated

T = TypeVar("T", bound="Tensor")


def validate_from_list(values: List) -> torch.Tensor:
    """Convert a list to a PyTorch tensor."""
    return torch.tensor(values)


def validate_from_numpy(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    return torch.from_numpy(array)


class Tensor:
    """
    A class for creating type hints and validators for PyTorch tensors.

    This class can be used with Pydantic to define and validate tensor fields
    in data models. It provides various methods to specify tensor properties
    such as dimensions, shape, data type, and value ranges.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Generate the Pydantic core schema for tensor validation."""
        from_list_schema = core_schema.chain_schema(
            [
                core_schema.list_schema(),
                core_schema.no_info_plain_validator_function(validate_from_list),
            ]
        )

        from_numpy_schema = core_schema.chain_schema(
            [
                core_schema.is_instance_schema(np.ndarray),
                core_schema.no_info_plain_validator_function(validate_from_numpy),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.is_instance_schema(torch.Tensor),
                            from_list_schema,
                            from_numpy_schema,
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
        """Generate the JSON schema for tensor validation."""
        return handler(core_schema.list_schema())

    @classmethod
    def validate(cls, data: Any, config: Any = None, field: Any = None) -> torch.Tensor:
        """Base validation method for tensor data."""
        return data

    @classmethod
    def ndim(cls: Type[T], ndim: int) -> Type[T]:
        """Specify the number of dimensions for the tensor."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritTensor

    @classmethod
    def dims(cls: Type[T], dims: str) -> Type[T]:
        """Specify the dimension names for the tensor."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)
                if data.ndim != len(dims):
                    raise ValueError(
                        f"Unexpected number of dims {data.ndim} for {dims}"
                    )
                return data

        return InheritTensor

    @classmethod
    def shape(cls: Type[T], *sizes: int) -> Type[T]:
        """Specify the shape of the tensor."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)
                for data_size, size in zip(data.shape, sizes):
                    if size != -1 and data_size != size:
                        raise ValueError(f"Expected size {size}, got {data_size}")
                return data

        return InheritTensor

    @classmethod
    def between(cls: Type[T], ge: float, le: float) -> Type[T]:
        """Specify a range for tensor values."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
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
    def ge(cls: Type[T], ge: float) -> Type[T]:
        """Specify a minimum value for tensor elements."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)
                if data.min() < ge:
                    raise ValueError(
                        f"Expected greater than or equal to {ge}, got {data.min()}"
                    )
                return data

        return InheritTensor

    @classmethod
    def le(cls: Type[T], le: float) -> Type[T]:
        """Specify a maximum value for tensor elements."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)

                if data.max() > le:
                    raise ValueError(
                        f"Expected less than or equal to {le}, got {data.max()}"
                    )
                return data

        return InheritTensor

    @classmethod
    def gt(cls: Type[T], gt: float) -> Type[T]:
        """Specify a strict minimum value for tensor elements."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)

                if data.min() <= gt:
                    raise ValueError(f"Expected greater than {gt}, got {data.min()}")
                return data

        return InheritTensor

    @classmethod
    def lt(cls: Type[T], lt: float) -> Type[T]:
        """Specify a strict maximum value for tensor elements."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)

                if data.max() >= lt:
                    raise ValueError(f"Expected less than {lt}, got {data.max()}")
                return data

        return InheritTensor

    @classmethod
    def ne(cls: Type[T], ne: float) -> Type[T]:
        """Specify a value that tensor elements should not equal."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)

                if (data == ne).any():
                    raise ValueError(f"Unexpected value {ne}")
                return data

        return InheritTensor

    @classmethod
    def device(cls: Type[T], device: torch.device) -> Type[T]:
        """Specify the device for the tensor."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                return super().validate(data).to(device)

        return InheritTensor

    @classmethod
    def cpu(cls: Type[T]) -> Type[T]:
        """Specify that the tensor should be on CPU."""
        return cls.device(torch.device("cpu"))

    @classmethod
    def cuda(cls: Type[T]) -> Type[T]:
        """Specify that the tensor should be on CUDA."""
        return cls.device(torch.device("cuda"))

    @classmethod
    def dtype(cls: Type[T], dtype: torch.dtype) -> Type[T]:
        """Specify the data type for the tensor."""

        class InheritTensor(cls):
            @classmethod
            def validate(
                cls, data: torch.Tensor, config: Any = None, field: Any = None
            ) -> torch.Tensor:
                data = super().validate(data)
                if data.dtype == dtype:
                    return data
                else:
                    new_data = data.type(dtype)
                    if not torch.allclose(
                        data.float(), new_data.float(), equal_nan=True
                    ):
                        raise ValueError(
                            f"Was unable to cast from {data.dtype} to {dtype}"
                        )
                    return new_data

        return InheritTensor

    @classmethod
    def float(cls: Type[T]) -> Type[T]:
        """Specify float32 data type for the tensor."""
        return cls.dtype(torch.float32)

    @classmethod
    def float32(cls: Type[T]) -> Type[T]:
        """Specify float32 data type for the tensor."""
        return cls.dtype(torch.float32)

    @classmethod
    def half(cls: Type[T]) -> Type[T]:
        """Specify float16 data type for the tensor."""
        return cls.dtype(torch.float16)

    @classmethod
    def float16(cls: Type[T]) -> Type[T]:
        """Specify float16 data type for the tensor."""
        return cls.dtype(torch.float16)

    @classmethod
    def double(cls: Type[T]) -> Type[T]:
        """Specify float64 data type for the tensor."""
        return cls.dtype(torch.float64)

    @classmethod
    def float64(cls: Type[T]) -> Type[T]:
        """Specify float64 data type for the tensor."""
        return cls.dtype(torch.float64)

    @classmethod
    def int(cls: Type[T]) -> Type[T]:
        """Specify int32 data type for the tensor."""
        return cls.dtype(torch.int32)

    @classmethod
    def int32(cls: Type[T]) -> Type[T]:
        """Specify int32 data type for the tensor."""
        return cls.dtype(torch.int32)

    @classmethod
    def long(cls: Type[T]) -> Type[T]:
        """Specify int64 data type for the tensor."""
        return cls.dtype(torch.int64)

    @classmethod
    def int64(cls: Type[T]) -> Type[T]:
        """Specify int64 data type for the tensor."""
        return cls.dtype(torch.int64)

    @classmethod
    def short(cls: Type[T]) -> Type[T]:
        """Specify int16 data type for the tensor."""
        return cls.dtype(torch.int16)

    @classmethod
    def int16(cls: Type[T]) -> Type[T]:
        """Specify int16 data type for the tensor."""
        return cls.dtype(torch.int16)

    @classmethod
    def byte(cls: Type[T]) -> Type[T]:
        """Specify uint8 data type for the tensor."""
        return cls.dtype(torch.uint8)

    @classmethod
    def uint8(cls: Type[T]) -> Type[T]:
        """Specify uint8 data type for the tensor."""
        return cls.dtype(torch.uint8)

    @classmethod
    def bool(cls: Type[T]) -> Type[T]:
        """Specify boolean data type for the tensor."""
        return cls.dtype(torch.bool)


def test_base_model():
    from pydantic import BaseModel

    class Test(BaseModel):
        tensor: Annotated[torch.Tensor, Tensor.dims("NCHW").float()]

    Test(tensor=torch.ones(10, 3, 32, 32))


def test_validate():
    from pytest import raises

    with raises(ValueError):
        Tensor.ndim(4).validate(torch.ones(3, 4, 5))


def test_conversion():
    import numpy as np
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Annotated[torch.Tensor, Tensor.dims("N")]
        numbers2: Annotated[torch.Tensor, Tensor.dims("N")]

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
        numbers: Annotated[torch.Tensor, Tensor.uint8()]

    Test(numbers=[1, 2, 3])

    with raises(ValueError):
        Test(numbers=[1.5, 2.2, 3.2])

    class TestBool(BaseModel):
        flags: Annotated[torch.Tensor, Tensor.bool()]

    TestBool(flags=[True, False, True])

    with raises(ValueError):
        TestBool(numbers=[1.5, 2.2, 3.2])


def test_device():
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Annotated[torch.Tensor, Tensor.float().cpu()]

    Test(numbers=[1, 2, 3])


def test_from_numpy():
    from pydantic import BaseModel

    class Test(BaseModel):
        numbers: Annotated[torch.Tensor, Tensor]

    numbers = np.array([1, 2, 3])
    torch_numbers = Test(numbers=numbers).numbers

    assert type(torch_numbers) == torch.Tensor
    assert np.allclose(torch_numbers.numpy(), numbers)


def test_ge():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Annotated[torch.Tensor, Tensor.ge(0)]

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[-1.5, 2.2, 3.2])


def test_ne():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Annotated[torch.Tensor, Tensor.ne(1)]

    Test(numbers=[1.5, 2.2, 3.2])

    with raises(ValueError):
        Test(numbers=[1, 2.2, 3.2])


def test_shorthand_syntax():
    from pydantic import BaseModel
    from pytest import raises

    class Test(BaseModel):
        numbers: Tensor.dims("N").float()

    Test(numbers=[1.5, 2.2, 3.2]).numbers

    with raises(ValueError):
        Test(numbers=[[1, 2.2, 3.2], [1, 2, 3]])
