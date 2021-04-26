import numpy as np
import io
import cv2


class Numpy(np.ndarray):
    @staticmethod
    def from_matplotlib_figure(figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=90, bbox_inches="tight")
        buffer.seek(0)
        image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        buffer.close()
        image = cv2.imdecode(image, 1)
        return image

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, data):
        if isinstance(data, cls):
            return data.view(np.ndarray)
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    @classmethod
    def ndim(cls, ndim):
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritNumpy

    @classmethod
    def dims(cls, dims):
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
    def shape(cls, *sizes):
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
    def between(cls, geq, leq):
        class InheritNumpy(cls):
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

        return InheritNumpy

    @classmethod
    def dtype(cls, dtype):
        class InheritNumpy(cls):
            @classmethod
            def validate(cls, data):
                data = super().validate(data)
                new_data = data.astype(dtype)
                if not np.allclose(data, new_data, equal_nan=True):
                    raise ValueError(f"Was unable to cast from {data.dtype} to {dtype}")
                return new_data

        return InheritNumpy


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
        numbers: Numpy.dtype(np.uint8)

    Test(numbers=[1, 2, 3])

    with raises(ValueError):
        Test(numbers=[1.5, 2.2, 3.2])
