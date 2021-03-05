import numpy as np


class Numpy(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, data, values, field, config):
        return data

    @classmethod
    def ndim(cls, ndim):
        class InheritNumpy(Numpy):
            @classmethod
            def validate(cls, data):
                if data.ndim != ndim:
                    raise ValueError(f"Expected {ndim} dims, got {data.ndim}")
                return data

        return InheritNumpy

    @classmethod
    def short(cls, dims):
        class InheritNumpy(Numpy):
            @classmethod
            def validate(cls, data):
                if data.ndim != len(dims):
                    raise ValueError(
                        f"Unexpected number of dims {data.ndim} for {dims}"
                    )
                return data

        return InheritNumpy

    @classmethod
    def shape(cls, *sizes):
        class InheritNumpy(Numpy):
            @classmethod
            def validate(cls, data):
                for data_size, size in zip(data.shape, sizes):
                    if size != -1 and data_size != size:
                        raise ValueError(f"Expected size {size}, got {data_size}")
                return data

        return InheritNumpy

    @classmethod
    def between(cls, geq, leq):
        class InheritNumpy(Numpy):
            @classmethod
            def validate(cls, data):
                data_min = data.min()
                if data_min < geq:
                    raise ValueError(f"Expected min value {geq}, got {data_min}")

                data_max = data.min()
                if data_max > leq:
                    raise ValueError(f"Expected max value {leq}, got {data_max}")
                return data

        return InheritNumpy


def test_base_model():
    from pydantic import BaseModel

    class Test(BaseModel):
        images: Numpy.short("nchw")

    Test(images=np.ones((10, 3, 32, 32)))


def test_validate():
    from pytest import raises

    with raises(ValueError):
        Numpy.ndim(4).validate(np.ones((3, 4, 5)))
