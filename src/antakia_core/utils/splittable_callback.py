from typing import Sequence


class ProgressCallback:

    def __call__(self, *args, **kwargs):
        raise NotImplemented

    def split(self,
              value: float | list[float]) -> Sequence['ProgressCallback']:
        raise NotImplemented


class DummyProgressCallback(ProgressCallback):

    def __call__(self, *args, **kwargs):
        pass

    def split(self, value: float | list[float]) -> Sequence[ProgressCallback]:
        if isinstance(value, list):
            return [self for _ in value] + [self]
        return [self, self]
