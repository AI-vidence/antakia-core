from typing import Sequence


class ProgressCallback:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)


    def split(self,
              value: float | list[float]) -> Sequence['ProgressCallback']:
        raise NotImplemented


class DummyProgressCallback(ProgressCallback):

    def __call__(self, *args, **kwargs):
        super().__init__()
        self.progress = 0

    def split(self, value: float | list[float]) -> Sequence[ProgressCallback]:
        if isinstance(value, list):
            return [self for _ in value] + [self]
        return [self, self]
