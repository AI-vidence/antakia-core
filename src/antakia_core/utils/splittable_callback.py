class ProgressCallback:
    def __call__(self, *args, **kwargs):
        raise NotImplemented

    def split(self, value: float | list[float]):
        raise NotImplemented
