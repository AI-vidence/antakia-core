from __future__ import annotations
import time
from abc import ABC, abstractmethod

import pandas as pd

from antakia_core.utils.splittable_callback import ProgressCallback, DummyProgressCallback


class LongTask(ABC):
    """
    Abstract class to compute long tasks, often in a separate thread.

    Attributes
    ----------
    X : dataframe
    progress_updated : an optional callback function to call when progress is updated
    start_time : float
    progress:int
    """

    def __init__(self,
                 X: pd.DataFrame | None = None,
                 progress_callback: ProgressCallback | None = None):
        if X is None:
            raise ValueError("You must provide a dataframe for a LongTask")
        self.X = X
        if progress_callback is None:
            progress_callback = DummyProgressCallback()
        self.progress_updated: ProgressCallback = progress_callback
        self.start_time = time.time()
        self.progress = 0

    def publish_progress(self, progress: int):
        self.progress = progress
        self.progress_updated(progress, time.time() - self.start_time)

    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        """
        Method to compute the long task and update listener with the progress.
        """
        pass
