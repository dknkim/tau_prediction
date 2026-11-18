from .base import BaseContainer
from .datasetentry import DatasetEntry


class DatasetContainer(BaseContainer):
    def __init__(self, entries: list[DatasetEntry] | None = None) -> None:
        super().__init__(entries if entries is not None else [])
