import json
from copy import deepcopy
from pathlib import Path

from .datasetentry import DatasetEntry


class BaseContainer:
    def __init__(self, entries: list[DatasetEntry]) -> None:
        self.entries = entries

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def __add__(self, other):
        new_container = deepcopy(self)
        new_container.entries += deepcopy(other.entries)
        return new_container

    def from_dict(self, in_dict):
        """
        Appends data into DatasetContainer from dict
        Args:
            in_dict (dict): Dict to append data from, meant to be used to recover when loading from file
        """
        for entry in in_dict['entries']:
            self.entries.append(DatasetEntry().from_dict(entry))

    @classmethod
    def from_json(cls, path: str | Path):
        """
        Load DatasetContainer from file
        Args:
            path (str, Path): Path to load from
        returns:
            The DatasetContainer from file
        """
        with open(path) as json_file:
            data = json.load(json_file)
        new_container = cls()
        new_container.from_dict(data)
        return new_container
