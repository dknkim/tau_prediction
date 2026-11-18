from pathlib import Path
from typing import List, Optional, Union

import h5py
import nibabel as nib
import numpy as np


class DatasetInstance:
    def __init__(self,
                 image_path: Union[str, Path, List[Union[str, Path]]] = None,
                 sequence_type: Optional[str] = None,
                 field_strength: float = None,
                 time: Optional[str] = None,
                 date: Optional[str] = None,
                 ):
        """
        Args:
            image_path (str, Path, list): The path where the data is stored
            sequence_type (str): The sequence type for MRI
            field_strength (float): Field strength of the scan
            pre_contrast (bool): Is the scan pre contrast
            post_contrast (bool): Is the scan post contrast
            shape (tuple): The shape of the data
        """

        if isinstance(image_path, (Path, str)):
            self.image_path = str(image_path)
            if not Path(image_path).is_file():
                print('The path: ' + str(image_path))
                print('Is not an existing file, are you sure this is the correct path?')
        else:
            self.image_path = image_path

        self.sequence_type = sequence_type
        self.field_strength = field_strength
        self.time = time
        self.date = date

    def open(self, open_func=None):
        """
        Open the file
        Args:
            open_func (the function to open the file)
        returns:
            the opened file
        """
        if open_func is not None:
            image = open_func(self.image_path)
        else:
            suffix = Path(self.image_path).suffix
            if suffix == '.h5':
                image = self.open_hdf5(self.image_path)
            elif suffix in ['.nii', '.gz']:
                image = self.open_nifti(self.image_path)
            elif suffix in ['.npy', '.npz']:
                image = self.open_numpy(self.image_path)
            else:
                raise TypeError('cannot open file: ', self.image_path)
        return image

    def open_hdf5(self, image_path):
        return h5py.File(image_path, 'r')

    def open_nifti(self, image_path):
        return nib.load(image_path)

    def open_numpy(self, image_path):
        if Path(image_path).suffix in "'.npz":
            images = np.load(image_path)
            return images[list(images.keys())[0]]
        return np.load(image_path)

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.date = in_dict['date']
            self.image_path = in_dict['image_path']
            self.sequence_type = in_dict['sequence_type']
            self.field_strength = in_dict['field_strength']
            self.time = in_dict['time']

        return self
