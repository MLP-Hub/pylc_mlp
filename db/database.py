"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: HDF5 Database Wrapper
File: database.py
"""
import json
import os
import math
import h5py
import numpy as np
from config import defaults
from utils.tools import confirm_write_file


class DB(object):
    """
     Wrapper class for Hierarchical Data Format
     version 5 (HDF5) database
     - General database operations for image/mask datasets
     - Multiprocessing enabled
     Data model schema:
     - img: images dataset
     - mask: masks dataset
     - meta: metadata (see Profiler for metadata schema)
    """

    def __init__(self, path=None, data=None, partition=None, worker=None, clip=None):
        """
            Load database from path or from input data array.

            Parameters
            ------
            path: str
                h5 database path.
            data: np.array
                Input data as {'img':img_data, 'mask':mask_data, 'meta':metadata}
            partition: tuple
                Training/validation dataset ratio [optional].
            worker: Worker
                Worker pool [optional].
        """

        assert (path is not None or data is not None) and not (path is not None and data is not None), \
            "Database requires either a path or input data to load."

        self.data = None
        self.path = None

        # db dims
        self.partition = partition if partition is not None else (0., 1.)
        self.partition_size = None
        self.img_shape = None
        self.mask_shape = None
        self.clip = clip if clip is not None else defaults.clip

        # db indicies
        self.start = None
        self.end = None
        self.current = None
        self.next = None

        try:
            # load db with input data
            if data is not None:
                self.size = int(self.clip * len(data['img']))
                self.img_shape = data['img'].shape
                self.mask_shape = data['mask'].shape
                self.data = data
            # load db from file
            else:
                assert os.path.exists(path), "Database path {} does not exist."
                self.path = path
                f = self.open()
                self.size = int(self.clip * len(f['img']))
                self.img_shape = f['img'].shape
                self.mask_shape = f['mask'].shape
                f.close()
        except Exception as e:
            print('Error loading database:\n\t{}'.format(data.shape if data is not None else path))
            print(e)

        # partition database for dataset
        self.start = int(math.ceil(self.partition[0] * self.size))
        self.end = int(math.ceil(self.partition[1] * self.size))
        self.partition_size = self.end - self.start

        # partition dataset for worker pool
        if worker:
            per_worker = int(math.ceil(self.partition_size / float(worker.num_workers)))
            self.start += worker.id * per_worker
            self.end = min(self.start + per_worker, self.end)
            self.start = self.end if self.end < self.start else self.start
            self.partition_size = self.end - self.start

        # initialize buffer size and iterator
        self.buffer_size = min(defaults.buffer_size, self.partition_size)
        self.current = self.start
        self.next = self.current + self.buffer_size

    def __iter__(self):
        """
        Iterator to load indicies for next dataset chunk
        """
        return self

    def __next__(self):
        """
        Iterate next to load indicies for next dataset chunk
        """
        if self.current == self.end:
            raise StopIteration

        # iterate; if last chunk, truncate
        db_sl = (np.s_[self.current:self.next], self.next - self.current)
        self.current = self.next
        self.next = self.next + self.buffer_size if self.next + self.buffer_size < self.end else self.end

        return db_sl

    def __len__(self):
        return self.size

    def get_meta(self):
        """
        Get metadata attribute from dataset.

        Returns
        -------
        attr: Parameters
            Database metadata.
        """
        if self.path:
            f = self.open()
            attr = f.attrs.get('meta')
            f.close()
            return defaults.update(json.loads(attr))
        else:
            return self.data['meta']

    def get_data(self, dset_key):
        """
        Get dataset from database by key.

        Parameters
        ------
        dset_key: str
            Dataset key.

        Returns
        -------
        data: np.array
            Dataset array.
        """
        if self.path:
            f = self.open()
            data = f[dset_key]
            f.close()
            return data
        else:
            return self.data[dset_key]

    def open(self):
        """
        Open dataset file pointer. Uses H5PY Single-Writer/Multiple-Reader (SWMR).
        """
        return h5py.File(self.path, mode='r', libver='latest', swmr=True)

    def save(self, file_path):
        """
        Saves data buffer to HDF5 database file.

        Parameters
        ------
        file_path: str
            Database file path.
        """

        assert file_path is not None, "File path must be specified to save data to database."

        if len(self.data['img']) == 0 or len(self.data['mask']) == 0:
            print('\n --- Note: Image or mask data is empty.\n')

        n_samples = len(self.data['img'])
        print('\nSaving buffer to database ... ')

        if confirm_write_file(file_path):
            print('\nCopying {} samples to:\n\t{}  '.format(n_samples, file_path))
            with h5py.File(file_path, 'w') as f:
                # create image dataset partition
                f.create_dataset(
                    "img",
                    self.data['img'].shape,
                    compression='gzip',
                    chunks=True,
                    data=self.data['img']
                )
                # create masks dataset partition
                f.create_dataset(
                    "mask",
                    self.data['mask'].shape,
                    compression='gzip',
                    chunks=True,
                    data=self.data['mask']
                )
                # include dataset metadata as new attribute to database
                # - store metadata as JSON string
                f.attrs['meta'] = json.dumps(vars(self.data['meta']))
                f.close()
                print('file saved.')
        else:
            print('Database was not saved.')