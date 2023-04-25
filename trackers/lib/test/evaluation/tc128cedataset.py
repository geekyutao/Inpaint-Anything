import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob
import six


class TC128CEDataset(BaseDataset):
    """
    TC-128 Dataset (78 newly added sequences)
    modified from the implementation in got10k-toolkit (https://github.com/got-10k/toolkit)
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tc128_path
        self.anno_files = sorted(glob.glob(
            os.path.join(self.base_path, '*/*_gt.txt')))
        """filter the newly added sequences (_ce)"""
        self.anno_files = [s for s in self.anno_files if "_ce" in s]
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        # valid frame range for each sequence
        self.range_files = [glob.glob(os.path.join(d, '*_frames.txt'))[0] for d in self.seq_dirs]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.seq_names])

    def _construct_sequence(self, sequence_name):
        if isinstance(sequence_name, six.string_types):
            if not sequence_name in self.seq_names:
                raise Exception('Sequence {} not found.'.format(sequence_name))
            index = self.seq_names.index(sequence_name)
        # load valid frame range
        frames = np.loadtxt(self.range_files[index], dtype=int, delimiter=',')
        img_files = [os.path.join(self.seq_dirs[index], 'img/%04d.jpg' % f) for f in range(frames[0], frames[1] + 1)]

        # load annotations
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        # return img_files, anno
        return Sequence(sequence_name, img_files, 'tc128', anno.reshape(-1, 4))

    def __len__(self):
        return len(self.seq_names)
