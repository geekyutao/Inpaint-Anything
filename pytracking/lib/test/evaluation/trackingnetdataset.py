import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
from lib.test.utils.load_text import load_text


class TrackingNetDataset(BaseDataset):
    """ TrackingNet test set.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.trackingnet_path

        sets = 'TEST'
        if not isinstance(sets, (list, tuple)):
            if sets == 'TEST':
                sets = ['TEST']
            elif sets == 'TRAIN':
                sets = ['TRAIN_{}'.format(i) for i in range(5)]

        self.sequence_list = self._list_sequences(self.base_path, sets)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(set, seq_name) for set, seq_name in self.sequence_list])

    def _construct_sequence(self, set, sequence_name):
        anno_path = '{}/{}/anno/{}.txt'.format(self.base_path, set, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/{}/frames/{}'.format(self.base_path, set, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'trackingnet', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _list_sequences(self, root, set_ids):
        sequence_list = []

        for s in set_ids:
            anno_dir = os.path.join(root, s, "anno")
            sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]

            sequence_list += sequences_cur_set

        return sequence_list
