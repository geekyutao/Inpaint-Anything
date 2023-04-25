import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LaSOTExtensionSubsetDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)
    Publication:
        LaSOT: A High-quality Large-scale Single Object Tracking Benchmark
        Heng Fan, Hexin Bai, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Harshit, Mingzhen Huang, Juehuan Liu,
        Yong Xu, Chunyuan Liao, Lin Yuan, Haibin Ling
        IJCV, 2020
        https://arxiv.org/pdf/2009.03465.pdf
    Download the dataset from http://vision.cs.stonybrook.edu/~lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasot_extension_subset_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot_extension_subset', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['atv-1',
                         'atv-2',
                         'atv-3',
                         'atv-4',
                         'atv-5',
                         'atv-6',
                         'atv-7',
                         'atv-8',
                         'atv-9',
                         'atv-10',
                         'badminton-1',
                         'badminton-2',
                         'badminton-3',
                         'badminton-4',
                         'badminton-5',
                         'badminton-6',
                         'badminton-7',
                         'badminton-8',
                         'badminton-9',
                         'badminton-10',
                         'cosplay-1',
                         'cosplay-10',
                         'cosplay-2',
                         'cosplay-3',
                         'cosplay-4',
                         'cosplay-5',
                         'cosplay-6',
                         'cosplay-7',
                         'cosplay-8',
                         'cosplay-9',
                         'dancingshoe-1',
                         'dancingshoe-2',
                         'dancingshoe-3',
                         'dancingshoe-4',
                         'dancingshoe-5',
                         'dancingshoe-6',
                         'dancingshoe-7',
                         'dancingshoe-8',
                         'dancingshoe-9',
                         'dancingshoe-10',
                         'footbag-1',
                         'footbag-2',
                         'footbag-3',
                         'footbag-4',
                         'footbag-5',
                         'footbag-6',
                         'footbag-7',
                         'footbag-8',
                         'footbag-9',
                         'footbag-10',
                         'frisbee-1',
                         'frisbee-2',
                         'frisbee-3',
                         'frisbee-4',
                         'frisbee-5',
                         'frisbee-6',
                         'frisbee-7',
                         'frisbee-8',
                         'frisbee-9',
                         'frisbee-10',
                         'jianzi-1',
                         'jianzi-2',
                         'jianzi-3',
                         'jianzi-4',
                         'jianzi-5',
                         'jianzi-6',
                         'jianzi-7',
                         'jianzi-8',
                         'jianzi-9',
                         'jianzi-10',
                         'lantern-1',
                         'lantern-2',
                         'lantern-3',
                         'lantern-4',
                         'lantern-5',
                         'lantern-6',
                         'lantern-7',
                         'lantern-8',
                         'lantern-9',
                         'lantern-10',
                         'misc-1',
                         'misc-2',
                         'misc-3',
                         'misc-4',
                         'misc-5',
                         'misc-6',
                         'misc-7',
                         'misc-8',
                         'misc-9',
                         'misc-10',
                         'opossum-1',
                         'opossum-2',
                         'opossum-3',
                         'opossum-4',
                         'opossum-5',
                         'opossum-6',
                         'opossum-7',
                         'opossum-8',
                         'opossum-9',
                         'opossum-10',
                         'paddle-1',
                         'paddle-2',
                         'paddle-3',
                         'paddle-4',
                         'paddle-5',
                         'paddle-6',
                         'paddle-7',
                         'paddle-8',
                         'paddle-9',
                         'paddle-10',
                         'raccoon-1',
                         'raccoon-2',
                         'raccoon-3',
                         'raccoon-4',
                         'raccoon-5',
                         'raccoon-6',
                         'raccoon-7',
                         'raccoon-8',
                         'raccoon-9',
                         'raccoon-10',
                         'rhino-1',
                         'rhino-2',
                         'rhino-3',
                         'rhino-4',
                         'rhino-5',
                         'rhino-6',
                         'rhino-7',
                         'rhino-8',
                         'rhino-9',
                         'rhino-10',
                         'skatingshoe-1',
                         'skatingshoe-2',
                         'skatingshoe-3',
                         'skatingshoe-4',
                         'skatingshoe-5',
                         'skatingshoe-6',
                         'skatingshoe-7',
                         'skatingshoe-8',
                         'skatingshoe-9',
                         'skatingshoe-10',
                         'wingsuit-1',
                         'wingsuit-2',
                         'wingsuit-3',
                         'wingsuit-4',
                         'wingsuit-5',
                         'wingsuit-6',
                         'wingsuit-7',
                         'wingsuit-8',
                         'wingsuit-9',
                         'wingsuit-10']
        return sequence_list