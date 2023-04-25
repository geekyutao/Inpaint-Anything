import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class ITBDataset(BaseDataset):
    """ NUS-PRO dataset
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.itb_path
        self.sequence_info_list = self._get_sequence_info_list(self.base_path)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: NUS has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def get_fileNames(self, rootdir):
        fs = []
        fs_all = []
        for root, dirs, files in os.walk(rootdir, topdown=True):
            files.sort()
            files.sort(key=len)
            if files is not None:
                for name in files:
                    _, ending = os.path.splitext(name)
                    if ending == ".jpg":
                        _, root_ = os.path.split(root)
                        fs.append(os.path.join(root_, name))
                        fs_all.append(os.path.join(root, name))

        return fs_all, fs

    def _get_sequence_info_list(self, base_path):
        sequence_info_list = []
        for scene in os.listdir(base_path):
            if '.' in scene:
                continue
            videos = os.listdir(os.path.join(base_path, scene))
            for video in videos:
                _, fs = self.get_fileNames(os.path.join(base_path, scene, video))
                video_tmp = {"name": video, "path": scene + '/' + video, "startFrame": 1, "endFrame": len(fs),
                             "nz": len(fs[0].split('/')[-1].split('.')[0]), "ext": "jpg",
                             "anno_path": scene + '/' + video + "/groundtruth.txt",
                             "object_class": "unknown"}
                sequence_info_list.append(video_tmp)

        return sequence_info_list  # sequence_info_list_50 #
