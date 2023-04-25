from typing import Union, TextIO

import numpy as np
from numba import jit

from lib.test.evaluation.data import SequenceList, BaseDataset, Sequence


class VOTDataset(BaseDataset):
    """
    VOT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html
    """
    def __init__(self, year=18):
        super().__init__()
        self.year = year
        if year == 18:
            self.base_path = self.env_settings.vot18_path
        elif year == 20:
            self.base_path = self.env_settings.vot20_path
        elif year == 22:
            self.base_path = self.env_settings.vot22_path
        self.sequence_list = self._get_sequence_list(year)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        if self.year == 18 or self.year == 22:
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

            end_frame = ground_truth_rect.shape[0]

            frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                      sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                      for frame_num in range(start_frame, end_frame+1)]

            # Convert gt
            if ground_truth_rect.shape[1] > 4:
                gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
                gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

                x1 = np.amin(gt_x_all, 1).reshape(-1,1)
                y1 = np.amin(gt_y_all, 1).reshape(-1,1)
                x2 = np.amax(gt_x_all, 1).reshape(-1,1)
                y2 = np.amax(gt_y_all, 1).reshape(-1,1)

                ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        elif self.year == 20:
            ground_truth_rect = read_file(str(anno_path))
            ground_truth_rect = np.array(ground_truth_rect, dtype=np.float64)
            end_frame = ground_truth_rect.shape[0]

            frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                                     sequence_path=sequence_path,
                                                                                     frame=frame_num, nz=nz, ext=ext)
                      for frame_num in range(start_frame, end_frame + 1)]

        else:
            raise NotImplementedError

        return Sequence(sequence_name, frames, 'vot', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, year):
        if year == 18:
            sequence_list= ['ants1',
                            'ants3',
                            'bag',
                            'ball1',
                            'ball2',
                            'basketball',
                            'birds1',
                            'blanket',
                            'bmx',
                            'bolt1',
                            'bolt2',
                            'book',
                            'butterfly',
                            'car1',
                            'conduction1',
                            'crabs1',
                            'crossing',
                            'dinosaur',
                            'drone_across',
                            'drone_flip',
                            'drone1',
                            'fernando',
                            'fish1',
                            'fish2',
                            'fish3',
                            'flamingo1',
                            'frisbee',
                            'girl',
                            'glove',
                            'godfather',
                            'graduate',
                            'gymnastics1',
                            'gymnastics2',
                            'gymnastics3',
                            'hand',
                            'handball1',
                            'handball2',
                            'helicopter',
                            'iceskater1',
                            'iceskater2',
                            'leaves',
                            'matrix',
                            'motocross1',
                            'motocross2',
                            'nature',
                            'pedestrian1',
                            'rabbit',
                            'racing',
                            'road',
                            'shaking',
                            'sheep',
                            'singer2',
                            'singer3',
                            'soccer1',
                            'soccer2',
                            'soldier',
                            'tiger',
                            'traffic',
                            'wiper',
                            'zebrafish1']
        elif year == 20:

            sequence_list= ['agility',
                            'ants1',
                            'ball2',
                            'ball3',
                            'basketball',
                            'birds1',
                            'bolt1',
                            'book',
                            'butterfly',
                            'car1',
                            'conduction1',
                            'crabs1',
                            'dinosaur',
                            'dribble',
                            'drone1',
                            'drone_across',
                            'drone_flip',
                            'fernando',
                            'fish1',
                            'fish2',
                            'flamingo1',
                            'frisbee',
                            'girl',
                            'glove',
                            'godfather',
                            'graduate',
                            'gymnastics1',
                            'gymnastics2',
                            'gymnastics3',
                            'hand',
                            'hand02',
                            'hand2',
                            'handball1',
                            'handball2',
                            'helicopter',
                            'iceskater1',
                            'iceskater2',
                            'lamb',
                            'leaves',
                            'marathon',
                            'matrix',
                            'monkey',
                            'motocross1',
                            'nature',
                            'polo',
                            'rabbit',
                            'rabbit2',
                            'road',
                            'rowing',
                            'shaking',
                            'singer2',
                            'singer3',
                            'soccer1',
                            'soccer2',
                            'soldier',
                            'surfing',
                            'tiger',
                            'wheel',
                            'wiper',
                            'zebrafish1']
        elif year == 22:
            sequence_list= ['agility',
                            'animal',
                            'ants1',
                            'bag',
                            'ball2',
                            'ball3',
                            'basketball',
                            'birds1',
                            'birds2',
                            'bolt1',
                            'book',
                            'bubble',
                            'butterfly',
                            'car1',
                            'conduction1',
                            'crabs1',
                            'dinosaur',
                            'diver',
                            'drone1',
                            'drone_across',
                            'fernando',
                            'fish1',
                            'fish2',
                            'flamingo1',
                            'frisbee',
                            'girl',
                            'graduate',
                            'gymnastics1',
                            'gymnastics2',
                            'gymnastics3',
                            'hand',
                            'hand2',
                            'handball1',
                            'handball2',
                            'helicopter',
                            'iceskater1',
                            'iceskater2',
                            'kangaroo',
                            'lamb',
                            'leaves',
                            'marathon',
                            'matrix',
                            'monkey',
                            'motocross1',
                            'nature',
                            'polo',
                            'rabbit',
                            'rabbit2',
                            'rowing',
                            'shaking',
                            'singer2',
                            'singer3',
                            'snake',
                            'soccer1',
                            'soccer2',
                            'soldier',
                            'surfing',
                            'tennis',
                            'tiger',
                            'wheel',
                            'wiper',
                            'zebrafish1']

        else:
            raise NotImplementedError

        return sequence_list


def parse(string):
    """
    parse string to the appropriate region format and return region object
    """
    from vot.region.shapes import Rectangle, Polygon, Mask


    if string[0] == 'm':
        # input is a mask - decode it
        m_, offset_, region = create_mask_from_string(string[1:].split(','))
        # return Mask(m_, offset=offset_)
        return region
    else:
        # input is not a mask - check if special, rectangle or polygon
        raise NotImplementedError
    print('Unknown region format.')
    return None


def read_file(fp: Union[str, TextIO]):
    if isinstance(fp, str):
        with open(fp) as file:
            lines = file.readlines()
    else:
        lines = fp.readlines()

    regions = []
    # iterate over all lines in the file
    for i, line in enumerate(lines):
        regions.append(parse(line.strip()))
    return regions


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)
    region = [tl_x, tl_y, region_w, region_h]

    return mask, (tl_x, tl_y), region

@jit(nopython=True)
def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]