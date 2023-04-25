import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class NFSDataset(BaseDataset):
    """ NFS dataset.
    Publication:
        Need for Speed: A Benchmark for Higher Frame Rate Object Tracking
        H. Kiani Galoogahi, A. Fagg, C. Huang, D. Ramanan, and S.Lucey
        ICCV, 2017
        http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Need_for_Speed_ICCV_2017_paper.pdf
    Download the dataset from http://ci2cv.net/nfs/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nfs_path
        self.sequence_info_list = self._get_sequence_info_list()

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
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)

        return Sequence(sequence_info['name'], frames, 'nfs', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "nfs_Gymnastics", "path": "sequences/Gymnastics", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_Gymnastics.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_MachLoop_jet", "path": "sequences/MachLoop_jet", "startFrame": 1, "endFrame": 99, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_MachLoop_jet.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_Skiing_red", "path": "sequences/Skiing_red", "startFrame": 1, "endFrame": 69, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_Skiing_red.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_Skydiving", "path": "sequences/Skydiving", "startFrame": 1, "endFrame": 196, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_Skydiving.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_airboard_1", "path": "sequences/airboard_1", "startFrame": 1, "endFrame": 425, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_airboard_1.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_airplane_landing", "path": "sequences/airplane_landing", "startFrame": 1, "endFrame": 81, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_airplane_landing.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_airtable_3", "path": "sequences/airtable_3", "startFrame": 1, "endFrame": 482, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_airtable_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_1", "path": "sequences/basketball_1", "startFrame": 1, "endFrame": 282, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_1.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_2", "path": "sequences/basketball_2", "startFrame": 1, "endFrame": 102, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_3", "path": "sequences/basketball_3", "startFrame": 1, "endFrame": 421, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_6", "path": "sequences/basketball_6", "startFrame": 1, "endFrame": 224, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_6.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_basketball_7", "path": "sequences/basketball_7", "startFrame": 1, "endFrame": 240, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_7.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_basketball_player", "path": "sequences/basketball_player", "startFrame": 1, "endFrame": 369, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_player.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_basketball_player_2", "path": "sequences/basketball_player_2", "startFrame": 1, "endFrame": 437, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_basketball_player_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_beach_flipback_person", "path": "sequences/beach_flipback_person", "startFrame": 1, "endFrame": 61, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_beach_flipback_person.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_bee", "path": "sequences/bee", "startFrame": 1, "endFrame": 45, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bee.txt", "object_class": "insect", 'occlusion': False},
            {"name": "nfs_biker_acrobat", "path": "sequences/biker_acrobat", "startFrame": 1, "endFrame": 128, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_acrobat.txt", "object_class": "bicycle", 'occlusion': False},
            {"name": "nfs_biker_all_1", "path": "sequences/biker_all_1", "startFrame": 1, "endFrame": 113, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_all_1.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_biker_head_2", "path": "sequences/biker_head_2", "startFrame": 1, "endFrame": 132, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_head_2.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_biker_head_3", "path": "sequences/biker_head_3", "startFrame": 1, "endFrame": 254, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_head_3.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_biker_upper_body", "path": "sequences/biker_upper_body", "startFrame": 1, "endFrame": 194, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_upper_body.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_biker_whole_body", "path": "sequences/biker_whole_body", "startFrame": 1, "endFrame": 572, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_biker_whole_body.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_billiard_2", "path": "sequences/billiard_2", "startFrame": 1, "endFrame": 604, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_billiard_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_3", "path": "sequences/billiard_3", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_billiard_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_6", "path": "sequences/billiard_6", "startFrame": 1, "endFrame": 771, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_billiard_6.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_7", "path": "sequences/billiard_7", "startFrame": 1, "endFrame": 724, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_billiard_7.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_billiard_8", "path": "sequences/billiard_8", "startFrame": 1, "endFrame": 778, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_billiard_8.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_bird_2", "path": "sequences/bird_2", "startFrame": 1, "endFrame": 476, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bird_2.txt", "object_class": "bird", 'occlusion': False},
            {"name": "nfs_book", "path": "sequences/book", "startFrame": 1, "endFrame": 288, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_book.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_bottle", "path": "sequences/bottle", "startFrame": 1, "endFrame": 2103, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bottle.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_bowling_1", "path": "sequences/bowling_1", "startFrame": 1, "endFrame": 303, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bowling_1.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bowling_2", "path": "sequences/bowling_2", "startFrame": 1, "endFrame": 710, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bowling_2.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bowling_3", "path": "sequences/bowling_3", "startFrame": 1, "endFrame": 271, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bowling_3.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bowling_6", "path": "sequences/bowling_6", "startFrame": 1, "endFrame": 260, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bowling_6.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_bowling_ball", "path": "sequences/bowling_ball", "startFrame": 1, "endFrame": 275, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bowling_ball.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_bunny", "path": "sequences/bunny", "startFrame": 1, "endFrame": 705, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_bunny.txt", "object_class": "mammal", 'occlusion': False},
            {"name": "nfs_car", "path": "sequences/car", "startFrame": 1, "endFrame": 2020, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car.txt", "object_class": "car", 'occlusion': True},
            {"name": "nfs_car_camaro", "path": "sequences/car_camaro", "startFrame": 1, "endFrame": 36, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_camaro.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_drifting", "path": "sequences/car_drifting", "startFrame": 1, "endFrame": 173, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_drifting.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_jumping", "path": "sequences/car_jumping", "startFrame": 1, "endFrame": 22, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_jumping.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_rc_rolling", "path": "sequences/car_rc_rolling", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_rc_rolling.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_rc_rotating", "path": "sequences/car_rc_rotating", "startFrame": 1, "endFrame": 80, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_rc_rotating.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_side", "path": "sequences/car_side", "startFrame": 1, "endFrame": 108, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_side.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_car_white", "path": "sequences/car_white", "startFrame": 1, "endFrame": 2063, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_car_white.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_cheetah", "path": "sequences/cheetah", "startFrame": 1, "endFrame": 167, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_cheetah.txt", "object_class": "mammal", 'occlusion': True},
            {"name": "nfs_cup", "path": "sequences/cup", "startFrame": 1, "endFrame": 1281, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_cup.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_cup_2", "path": "sequences/cup_2", "startFrame": 1, "endFrame": 182, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_cup_2.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_dog", "path": "sequences/dog", "startFrame": 1, "endFrame": 1030, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_dog.txt", "object_class": "dog", 'occlusion': True},
            {"name": "nfs_dog_1", "path": "sequences/dog_1", "startFrame": 1, "endFrame": 168, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_dog_1.txt", "object_class": "dog", 'occlusion': False},
            {"name": "nfs_dog_2", "path": "sequences/dog_2", "startFrame": 1, "endFrame": 594, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_dog_2.txt", "object_class": "dog", 'occlusion': True},
            {"name": "nfs_dog_3", "path": "sequences/dog_3", "startFrame": 1, "endFrame": 200, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_dog_3.txt", "object_class": "dog", 'occlusion': False},
            {"name": "nfs_dogs", "path": "sequences/dogs", "startFrame": 1, "endFrame": 198, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_dogs.txt", "object_class": "dog", 'occlusion': True},
            {"name": "nfs_dollar", "path": "sequences/dollar", "startFrame": 1, "endFrame": 1426, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_dollar.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_drone", "path": "sequences/drone", "startFrame": 1, "endFrame": 70, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_drone.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_ducks_lake", "path": "sequences/ducks_lake", "startFrame": 1, "endFrame": 107, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_ducks_lake.txt", "object_class": "bird", 'occlusion': False},
            {"name": "nfs_exit", "path": "sequences/exit", "startFrame": 1, "endFrame": 359, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_exit.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_first", "path": "sequences/first", "startFrame": 1, "endFrame": 435, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_first.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_flower", "path": "sequences/flower", "startFrame": 1, "endFrame": 448, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_flower.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_footbal_skill", "path": "sequences/footbal_skill", "startFrame": 1, "endFrame": 131, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_footbal_skill.txt", "object_class": "ball", 'occlusion': True},
            {"name": "nfs_helicopter", "path": "sequences/helicopter", "startFrame": 1, "endFrame": 310, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_helicopter.txt", "object_class": "aircraft", 'occlusion': False},
            {"name": "nfs_horse_jumping", "path": "sequences/horse_jumping", "startFrame": 1, "endFrame": 117, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_horse_jumping.txt", "object_class": "horse", 'occlusion': True},
            {"name": "nfs_horse_running", "path": "sequences/horse_running", "startFrame": 1, "endFrame": 139, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_horse_running.txt", "object_class": "horse", 'occlusion': False},
            {"name": "nfs_iceskating_6", "path": "sequences/iceskating_6", "startFrame": 1, "endFrame": 603, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_iceskating_6.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_jellyfish_5", "path": "sequences/jellyfish_5", "startFrame": 1, "endFrame": 746, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_jellyfish_5.txt", "object_class": "invertebrate", 'occlusion': False},
            {"name": "nfs_kid_swing", "path": "sequences/kid_swing", "startFrame": 1, "endFrame": 169, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_kid_swing.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_motorcross", "path": "sequences/motorcross", "startFrame": 1, "endFrame": 39, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_motorcross.txt", "object_class": "vehicle", 'occlusion': True},
            {"name": "nfs_motorcross_kawasaki", "path": "sequences/motorcross_kawasaki", "startFrame": 1, "endFrame": 65, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_motorcross_kawasaki.txt", "object_class": "vehicle", 'occlusion': False},
            {"name": "nfs_parkour", "path": "sequences/parkour", "startFrame": 1, "endFrame": 58, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_parkour.txt", "object_class": "person head", 'occlusion': False},
            {"name": "nfs_person_scooter", "path": "sequences/person_scooter", "startFrame": 1, "endFrame": 413, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_person_scooter.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_pingpong_2", "path": "sequences/pingpong_2", "startFrame": 1, "endFrame": 1277, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_pingpong_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_pingpong_7", "path": "sequences/pingpong_7", "startFrame": 1, "endFrame": 1290, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_pingpong_7.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_pingpong_8", "path": "sequences/pingpong_8", "startFrame": 1, "endFrame": 296, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_pingpong_8.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_purse", "path": "sequences/purse", "startFrame": 1, "endFrame": 968, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_purse.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_rubber", "path": "sequences/rubber", "startFrame": 1, "endFrame": 1328, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_rubber.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_running", "path": "sequences/running", "startFrame": 1, "endFrame": 677, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_running.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_running_100_m", "path": "sequences/running_100_m", "startFrame": 1, "endFrame": 313, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_running_100_m.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_running_100_m_2", "path": "sequences/running_100_m_2", "startFrame": 1, "endFrame": 337, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_running_100_m_2.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_running_2", "path": "sequences/running_2", "startFrame": 1, "endFrame": 363, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_running_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_shuffleboard_1", "path": "sequences/shuffleboard_1", "startFrame": 1, "endFrame": 42, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_1.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_2", "path": "sequences/shuffleboard_2", "startFrame": 1, "endFrame": 41, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_2.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_4", "path": "sequences/shuffleboard_4", "startFrame": 1, "endFrame": 62, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_4.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_5", "path": "sequences/shuffleboard_5", "startFrame": 1, "endFrame": 32, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_5.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffleboard_6", "path": "sequences/shuffleboard_6", "startFrame": 1, "endFrame": 52, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffleboard_6.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffletable_2", "path": "sequences/shuffletable_2", "startFrame": 1, "endFrame": 372, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffletable_2.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffletable_3", "path": "sequences/shuffletable_3", "startFrame": 1, "endFrame": 368, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffletable_3.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_shuffletable_4", "path": "sequences/shuffletable_4", "startFrame": 1, "endFrame": 101, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_shuffletable_4.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_ski_long", "path": "sequences/ski_long", "startFrame": 1, "endFrame": 274, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_ski_long.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_soccer_ball", "path": "sequences/soccer_ball", "startFrame": 1, "endFrame": 163, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_soccer_ball.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_soccer_ball_2", "path": "sequences/soccer_ball_2", "startFrame": 1, "endFrame": 1934, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_soccer_ball_2.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_soccer_ball_3", "path": "sequences/soccer_ball_3", "startFrame": 1, "endFrame": 1381, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_soccer_ball_3.txt", "object_class": "ball", 'occlusion': False},
            {"name": "nfs_soccer_player_2", "path": "sequences/soccer_player_2", "startFrame": 1, "endFrame": 475, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_soccer_player_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_soccer_player_3", "path": "sequences/soccer_player_3", "startFrame": 1, "endFrame": 319, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_soccer_player_3.txt", "object_class": "person", 'occlusion': True},
            {"name": "nfs_stop_sign", "path": "sequences/stop_sign", "startFrame": 1, "endFrame": 302, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_stop_sign.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_suv", "path": "sequences/suv", "startFrame": 1, "endFrame": 2584, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_suv.txt", "object_class": "car", 'occlusion': False},
            {"name": "nfs_tiger", "path": "sequences/tiger", "startFrame": 1, "endFrame": 1556, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_tiger.txt", "object_class": "mammal", 'occlusion': False},
            {"name": "nfs_walking", "path": "sequences/walking", "startFrame": 1, "endFrame": 555, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_walking.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_walking_3", "path": "sequences/walking_3", "startFrame": 1, "endFrame": 1427, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_walking_3.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_water_ski_2", "path": "sequences/water_ski_2", "startFrame": 1, "endFrame": 47, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_water_ski_2.txt", "object_class": "person", 'occlusion': False},
            {"name": "nfs_yoyo", "path": "sequences/yoyo", "startFrame": 1, "endFrame": 67, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_yoyo.txt", "object_class": "other", 'occlusion': False},
            {"name": "nfs_zebra_fish", "path": "sequences/zebra_fish", "startFrame": 1, "endFrame": 671, "nz": 5, "ext": "jpg", "anno_path": "anno/nfs_zebra_fish.txt", "object_class": "fish", 'occlusion': False},
        ]

        return sequence_info_list