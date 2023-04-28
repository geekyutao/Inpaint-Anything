import os
from pytracking.lib.test.utils import TrackerParams
from pytracking.lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    # prj_dir = env_settings().prj_dir
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    # save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ostrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" %
    #                                  (yaml_name, cfg.TEST.EPOCH))
    params.checkpoint = os.path.join(prj_dir, 'pretrain', f'{yaml_name}.pth')
    assert os.path.exists(params.checkpoint), f'checkpoint not found at {params.checkpoint}'

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
