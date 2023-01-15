import platform

import torch

from ltr.external.Trans2Seg.segmentron.models.model_zoo import get_segmentation_model
from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone
from ltr.external.Trans2Seg.segmentron.utils.default_setup import default_setup

def parameters():
    params = TrackerParams()

    # ONLY FOR PYCHARM DEBUG
    if platform.uname().node == 'kalyans-galaxybook-pro':
        params.debug = 1
        params.visualization = True
        params.use_gpu = False
    else:
        params.debug = 0
        params.visualization = False
        params.use_gpu = True

    params.train_feature_size = 18
    params.feature_stride = 16
    params.image_sample_size = params.train_feature_size*params.feature_stride
    params.search_area_scale = 5
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 2
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = False
    params.augmentation = {}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True
    params.conf_ths = 0.9
    params.search_area_rescaling_at_occlusion = True

    params.net = NetWithBackbone(net_path='TransToMPnet_ep0025_objwise_trans2k_2step.pth.tar', use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    params.use_gt_box = True
    params.plot_iou = True

    # cfg.update_from_file('../ltr/external/Trans2Seg/configs/trans10kv2/trans2seg/trans2seg_medium.yaml')
    # # cfg.update_from_list(args.opts)
    # # cfg.PHASE = 'test'
    # # # cfg.ROOT_PATH = root_path
    # # cfg.check_and_freeze()
    # # device = torch.device('cuda' if params.use_gpu else 'cpu')
    # # params.trans2seg = get_segmentation_model().to(device)

    return params
