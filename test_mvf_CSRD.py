import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from train_net_mvf import *
from config.config import *
import sys

sys.path.append(".")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

cfg = Config('CSRD_Virtual')  # CSRD_Virtual or CSRD_O
cfg.channel = 3  # control the channel of input image

# test_about
cfg.test_batch_size = 1
cfg.test_interval_epoch = 10
cfg.save_model_interval_epoch = 20
cfg.test_before_train = True
cfg.only_test = True  # if only test, set to True
# draw
cfg.draw_train_loss_fig = False
cfg.draw_train_loss_epoch_interval = 20

# extra note
cfg.model_name = 'mono21'
cfg.train_test_split = '11'  # split ratio
cfg.core_task = 'go_mvf'
cfg.extra = 'test' if cfg.only_test else 'train'
cfg.exp_note = '+'.join([cfg.model_name, cfg.train_test_split, cfg.core_task, cfg.extra])

# visualization
cfg.draw_fig = False
cfg.draw_fig_interval_epoch = 10  # epoch interval of loss curves
cfg.is_colorful = True

# train about
cfg.pre_know_real_match = False
cfg.max_epoch = 200
cfg.batch_size = 1  # batch size only support 1
cfg.train_learning_rate = 1e-6
cfg.train_dropout_prob = 0
cfg.weight_decay = 1e-4
# cfg.lr_plan = {500: 1e-7, 800: 1e-8, 1100: 1e-10}
cfg.train_pin_memory = True

# continue training setting
cfg.iscontinue = False
cfg.continue_path = ''
cfg.continue_dir = ''  # not necessary

# loss coefficient
cfg.xy_ratio = 0.2
cfg.r_ratio = 0.2
cfg.reid_ratio = 1
cfg.consistency_xy_ratio = 0.1
cfg.consistency_r_ratio = 0.1
cfg.target_xy_ratio = 0.2
cfg.target_r_ratio = 0.2

# random blur
cfg.target_r_blur = False  # not use
cfg.target_r_blur_bound = 3  # ±3 degree

# dataset about
cfg.dataset_shuffle = False  # set false for CSRD, true for CSRD_O
cfg.train_random_seed = 0
cfg.train_num = 800  # 800 for CSRD, 500 for CSRD_O
cfg.test_num = 200   # 200 for CSRD, 50  for CSRD_O
cfg.gt_ratio = 0.212  # 0.212 for CSRD, 1 for CSRD-O

# trian about
# dis_pesudo_ratio + r_pesudo_ratio = 1.0
cfg.dis_pseudo_ratio = 0.5
cfg.sim_matrix_gt_ratio = 0.5  # sim matrix gt ratio
cfg.matrix_threshold_train = 0.32
cfg.matrix_threshold_test = 0.24
cfg.distance_threshold_train = 2.0
cfg.distance_threshold_test = 5.0
cfg.reid_f1_threshold = 0.7
cfg.view_num = 5
cfg.match_target_topK = 1  # match target topK
cfg.graph_search = 'bfs'  # bfs or dfs

# loconet correct
cfg.bias_correct = False  # set true for CSRD_O, false for CSRD
cfg.x_bias = -1.7579
cfg.y_bias = -0.4667

# freeze resnet
cfg.freeze_resnet_detect = False  # not use
cfg.freeze_resnet = False

# whether loading pretrained loconet
cfg.without_load_model = True

cfg.load_loconet_model = True
cfg.loconet_pretrained_model_path = ('/home/nv/ws/GO-MVF/models/CSRD_test.pth')

cfg.load_resnet_model = True
cfg.resnet_pretrained_model_path = ('/home/nv/ws/GO-MVF/models/CSRD_test.pth')

train_monoreid_net_with_xy_and_r_swarm(cfg)
