import torch.optim as optim
from nets import reidnet, mono21
from dataset.dataset import *
from utils.utils import *
from utils.draw_fig import *
from itertools import chain
import random
from collections import defaultdict
from utils.my_draw_fig import PlotCurvesGeneral
import copy

rgb_table = [[191, 36, 42], [255, 70, 31], [255, 181, 30], [23, 133, 170],
             [22, 169, 81], [255, 242, 223], [0, 52, 115], [255, 0, 255],
             [254, 71, 119], [0, 100, 0], [189, 221, 34], [163, 226, 197],
             [62, 237, 232], [0, 191, 255], [186, 202, 199], [204, 164, 227],
             [87, 0, 79], [205, 92, 92], [0, 0, 255], [255, 0, 0], [0, 255, 0],
             [77, 34, 26], [254, 241, 67], [132, 90, 50], [65, 85, 92], [119, 32, 55]]

def train_monoreid_net_with_xy_and_r_swarm(cfg, training_set=None, validation_set=None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    cfg.fps_dict = torch.load(os.path.join(cfg.label,
                                           'fps.pth'))  # f_pids_dict[int(frame_id)][int(p_id)] = [x, y, r]
    cfg.fp_dict = torch.load(
        os.path.join(cfg.label, 'fp.pth'))  # f_pid_dict[f"{frame_id}_{p_id}"] = [x, y, r]

    # Reading dataset
    dataset = return_more_view_dataset_mvf(cfg)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Set data position
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Build model and optimizer
    model_mono = mono21.mono(model=cfg.loconet_pretrained_model_path,
                             without_load_model=cfg.without_load_model)  # loconet
    if cfg.dataset_name == 'CSRD_O':
        model_mono.kk = [[886.8100, 0.000000, 512.0],
                   [0.000000, 886.8100, 384.0],
                   [0.0, 0.0, 1.0]]  # camera intrinsic for CSRD-O
        model_mono.kk = torch.tensor(model_mono.kk, device=device)
    model_reid = reidnet.reidnet()  # resnet-50

    ## choosing model ##

    if cfg.iscontinue:
        assert cfg.continue_path != ''
        model_mono.loadmodel(cfg.continue_path)
        model_reid.loadmodel(cfg.continue_path)

    if cfg.load_resnet_model:
        model_reid.loadmodel(cfg.resnet_pretrained_model_path)

    if cfg.load_loconet_model:
        model_mono.loadmodel(cfg.loconet_pretrained_model_path)

    model_mono = model_mono.to(device=device)
    model_reid = model_reid.to(device=device)
    model_reid.train()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, chain(model_mono.parameters(), model_reid.parameters())),
                           lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    start_epoch = 1
    if cfg.iscontinue:
        state = torch.load(cfg.continue_path)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

    train = _train_monoreid_cam_gt_centroid_swarm
    test = _test_monoreid_cam_gt_centroid_soft_choose_swarm

    # recording loss curves and best model result
    train_total_loss = []
    train_reid_loss = []
    train_cam_xy_loss = []
    train_cam_r_loss = []
    train_consistency_xy_loss = []
    train_consistency_r_loss = []
    train_target_xy_loss = []
    train_target_r_loss = []

    test_total_loss = []
    test_reid_loss = []
    test_cam_xy_loss = []
    test_cam_r_loss = []
    test_consistency_xy_loss = []
    test_consistency_r_loss = []
    test_target_xy_loss = []
    test_target_r_loss = []

    best_loss = 999999
    best_loss_info = ''
    high_f1_loss = 999999
    high_f1_info = ''

    plotter_train = PlotCurvesGeneral()
    plotter_test = PlotCurvesGeneral()

    if cfg.test_before_train or cfg.only_test:
        test_info = test(dataset, model_mono, model_reid, device, 0, cfg, draw_heat=True, draw_bbox=False)
        print_log_info('Test', cfg.log_path, test_info)
        torch.cuda.empty_cache()

        if cfg.only_test:
            return

        test_target_xy_loss.append(test_info['target_xy_loss'])
        test_target_r_loss.append(test_info['target_r_loss'])
        test_cam_xy_loss.append(test_info['cam_xy_loss'])
        test_cam_r_loss.append(test_info['cam_r_loss'])
        test_consistency_xy_loss.append(test_info['consistency_xy_loss'])
        test_consistency_r_loss.append(test_info['consistency_r_loss'])
        test_reid_loss.append(test_info['re_id_loss'])
        test_total_loss.append(test_info['total_loss'])

    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):

        # if epoch in cfg.lr_plan:
        #     adjust_lr(optimizer, cfg.lr_plan[epoch])
        if cfg.freeze_resnet and cfg.freeze_resnet_detect == False:
            model_reid.eval()
            for param in model_reid.parameters():
                param.requires_grad = False
            optimizer = optim.Adam(model_mono.parameters(),
                lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
            print_log(cfg.log_path, 'freeze resnet at epoch %d' % (epoch))
            cfg.freeze_resnet = False
        # print(next(model_reid.parameters()).requires_grad)

        train_info = train(dataset, model_mono, model_reid, device, optimizer, epoch, cfg)
        print_log_info('Train', cfg.log_path, train_info)

        train_target_xy_loss.append(train_info['target_xy_loss'])
        train_target_r_loss.append(train_info['target_r_loss'])
        train_cam_xy_loss.append(train_info['cam_xy_loss'])
        train_cam_r_loss.append(train_info['cam_r_loss'])
        train_consistency_xy_loss.append(train_info['consistency_xy_loss'])
        train_consistency_r_loss.append(train_info['consistency_r_loss'])
        train_reid_loss.append(train_info['re_id_loss'])
        train_total_loss.append(train_info['total_loss'])

        # Save model
        if epoch % cfg.save_model_interval_epoch == 0:
            state = {
                'epoch': epoch,
                'state_dict_mono': model_mono.state_dict(),
                'state_dict_reid': model_reid.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filepath = cfg.result_path + '/epoch%d.pth' % (epoch)
            torch.save(state, filepath)
            print('model saved to:', filepath)

        # test
        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(dataset, model_mono, model_reid, device, epoch, cfg, draw_heat=True, draw_bbox=False)
            # show_monoreid_epoch_test_info('Test', cfg.log_path, test_info)
            print_log_info('Test', cfg.log_path, test_info)
            torch.cuda.empty_cache()

            if test_info['total_loss'] < best_loss:
                best_loss = test_info['total_loss']
                best_loss_info = test_info

            # show best loss
            if best_loss_info != '':
                print_log(cfg.log_path, '')
                print_log(cfg.log_path, '====> best_loss_info:')
                print_log_info('Best Test', cfg.log_path, best_loss_info)

            test_target_xy_loss.append(test_info['target_xy_loss'])
            test_target_r_loss.append(test_info['target_r_loss'])
            test_cam_xy_loss.append(test_info['cam_xy_loss'])
            test_cam_r_loss.append(test_info['cam_r_loss'])
            test_consistency_xy_loss.append(test_info['consistency_xy_loss'])
            test_consistency_r_loss.append(test_info['consistency_r_loss'])
            test_reid_loss.append(test_info['re_id_loss'])
            test_total_loss.append(test_info['total_loss'])

            if cfg.freeze_resnet_detect:
                if len(test_reid_loss) > 1 and test_reid_loss[-1] > test_reid_loss[-2] * 1.1:
                    cfg.freeze_resnet = True  # 冻结resnet
                    cfg.freeze_resnet_detect = False

        # visualization of loss curves
        if cfg.draw_fig and epoch % cfg.draw_fig_interval_epoch == 0:
            filepath = cfg.result_path

            draw_line_fig(train_total_loss, filepath, is_train=True, is_timestample=False, extra='total')
            draw_line_fig(train_reid_loss, filepath, is_train=True, is_timestample=False, extra='reid')
            draw_line_fig(train_cam_xy_loss, filepath, is_train=True, is_timestample=False, extra='cam_xy')
            draw_line_fig(train_cam_r_loss, filepath, is_train=True, is_timestample=False, extra='cam_angle')
            draw_line_fig(train_consistency_xy_loss, filepath, is_train=True, is_timestample=False, extra='consistency_xy')
            draw_line_fig(train_consistency_r_loss, filepath, is_train=True, is_timestample=False, extra='consistency_r')
            draw_line_fig(train_target_xy_loss, filepath, is_train=True, is_timestample=False, extra='target_xy')
            draw_line_fig(train_target_r_loss, filepath, is_train=True, is_timestample=False, extra='target_r')

            draw_line_fig(test_total_loss, filepath, is_train=False, is_timestample=False, extra='total')
            draw_line_fig(test_reid_loss, filepath, is_train=False, is_timestample=False, extra='reid')
            draw_line_fig(test_cam_xy_loss, filepath, is_train=False, is_timestample=False, extra='cam_xy')
            draw_line_fig(test_cam_r_loss, filepath, is_train=False, is_timestample=False, extra='cam_angle')
            draw_line_fig(test_consistency_xy_loss, filepath, is_train=False, is_timestample=False, extra='consistency_xy')
            draw_line_fig(test_consistency_r_loss, filepath, is_train=False, is_timestample=False, extra='consistency_r')
            draw_line_fig(test_target_xy_loss, filepath, is_train=False, is_timestample=False, extra='target_xy')
            draw_line_fig(test_target_r_loss, filepath, is_train=False, is_timestample=False, extra='target_r')

        # draw loss for whole train process
        dir_path = os.path.join(cfg.result_path, f"figs")
        plotter_train.register('train_cam_xy_loss', train_cam_xy_loss)
        plotter_train.register('train_cam_r_loss', train_cam_r_loss)
        plotter_train.register('train_consistency_xy_loss', train_consistency_xy_loss)
        plotter_train.register('train_consistency_r_loss', train_consistency_r_loss)
        plotter_train.register('train_target_xy_loss', train_target_xy_loss)
        plotter_train.register('train_target_r_loss', train_target_r_loss)
        plotter_train.register('train_reid_loss', train_reid_loss)
        plotter_train.register('train_total_loss', train_total_loss)
        plotter_train.plot_all(dir_path, 'combine_train_loss')

        plotter_test.register('test_cam_xy_loss', test_cam_xy_loss)
        plotter_test.register('test_cam_r_loss', test_cam_r_loss)
        plotter_test.register('test_consistency_xy_loss', test_consistency_xy_loss)
        plotter_test.register('test_consistency_r_loss', test_consistency_r_loss)
        plotter_test.register('test_target_xy_loss', test_target_xy_loss)
        plotter_test.register('test_target_r_loss', test_target_r_loss)
        plotter_test.register('test_reid_loss', test_reid_loss)
        plotter_test.register('test_total_loss', test_total_loss)
        plotter_test.plot_all(dir_path, 'combine_test_loss')

# 5 view train
def _train_monoreid_cam_gt_centroid_swarm(dataset, model_mono, model_reid, device, optimizer, epoch, cfg):
    cfg.epoch = epoch
    if cfg.matrix_threshold_train > 0.5:
        matrix_threshold = cfg.matrix_threshold_train / 2
    else:
        matrix_threshold = cfg.matrix_threshold_train

    distance_threshold = cfg.distance_threshold_train
    epoch_timer = Timer()

    loss_meter = AverageMeterGeneral()
    plotter = PlotCurvesGeneral()
    hit_total_one_meter = HitProbabilityMeter_Monoreid()
    hit_total_mean_meter = HitProbabilityMeter_Monoreid()
    camera_meter = HitProbabilityMeter_Monoreid()

    # loss
    re_id_loss_frame = []
    cam_xy_loss_frame = []
    cam_r_loss_frame = []
    total_loss_frame = []
    consistency_xy_loss_frame = []
    consistency_r_loss_frame = []
    target_xy_loss_frame = []
    target_r_loss_frame = []

    ####        setting train and eval mode     ####
    model_mono.train()
    for m in model_mono.net.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    model_reid.train()

    # generate shuffled frame indices
    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)
    if cfg.dataset_shuffle:
        shuffled_frame_indices = torch.randperm(cfg.train_num+cfg.test_num, generator=g).tolist()
    else:
        shuffled_frame_indices = list(range(cfg.train_num+cfg.test_num))

    # random degree
    fp_dict = copy.deepcopy(cfg.fp_dict)
    if cfg.target_r_blur:
        if cfg.dataset_name == 'CSRD_O':
            for frame_id in range(1, cfg.train_num+1):
                for p_id in cfg.fps_dict[frame_id].keys():
                    new_r = float(fp_dict[f"{frame_id}_{p_id}"][2]) + np.random.uniform(-cfg.target_r_blur_bound, cfg.target_r_blur_bound)
                    new_r = new_r % 360
                    fp_dict[f"{frame_id}_{p_id}"][2] = str(new_r)

    # for frame_idx in range(cfg.train_num):
    for shuffled_i, frame_idx in enumerate(shuffled_frame_indices):
        if shuffled_i >= cfg.train_num:
            continue
        print(f'processing train case {shuffled_i} (original frame {frame_idx})')
        optimizer.zero_grad()
        draw_gt_id_xyangle_dict_total = {}  # {person_id: [x, y, angle]}

        top_pred_xy_total_list = []
        top_pred_r_total_list = []
        bbox_total_list = []

        # 存第frame_idx帧
        original_xz_list = []
        original_angle_list = []
        bbox_list = []  # [[], [], [], [], []]
        original_xz_total = []
        original_angle_total = []
        view_reid_feature_list = []
        list_id_total_list = []

        person_num_list = []
        x_gt_list = []
        y_gt_list = []
        r_gt_list = []
        for view_idx in range(len(dataset)):
            batch_data = dataset[view_idx][frame_idx]

            # prepare batch data
            batch_data = [try_to(b.unsqueeze(dim=0), device) if b is not None and b.numel() > 0 else None for b in
                          batch_data]

            #### [VTM]       loconet forward        ####
            output_dict1 = model_mono(batch_data[4].squeeze(dim=0), batch_data[5].squeeze(dim=0))
            original_xz = output_dict1['xz_pred']
            original_angle = output_dict1['angles']

            if cfg.bias_correct:
                for i in range(len(original_xz)):
                    original_xz[i][0] -= cfg.x_bias
                    original_xz[i][1] -= cfg.y_bias

            pi = torch.tensor(math.pi, device=original_angle[0].device)
            bbox = [[int(batch_data[9][0][1]), bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for
                    bbox_score in
                    batch_data[5].squeeze(dim=0).cpu().tolist()]

            original_xz_list.append(original_xz)
            original_angle_list.append(original_angle)
            bbox_list.append(bbox)

            ####        reid forward        ####
            reid_output1 = model_reid(batch_data[2].squeeze(dim=0))

            view_reid_feature_list.append(reid_output1)
            person_num_list.append(reid_output1.shape[0])

            id_used_here = batch_data[10].squeeze(dim=0).tolist()
            list_id_total_list.append(id_used_here)

            # creating camera x, y, r gt
            x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
            x_gt *= cfg.gt_ratio
            y_gt *= cfg.gt_ratio
            r_gt -= 90
            if r_gt > 180:
                r_gt = r_gt - 360
            r_gt /= (180/pi)

            x_gt_list.append(x_gt)
            y_gt_list.append(y_gt)
            r_gt_list.append(r_gt)

            # record ground truth coverage
            frame_id = int(batch_data[9][0][0].item())
            id1_used_here = batch_data[10].squeeze(dim=0).tolist()
            id1_set = set(id1_used_here)
            id_list = list(id1_set)

            gt_xyangle_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id_list, cfg.gt_ratio)  # Order matches id_list
            for index_draw_gt, id_draw_gt in enumerate(id_list):
                draw_gt_id_xyangle_dict_total[id_draw_gt] = gt_xyangle_list[
                    index_draw_gt]


        ############ finish 5 view ###########

        # calculate prefix sum of person number
        person_interval_list = []
        tmp_sum = 0
        for person in person_num_list:
            person_interval_list.append(tmp_sum)
            tmp_sum += person
        person_interval_list.append(tmp_sum)

        # all person id in 5 view
        id_total_list = [item for sublist in list_id_total_list for item in sublist]

        # calculate self similarity matrix
        total_feature_cat = torch.cat(view_reid_feature_list, dim=0)  # feature concatenation in person number dimension
        match_total_matrix = get_eu_distance_mtraix(total_feature_cat, total_feature_cat)  # calculate self similarity matrix, both row and column are M

        match_total_matrix_gt = torch.zeros_like(match_total_matrix)  # total self similarity matrix ground truth
        n, m = match_total_matrix.shape
        for i in range(n):
            for j in range(m):
                if id_total_list[i] == id_total_list[j]:
                    match_total_matrix_gt[i][j] = 1

        if cfg.pre_know_real_match:
            reid_val_mask = match_total_matrix_gt >= matrix_threshold  # decide based on sim threshold
        else:
            reid_val_mask = match_total_matrix >= matrix_threshold  # decide based on sim threshold

        # match between 5 views according to similarity
        mask = reid_val_mask.clone()
        mask_up = torch.triu(mask, 1)  # use upper triangle matrix

        # remove self matrix mask
        person_num_sum = 0
        for person_num in person_num_list:
            mask_up[person_num_sum:person_num_sum + person_num,
            person_num_sum:person_num_sum + person_num] = False  # set self matrix mask as False
            person_num_sum += person_num

        # search if there is a chain of camera pose calculation from other cameras to camera 1
        # Step 1:  person_index -> camera_id mapping
        num_cameras = len(person_num_list)
        person_to_camera = []
        for cam_id, person_num in enumerate(person_num_list):
            person_to_camera.extend([cam_id] * person_num)  # map each person index to camera id (0~4)

        # Step 2: Iterate over all True entries in mask_up (i.e., valid cross-view matches) to build an inter-camera connectivity graph and pedestrian index correspondences.
        #  { (c1, c2): [(person i, person j), ...] }  c1 < c2
        camera_match_pairs = defaultdict(list)
        N = mask_up.shape[0]
        for i in range(N):
            for j in range(i + 1, N):  # because mask_up is triu(1), j > i
                if mask_up[i, j].item():
                    cam_i = person_to_camera[i]
                    cam_j = person_to_camera[j]
                    if cam_i == cam_j:
                        continue

                    # make sure key is (min, max)
                    c1, c2 = min(cam_i, cam_j), max(cam_i, cam_j)
                    similarity = match_total_matrix[i][j]
                    # when camera order is swapped, pedestrian index order also needs to be swapped
                    if cam_i == c1 and cam_j == c2:
                        camera_match_pairs[(c1, c2)].append((i, j, similarity))
                    else:
                        camera_match_pairs[(c1, c2)].append((j, i, similarity))

        # Step 3: Initialize a Union-Find data structure for cameras, then iterate over all camera match pairs to merge connected components.
        cam_uf = CameraUnionFind(num_cameras)
        for (c1, c2), pairs_list in camera_match_pairs.items():
            # if there is at least one match pair between c1 and c2, then merge them
            cam_uf.union(c1, c2)

        # Step 4: Check if camera 0 is connected to all other cameras (1~4)
        root_cam0 = cam_uf.find(0)
        cam_all_connected_flag = all(cam_uf.find(cam_id) == root_cam0 for cam_id in range(1, num_cameras))

        # Step 5(Option): Output camera connected components for debugging
        camera_components = defaultdict(set)
        for cam_id in range(num_cameras):
            root = cam_uf.find(cam_id)
            camera_components[root].add(cam_id)
        camera_groups = list(camera_components.values())

        # if camera 2~5 are all connected to camera 0, then the similarity threshold is set properly
        if cam_all_connected_flag:
            # print('satisfy the joint optimization condition')
            pass
        else:
            print("There is at least one camera that is not connected to camera 0. skip: ", frame_idx + 1)
            print("graph: ", [sorted(list(comp)) for comp in camera_groups])
            continue

        # based on the connected graph, optimize the relative pose of camera 2~5 to camera 1
        # order the pairs by similarity from high to low in camera_match_pairs
        for key in camera_match_pairs:
            camera_match_pairs[key].sort(key=lambda x: x[2], reverse=True)  # high to low

        original_xz_total = [item for sublist in original_xz_list for item in sublist]
        original_angle_total = [item for sublist in original_angle_list for item in sublist]

        # camera pose estimation method:
        # if camera j has common person with camera 0, then estimate T_0j directly, otherwise estimate T_0j indirectly
        # camera j to camera 0: T_0j = (dx, dy, theta)
        T0j_dict = {}  # {1: T_01, 2: T_02, ...}
        direct_estimates = {}  # key: camera_id, value: T_0j = (dx, dy, theta)

        # make camera graph according to camera_match_pairs
        if cfg.graph_search == 'bfs':
            estimation_order = get_pose_estimation_order_bfs(camera_match_pairs, start_cam=0)
        elif cfg.graph_search == 'dfs':
            estimation_order = get_pose_estimation_order_dfs(camera_match_pairs, start_cam=0)

        for camera_match_pair in estimation_order:
            c1, c2 = camera_match_pair
            # if camera c2 has common person with camera 0, then estimate T_0j directly
            if c1 == 0:
                T_direct = estimate_T_0j_direct(c2, camera_match_pairs, original_xz_total, original_angle_total, topK=cfg.match_target_topK)
                direct_estimates[c2] = T_direct
                T0j_dict[c2] = T_direct

            # if camera c2 has no common person with camera 0, then estimate T_0j indirectly
            else:
                # find the camera that has direct estimate
                if c1 in direct_estimates:
                    T_indrect = estimate_T_0j_indirect_specify(c2, c1, camera_match_pairs, num_cameras, T0j_dict,
                                                               person_to_camera,
                                                               original_xz_total, original_angle_total, topK=cfg.match_target_topK)
                    direct_estimates[c2] = T_indrect
                    T0j_dict[c2] = T_indrect
                elif c2 in direct_estimates:
                    T_indrect = estimate_T_0j_indirect_specify(c1, c2, camera_match_pairs, num_cameras, T0j_dict,
                                                               person_to_camera,
                                                               original_xz_total, original_angle_total, topK=cfg.match_target_topK)
                    direct_estimates[c1] = T_indrect
                    T0j_dict[c1] = T_indrect
                else:
                    print(f'camera {c1} and camera {c2} have no common person with camera 0, skip')
                    print('estimation_order:', estimation_order)
                    print('camera_match_pairs:', camera_match_pairs)

        # check
        assert len(T0j_dict) == num_cameras - 1, "not all cameras have pose estimation"

        # cam_loss
        cam_xy_loss = 0
        cam_r_loss = 0
        for j in range(1, num_cameras):
            x_gt, y_gt, r_gt = x_gt_list[j], y_gt_list[j], r_gt_list[j]
            x_pred, y_pred, r_pred = T0j_dict[j]

            r_pred = r_pred * (180/pi)
            if r_pred < 0:
                r_pred += 360
            r_pred = 360 - r_pred
            r_pred -= 90
            if r_pred > 180:
                r_pred = r_pred - 360
            r_pred = r_pred / (180/pi)

            cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
            cam_r_loss += angular_loss(r_gt, r_pred)
            camera_meter.update([[x_pred, y_pred, r_pred]], [[x_gt, y_gt, r_gt]], 0, 0)

        cam_xy_loss = cam_xy_loss / (num_cameras - 1)
        cam_r_loss = cam_r_loss / (num_cameras - 1)

        # transform pedestrian pose from camera j to camera 1
        for j in range(0, num_cameras):
            if j == 0:
                original_xz = original_xz_list[0]
                original_angle = original_angle_list[0]
                for k in range(len(original_xz)):
                    xz = original_xz[k]
                    angle = original_angle[k]
                    top_pred_xy_total_list.append(torch.stack([xz[0], xz[1]]))
                    top_pred_r_total_list.append(angle[0])

                bbox_total_list += bbox_list[0]
            else:
                original_xz = original_xz_list[j]
                original_angle = original_angle_list[j]
                delta_x, delta_y, delta_theta = T0j_dict[j]
                local_to_global_tf = torch.stack([delta_x, delta_y, delta_theta])

                for k in range(len(original_xz)):
                    xz = original_xz[k]
                    angle = original_angle[k]
                    local_pose = torch.stack([xz[0], xz[1], angle[0]])
                    new_xzr_pred = transform_pose_local_to_global(local_pose, local_to_global_tf)
                    top_pred_xy_total_list.append(new_xzr_pred[:2])
                    top_pred_r_total_list.append(new_xzr_pred[2])

                bbox_total_list += bbox_list[j]

        # compute distance and angle difference matrix
        distance_total_matrix = torch.zeros_like(match_total_matrix)
        r_total_matrix = torch.zeros_like(match_total_matrix)
        # match_total_matrix_gt = torch.zeros_like(match_total_matrix)
        n, m = distance_total_matrix.shape
        for i in range(n):
            for j in range(m):
                # if id_total_list[i] == id_total_list[j]:
                #     match_total_matrix_gt[i][j] = 1
                distance_total_matrix[i][j] = math.sqrt(
                    (top_pred_xy_total_list[i][0].item() - top_pred_xy_total_list[j][0].item()) ** 2 + (
                            top_pred_xy_total_list[i][1].item() - top_pred_xy_total_list[j][1].item()) ** 2)
                r_total_matrix[i][j] = clac_rad_distance(top_pred_r_total_list[i],
                                                               top_pred_r_total_list[j]).item()

        distance_total_matrix[distance_total_matrix < 0.0002] += 0.0001
        r_total_matrix[r_total_matrix < 0.0002] += 0.0001
        matrix_dis_pseudo = 1 / distance_total_matrix
        matrix_r_pseudo = 1 / r_total_matrix

        # processing distance matrix
        for i in range(n):
            line_max = matrix_dis_pseudo[i].max()
            line_min = matrix_dis_pseudo[i].min()
            size = line_max - line_min
            for j in range(m):
                matrix_dis_pseudo[i][j] = (matrix_dis_pseudo[i][j] - line_min) / size

        # processing angle matrix
        for i in range(n):
            line_max = matrix_r_pseudo[i].max()
            line_min = matrix_r_pseudo[i].min()
            size = line_max - line_min
            for j in range(m):
                matrix_r_pseudo[i][j] = (matrix_r_pseudo[i][j] - line_min) / size
        matrix_r_pseudo[matrix_r_pseudo < 0.9] = 0
        matrix_pseudo = cfg.dis_pseudo_ratio * matrix_dis_pseudo + (1 - cfg.dis_pseudo_ratio) * matrix_r_pseudo

        match_total_matrix_label = cfg.sim_matrix_gt_ratio * match_total_matrix_gt + (1 - cfg.sim_matrix_gt_ratio) * matrix_pseudo
        re_id_matrix_loss = torch.nn.MSELoss()(match_total_matrix, match_total_matrix_label)

        ###################### aggregate according to similarity and distance ######################
        dis_mask = distance_total_matrix <= distance_threshold
        if cfg.pre_know_real_match:
            reid_val_mask = match_total_matrix_gt >= matrix_threshold
        else:
            reid_val_mask = match_total_matrix >= matrix_threshold
        mask = torch.logical_and(dis_mask, reid_val_mask)  # position with True satisfy both distance and similarity threshold
        mask_up = torch.triu(mask, 1)

        # remove self matrix mask
        person_num_sum = 0
        for person_num in person_num_list:
            mask_up[person_num_sum:person_num_sum + person_num,
            person_num_sum:person_num_sum + person_num] = False
            person_num_sum += person_num

        union_find = UnionFind(len(id_total_list))

        for i in range(len(id_total_list)):
            for j in range(len(id_total_list)):
                if mask_up[i][j]:
                    # pair_dict[i].append([j, match_total_matrix[i][j]])
                    union_find.union(i, j)
        union_collection_dict = {i: [] for i in range(len(id_total_list))}
        for i in range(len(id_total_list)):
            union_collection_dict[union_find.find(i)].append(i)
        aggregated_list = []
        for key, val in union_collection_dict.items():
            if len(val) > 1:
                aggregated_list.append(val)

        # using sub-graph algorithm
        pop_index_list = []
        for index, aggregated_sub_list in enumerate(aggregated_list):
            split_list = []
            person_interval_counter = [[] for i in range(len(person_interval_list) - 1)]
            for i in range(len(person_interval_list) - 1):  # 0~4
                for j in range(len(aggregated_sub_list)):
                    if person_interval_list[i] <= aggregated_sub_list[j] < person_interval_list[i + 1]:
                        person_interval_counter[i].append(aggregated_sub_list[j])
            person_counter_max = max(
                [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
            if person_counter_max > 1:
                # split
                while person_counter_max > 1:
                    for i in range(len(person_interval_counter)):
                        tmp_split = []
                        if len(person_interval_counter[i]) > 0:
                            pivot = person_interval_counter[i][0]
                            tmp_split.append(person_interval_counter[i].pop(0))
                            for j in range(i + 1, len(person_interval_counter)):
                                score_list = [match_total_matrix[pivot][elem_index] for elem_index in
                                              person_interval_counter[j]]
                                if len(score_list) == 0:
                                    continue
                                if max(score_list) < cfg.matrix_threshold_test:
                                    break
                                else:
                                    max_index = score_list.index(max(score_list))
                                    tmp_split.append(person_interval_counter[j].pop(max_index))
                        if tmp_split != []:
                            aggregated_list.append(tmp_split)

                    person_counter_max = max(
                        [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                    # collecting splited list
                # collecting remaining list
                pop_index_list.append(index)
                remaining_list = []
                for i in range(len(person_interval_counter)):
                    for j in range(len(person_interval_counter[i])):
                        remaining_list.append(person_interval_counter[i][j])
                if remaining_list != []:
                    aggregated_list.append(remaining_list)
            else:
                pass
        aggregated_list_new = []
        for i in range(len(aggregated_list)):
            if i in pop_index_list:
                continue
            else:
                aggregated_list_new.append(aggregated_list[i])
        aggregated_list = aggregated_list_new

        ###################### aggregate end ######################

        ###################### metric calculate ######################
        index_used_set = set()
        total_index_set = set([i for i in range(len(id_total_list))])

        # mean
        final_mean_xy_total_list = []
        final_mean_r_total_list = []

        # centroid 1 choosing
        final_only_one_xy_total_list = []
        final_only_one_r_total_list = []
        final_only_one_bbox_total_list = []

        final_xy_total_gt_list = []
        final_r_total_gt_list = []
        final_id_total_list = []

        # aggregated more view points here
        consistency_xy_loss = torch.tensor(0.0, device=device)
        consistency_r_loss = torch.tensor(0.0, device=device)
        num_groups = 0
        for aggregated_sub_list in aggregated_list:
            x_sum = 0
            y_sum = 0
            r_sum = 0
            # used to caculate centroid
            x_cache = []
            y_cache = []
            r_cache = []

            for index_sub in range(len(aggregated_sub_list)):
                index_used_set.add(aggregated_sub_list[index_sub])

                x_cache.append(top_pred_xy_total_list[aggregated_sub_list[index_sub]][0])
                y_cache.append(top_pred_xy_total_list[aggregated_sub_list[index_sub]][1])
                r_cache.append(top_pred_r_total_list[aggregated_sub_list[index_sub]])

                x_sum += top_pred_xy_total_list[aggregated_sub_list[index_sub]][0]
                y_sum += top_pred_xy_total_list[aggregated_sub_list[index_sub]][1]
                r_sum += top_pred_r_total_list[aggregated_sub_list[index_sub]]
            x_mean = x_sum / len(aggregated_sub_list)
            y_mean = y_sum / len(aggregated_sub_list)
            r_mean = r_sum / len(aggregated_sub_list)

            # cross-view consistency loss
            for index_sub in range(len(aggregated_sub_list)):
                consistency_xy_loss += torch.abs(top_pred_xy_total_list[aggregated_sub_list[index_sub]][0] - x_mean)
                consistency_xy_loss += torch.abs(top_pred_xy_total_list[aggregated_sub_list[index_sub]][1] - y_mean)
                consistency_r_loss += angular_loss(r_mean, top_pred_r_total_list[aggregated_sub_list[index_sub]])
            num_groups += 1

            # centroid about
            to_centroid_distance_list = [math.sqrt((x_cache[i] - x_mean) ** 2 + (y_cache[i] - y_mean) ** 2) for i in
                                         range(len(x_cache))]
            rank_list = torch.argsort(torch.tensor(to_centroid_distance_list)).tolist()
            rank1_index = rank_list[0]

            total1_index = aggregated_sub_list[rank1_index]

            final_only_one_xy_total_list.append(top_pred_xy_total_list[total1_index])
            final_only_one_r_total_list.append(top_pred_r_total_list[total1_index])
            final_only_one_bbox_total_list.append(
                [bbox_total_list[aggregated_sub_list[i]] for i in range(len(aggregated_sub_list))])

            if r_mean > math.pi:
                r_mean = r_mean - 2 * math.pi
            if r_mean < -math.pi:
                r_mean = 2 * math.pi + r_mean
            final_mean_xy_total_list.append(torch.stack([x_mean, y_mean]))
            final_mean_r_total_list.append(r_mean)

            id_used_here = id_total_list[aggregated_sub_list[0]]

            final_id_total_list.append(id_used_here)
            final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])
            final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])
        # aggregated more view points here
        # aggregated single points here
        single_index_list = sorted(list(total_index_set - index_used_set))

        for single_index in single_index_list:
            id_used_here = id_total_list[single_index]

            final_only_one_xy_total_list.append(top_pred_xy_total_list[single_index])
            final_only_one_r_total_list.append(top_pred_r_total_list[single_index])
            final_only_one_bbox_total_list.append([bbox_total_list[single_index]])

            final_mean_xy_total_list.append(top_pred_xy_total_list[single_index])
            final_mean_r_total_list.append(top_pred_r_total_list[single_index])

            final_id_total_list.append(id_used_here)
            final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])
            final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])

        # aggregated single points here

        statistic_counter_dict = {id: [] for id in list(set(id_total_list))}
        for index_val, id_val in enumerate(id_total_list):
            statistic_counter_dict[id_val].append(index_val)
        # sort list
        for key in statistic_counter_dict.keys():
            statistic_counter_dict[key].sort()
        gt_aggregated_list = [val for val in statistic_counter_dict.values()]
        # statistic every person's probability
        prob_counter = 0
        for i in range(len(aggregated_list)):
            if aggregated_list[i] in gt_aggregated_list:
                prob_counter += 1

        hit_total_one_meter.update([[final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1],
                                     final_only_one_r_total_list[i]] for i in
                                    range(len(final_only_one_xy_total_list))],
                                   [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1],
                                     final_r_total_gt_list[i]] for i in range(len(final_xy_total_gt_list))], 0, 0)
        hit_total_mean_meter.update(
            [[final_mean_xy_total_list[i][0], final_mean_xy_total_list[i][1], final_mean_r_total_list[i]] for i in
             range(len(final_mean_xy_total_list))],
            [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1], final_r_total_gt_list[i]] for i in
             range(len(final_xy_total_gt_list))], 0, 0)
        ###################### metric calculate end ######################
        # BEV target loss
        target_xy_loss = cal_target_xy_loss(final_only_one_xy_total_list, final_xy_total_gt_list, loss_type='l2')
        target_r_loss = cal_target_r_loss(final_only_one_r_total_list, final_r_total_gt_list)

        # cross-view consistency loss
        if num_groups > 0:
            consistency_xy_loss /= num_groups
            consistency_r_loss /= num_groups

        # getting the total loss
        total_loss = (cfg.reid_ratio * re_id_matrix_loss + cfg.xy_ratio * cam_xy_loss + cfg.r_ratio * cam_r_loss
                      + cfg.consistency_xy_ratio * consistency_xy_loss + cfg.consistency_r_ratio * consistency_r_loss
                      + cfg.target_xy_ratio * target_xy_loss + cfg.target_r_ratio * target_r_loss)

        # loss
        re_id_loss_frame.append(re_id_matrix_loss.item())
        cam_xy_loss_frame.append(cam_xy_loss.item())
        cam_r_loss_frame.append(cam_r_loss.item())
        total_loss_frame.append(total_loss.item())
        consistency_xy_loss_frame.append(consistency_xy_loss.item())
        consistency_r_loss_frame.append(consistency_r_loss.item())
        target_xy_loss_frame.append(target_xy_loss.item())
        target_r_loss_frame.append(target_r_loss.item())

        loss_meter.update('target_xy_loss', target_xy_loss.item())
        loss_meter.update('target_r_loss', target_r_loss.item())
        loss_meter.update('cam_xy_loss', cam_xy_loss.item())
        loss_meter.update('cam_r_loss', cam_r_loss.item())
        loss_meter.update('consistency_xy_loss', consistency_xy_loss.item())
        loss_meter.update('consistency_r_loss', consistency_r_loss.item())
        loss_meter.update('re_id_loss', re_id_matrix_loss.item())
        loss_meter.update('total_loss', total_loss.item())

        total_loss.backward()
        optimizer.step()

    if cfg.draw_train_loss_fig and (epoch == 1 or epoch % cfg.draw_train_loss_epoch_interval == 0):
        dir_path = os.path.join(cfg.result_path, f"figs")
        plotter.register('train_cam_xy_loss', cam_xy_loss_frame)
        plotter.register('train_cam_r_loss', cam_r_loss_frame)
        plotter.register('train_consistency_xy_loss', consistency_xy_loss_frame)
        plotter.register('train_consistency_r_loss', consistency_r_loss_frame)
        plotter.register('train_target_xy_loss', target_xy_loss_frame)
        plotter.register('train_target_r_loss', target_r_loss_frame)
        plotter.register('train_reid_loss', re_id_loss_frame)
        plotter.register('train_total_loss', total_loss_frame)
        plotter.plot_all(dir_path, f'combine_train_loss_epoch_{epoch}')

    train_info = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'target_xy_loss': loss_meter.read('target_xy_loss'),
        'target_r_loss': loss_meter.read('target_r_loss'),
        'cam_xy_loss': loss_meter.read('cam_xy_loss'),
        'cam_r_loss': loss_meter.read('cam_r_loss'),
        'consistency_xy_loss': loss_meter.read('consistency_xy_loss'),
        'consistency_r_loss': loss_meter.read('consistency_r_loss'),
        're_id_loss': loss_meter.read('re_id_loss'),
        'total_loss': loss_meter.read('total_loss'),

        'cam_prob': camera_meter.get_xy_r_prob_dict(),
        'total_sub_prob_one': hit_total_one_meter.get_xy_r_prob_dict(),
        'total_sub_xy_one': hit_total_one_meter.get_xy_mean_error(),
        'total_sub_r_one': hit_total_one_meter.get_r_mean_error(),

        # 'total_sub_prob_mean': hit_total_mean_meter.get_xy_r_prob_dict(),
        # 'total_sub_xy_mean': hit_total_mean_meter.get_xy_mean_error(),
        # 'total_sub_r_mean': hit_total_mean_meter.get_r_mean_error(),
    }
    return train_info


# 5 view test
def _test_monoreid_cam_gt_centroid_soft_choose_swarm(dataset, model_mono, model_reid, device, epoch, cfg,
                                                     draw_heat=False, draw_bbox=False):
    cfg.epoch = epoch
    if cfg.matrix_threshold_test > 0.5:
        matrix_threshold = cfg.matrix_threshold_test / 2
    else:
        matrix_threshold = cfg.matrix_threshold_test

    distance_threshold = cfg.distance_threshold_test
    epoch_timer = Timer()

    loss_meter = AverageMeterGeneral()
    hit_total_one_meter = HitProbabilityMeter_Monoreid()
    hit_total_mean_meter = HitProbabilityMeter_Monoreid()
    camera_meter = HitProbabilityMeter_Monoreid()

    total_f1_sum = 0

    ####        setting train and eval mode     ####
    model_mono.eval()
    model_reid.eval()

    # generate shuffled frame indices
    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)
    if cfg.dataset_shuffle:
        shuffled_frame_indices = torch.randperm(cfg.train_num+cfg.test_num, generator=g).tolist()
    else:
        shuffled_frame_indices = list(range(cfg.train_num+cfg.test_num))

    with torch.no_grad():
        for shuffled_i, frame_idx in enumerate(shuffled_frame_indices):
            if shuffled_i < cfg.train_num:
                continue
            # if shuffled_i != 814:  # used for draw specific frame
            #     continue
            # if frame_idx != 432:  # used for draw specific frame
            #     continue
            print(f'processing test case {shuffled_i} (original frame {frame_idx})')

        # for frame_idx in range(cfg.train_num, cfg.train_num + cfg.test_num):
        #     print(f'processing test case {frame_idx}')

            draw_gt_id_xyangle_dict_total = {}  # {person_id: [x, y, angle]}

            top_pred_xy_total_list = []
            top_pred_r_total_list = []
            bbox_total_list = []

            # frame_idx
            original_xz_list = []
            original_angle_list = []
            bbox_list = []  # [[], [], [], [], []]
            original_xz_total = []
            original_angle_total = []
            view_reid_feature_list = []
            list_id_total_list = []

            person_num_list = []
            x_gt_list = []
            y_gt_list = []
            r_gt_list = []
            for view_idx in range(len(dataset)):
                batch_data = dataset[view_idx][frame_idx]

                # prepare batch data
                batch_data = [try_to(b.unsqueeze(dim=0), device) if b is not None and b.numel() > 0 else None for b in
                              batch_data]

                #### [VTM]       loconet forward        ####
                output_dict1 = model_mono(batch_data[4].squeeze(dim=0), batch_data[5].squeeze(dim=0))
                original_xz = output_dict1['xz_pred']
                original_angle = output_dict1['angles']

                if cfg.bias_correct:
                    for i in range(len(original_xz)):
                        original_xz[i][0] -= cfg.x_bias
                        original_xz[i][1] -= cfg.y_bias

                pi = torch.tensor(math.pi, device=original_angle[0].device)
                bbox = [[int(batch_data[9][0][1].item()), bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]]
                        for bbox_score in
                        batch_data[5].squeeze(dim=0).cpu().tolist()]

                original_xz_list.append(original_xz)
                original_angle_list.append(original_angle)
                bbox_list.append(bbox)

                ####        reid forward        ####
                reid_output1 = model_reid(batch_data[2].squeeze(dim=0))

                view_reid_feature_list.append(reid_output1)
                person_num_list.append(reid_output1.shape[0])

                id_used_here = batch_data[10].squeeze(dim=0).tolist()
                list_id_total_list.append(id_used_here)

                # creating camera x, y, r gt
                x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
                x_gt *= cfg.gt_ratio
                y_gt *= cfg.gt_ratio
                r_gt -= 90
                if r_gt > 180:
                    r_gt = r_gt - 360
                r_gt /= (180/pi)

                x_gt_list.append(x_gt)
                y_gt_list.append(y_gt)
                r_gt_list.append(r_gt)

                # record ground truth coverage
                fp_dict = cfg.fp_dict
                frame_id = int(batch_data[9][0][0].item())
                id1_used_here = batch_data[10].squeeze(dim=0).tolist()
                id1_set = set(id1_used_here)
                id_list = list(id1_set)

                gt_xyangle_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id_list, cfg.gt_ratio)  # Order matches id_list
                for index_draw_gt, id_draw_gt in enumerate(id_list):
                    draw_gt_id_xyangle_dict_total[id_draw_gt] = gt_xyangle_list[
                        index_draw_gt]

            ############ finish 5 view  ###########

            # calculate prefix sum of person number
            person_interval_list = []
            tmp_sum = 0
            for person in person_num_list:
                person_interval_list.append(tmp_sum)
                tmp_sum += person
            person_interval_list.append(tmp_sum)

            # all person id in 5 view
            id_total_list = [item for sublist in list_id_total_list for item in sublist]

            # calculate self similarity matrix
            total_feature_cat = torch.cat(view_reid_feature_list, dim=0)  # feature concatenation in person number dimension
            match_total_matrix = get_eu_distance_mtraix(total_feature_cat, total_feature_cat)  # calculate self similarity matrix, both row and column are M

            match_total_matrix_gt = torch.zeros_like(match_total_matrix)  # total self similarity matrix ground truth
            n, m = match_total_matrix.shape
            for i in range(n):
                for j in range(m):
                    if id_total_list[i] == id_total_list[j]:
                        match_total_matrix_gt[i][j] = 1

            if cfg.pre_know_real_match:
                reid_val_mask = match_total_matrix_gt >= matrix_threshold  # decide based on sim threshold
            else:
                reid_val_mask = match_total_matrix >= matrix_threshold  # decide based on sim threshold

            # match between 5 views according to similarity
            mask = reid_val_mask.clone()
            mask_up = torch.triu(mask, 1)  # use upper triangle matrix

            # remove self matrix mask
            person_num_sum = 0
            for person_num in person_num_list:
                mask_up[person_num_sum:person_num_sum + person_num,
                person_num_sum:person_num_sum + person_num] = False  # set self matrix mask as False
                person_num_sum += person_num

            # search if there is a chain of camera pose calculation from other cameras to camera 1
            # Step 1:  person_index -> camera_id mapping
            num_cameras = len(person_num_list)
            person_to_camera = []
            for cam_id, person_num in enumerate(person_num_list):
                person_to_camera.extend([cam_id] * person_num)  # map each person index to camera id (0~4)

            # Step 2: Iterate over all True entries in mask_up (i.e., valid cross-view matches) to build an inter-camera connectivity graph and pedestrian index correspondences.
            #  { (c1, c2): [(person i, person j), ...] }  c1 < c2
            camera_match_pairs = defaultdict(list)
            N = mask_up.shape[0]
            for i in range(N):
                for j in range(i + 1, N):  # because mask_up is triu(1), j > i
                    if mask_up[i, j].item():
                        cam_i = person_to_camera[i]
                        cam_j = person_to_camera[j]
                        if cam_i == cam_j:
                            continue

                        # make sure key is (min, max)
                        c1, c2 = min(cam_i, cam_j), max(cam_i, cam_j)
                        similarity = match_total_matrix[i][j]
                        # when camera order is swapped, pedestrian index order also needs to be swapped
                        if cam_i == c1 and cam_j == c2:
                            camera_match_pairs[(c1, c2)].append((i, j, similarity))
                        else:
                            camera_match_pairs[(c1, c2)].append((j, i, similarity))

            # Step 3: Initialize a Union-Find data structure for cameras, then iterate over all camera match pairs to merge connected components.
            cam_uf = CameraUnionFind(num_cameras)
            for (c1, c2), pairs_list in camera_match_pairs.items():
                # if there is at least one match pair between c1 and c2, then merge them
                cam_uf.union(c1, c2)

            # Step 4: Check if camera 0 is connected to all other cameras (1~4)
            root_cam0 = cam_uf.find(0)
            cam_all_connected_flag = all(cam_uf.find(cam_id) == root_cam0 for cam_id in range(1, num_cameras))

            # Step 5(Option): Output camera connected components for debugging
            camera_components = defaultdict(set)
            for cam_id in range(num_cameras):
                root = cam_uf.find(cam_id)
                camera_components[root].add(cam_id)
            camera_groups = list(camera_components.values())  # set

            # if camera 2~5 are all connected to camera 0, then the similarity threshold is set properly
            if cam_all_connected_flag:
                # print('satisfy the joint optimization condition')
                pass
            else:
                print("There is at least one camera that is not connected to camera 0. skip: ", frame_idx + 1)
                print("graph: ", [sorted(list(comp)) for comp in camera_groups])
                continue

            # based on the connected graph, optimize the relative pose of camera 2~5 to camera 1
            # order the pairs by similarity from high to low in camera_match_pairs
            for key in camera_match_pairs:
                camera_match_pairs[key].sort(key=lambda x: x[2], reverse=True)  # high to low

            original_xz_total = [item for sublist in original_xz_list for item in sublist]
            original_angle_total = [item for sublist in original_angle_list for item in sublist]

            # camera pose estimation method:
            # if camera j has common person with camera 0, then estimate T_0j directly, otherwise estimate T_0j indirectly
            # camera j to camera 0: T_0j = (dx, dy, theta)
            T0j_dict = {}  # {1: T_01, 2: T_02, ...}
            direct_estimates = {}  # key: camera_id, value: T_0j = (dx, dy, theta)

            # make camera graph according to camera_match_pairs
            if cfg.graph_search == 'bfs':
                estimation_order = get_pose_estimation_order_bfs(camera_match_pairs, start_cam=0)
            elif cfg.graph_search == 'dfs':
                estimation_order = get_pose_estimation_order_dfs(camera_match_pairs, start_cam=0)

            for camera_match_pair in estimation_order:
                c1, c2 = camera_match_pair
                # if camera c2 has common person with camera 0, then estimate T_0j directly
                if c1 == 0:
                    T_direct = estimate_T_0j_direct(c2, camera_match_pairs, original_xz_total, original_angle_total, topK=cfg.match_target_topK)
                    direct_estimates[c2] = T_direct
                    T0j_dict[c2] = T_direct

                # if camera c2 has no common person with camera 0, then estimate T_0j indirectly
                else:
                    # find the camera that has direct estimate
                    if c1 in direct_estimates:
                        T_indrect = estimate_T_0j_indirect_specify(c2, c1, camera_match_pairs, num_cameras, T0j_dict,
                                                                      person_to_camera,
                                                                      original_xz_total, original_angle_total, topK=cfg.match_target_topK)
                        direct_estimates[c2] = T_indrect
                        T0j_dict[c2] = T_indrect
                    elif c2 in direct_estimates:
                        T_indrect = estimate_T_0j_indirect_specify(c1, c2, camera_match_pairs, num_cameras, T0j_dict,
                                                                      person_to_camera,
                                                                      original_xz_total, original_angle_total, topK=cfg.match_target_topK)
                        direct_estimates[c1] = T_indrect
                        T0j_dict[c1] = T_indrect
                    else:
                        print(f'camera {c1} and camera {c2} have no common person with camera 0, skip')
                        print('estimation_order:', estimation_order)
                        print('camera_match_pairs:', camera_match_pairs)

            # check
            assert len(T0j_dict) == num_cameras - 1, "not all cameras have pose estimation"

            # cam_loss
            cam_xy_loss = 0
            cam_r_loss = 0
            for j in range(1, num_cameras):
                x_gt, y_gt, r_gt = x_gt_list[j], y_gt_list[j], r_gt_list[j]
                x_pred, y_pred, r_pred = T0j_dict[j]

                r_pred = r_pred * (180/pi)
                if r_pred < 0:
                    r_pred += 360
                r_pred = 360 - r_pred
                r_pred -= 90
                if r_pred > 180:
                    r_pred = r_pred - 360
                r_pred = r_pred / (180/pi)

                cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
                cam_r_loss += angular_loss(r_gt, r_pred)
                camera_meter.update([[x_pred, y_pred, r_pred]], [[x_gt, y_gt, r_gt]], 0, 0)

            cam_xy_loss = cam_xy_loss / (num_cameras - 1)
            cam_r_loss = cam_r_loss / (num_cameras - 1)

            # transform pedestrian pose from camera j to camera 1
            for j in range(0, num_cameras):
                if j == 0:
                    original_xz = original_xz_list[0]
                    original_angle = original_angle_list[0]
                    for k in range(len(original_xz)):
                        xz = original_xz[k]
                        angle = original_angle[k]
                        top_pred_xy_total_list.append(torch.stack([xz[0], xz[1]]))
                        top_pred_r_total_list.append(angle[0])

                    bbox_total_list += bbox_list[0]
                else:
                    original_xz = original_xz_list[j]
                    original_angle = original_angle_list[j]
                    delta_x, delta_y, delta_theta = T0j_dict[j]
                    local_to_global_tf = torch.stack([delta_x, delta_y, delta_theta])

                    for k in range(len(original_xz)):
                        xz = original_xz[k]
                        angle = original_angle[k]
                        local_pose = torch.stack([xz[0], xz[1], angle[0]])
                        new_xzr_pred = transform_pose_local_to_global(local_pose, local_to_global_tf)
                        top_pred_xy_total_list.append(new_xzr_pred[:2])
                        top_pred_r_total_list.append(new_xzr_pred[2])

                    bbox_total_list += bbox_list[j]

            # compute distance and angle difference matrix
            distance_total_matrix = torch.zeros_like(match_total_matrix)
            r_total_matrix = torch.zeros_like(match_total_matrix)
            # match_total_matrix_gt = torch.zeros_like(match_total_matrix)
            n, m = distance_total_matrix.shape
            for i in range(n):
                for j in range(m):
                    # if id_total_list[i] == id_total_list[j]:
                    #     match_total_matrix_gt[i][j] = 1
                    distance_total_matrix[i][j] = math.sqrt(
                        (top_pred_xy_total_list[i][0].item() - top_pred_xy_total_list[j][0].item()) ** 2 + (
                                top_pred_xy_total_list[i][1].item() - top_pred_xy_total_list[j][1].item()) ** 2)
                    r_total_matrix[i][j] = clac_rad_distance(top_pred_r_total_list[i], top_pred_r_total_list[j]).item()

            dis_mask = distance_total_matrix <= distance_threshold
            if cfg.pre_know_real_match:
                reid_val_mask = match_total_matrix_gt >= matrix_threshold
            else:
                reid_val_mask = match_total_matrix >= matrix_threshold
            mask = torch.logical_and(dis_mask, reid_val_mask)  # position with True satisfy both distance and similarity threshold
            mask_up = torch.triu(mask, 1)

            total_f1 = calc_f1_loss_by_matrix(torch.triu(match_total_matrix_gt, 1),
                                              mask_up.int()).item()
            total_f1_sum += total_f1

            # remove self matrix mask
            person_num_sum = 0
            for person_num in person_num_list:
                mask_up[person_num_sum:person_num_sum + person_num,
                person_num_sum:person_num_sum + person_num] = False
                person_num_sum += person_num

            ###################### aggregate according to similarity and distance ######################
            union_find = UnionFind(len(id_total_list))

            for i in range(len(id_total_list)):
                for j in range(len(id_total_list)):
                    if mask_up[i][j]:
                        # pair_dict[i].append([j, match_total_matrix[i][j]])
                        union_find.union(i, j)
            union_collection_dict = {i: [] for i in range(len(id_total_list))}
            for i in range(len(id_total_list)):
                union_collection_dict[union_find.find(i)].append(i)  # {root：[sub1, sub2, ...]}
            aggregated_list = []
            for key, val in union_collection_dict.items():
                if len(val) > 1:
                    aggregated_list.append(val)

            # using sub-graph algorithm
            pop_index_list = []
            for index, aggregated_sub_list in enumerate(aggregated_list):
                split_list = []
                person_interval_counter = [[] for i in range(len(person_interval_list) - 1)]
                for i in range(len(person_interval_list) - 1):  # 0~4
                    for j in range(len(aggregated_sub_list)):
                        if person_interval_list[i] <= aggregated_sub_list[j] < person_interval_list[i + 1]:
                            person_interval_counter[i].append(aggregated_sub_list[j])
                person_counter_max = max(
                    [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                if person_counter_max > 1:
                    # split
                    while person_counter_max > 1:
                        for i in range(len(person_interval_counter)):
                            tmp_split = []
                            if len(person_interval_counter[i]) > 0:
                                pivot = person_interval_counter[i][0]
                                tmp_split.append(person_interval_counter[i].pop(0))
                                for j in range(i + 1, len(person_interval_counter)):
                                    score_list = [match_total_matrix[pivot][elem_index] for elem_index in
                                                  person_interval_counter[j]]
                                    if len(score_list) == 0:
                                        continue
                                    if max(score_list) < cfg.matrix_threshold_test:
                                        break
                                    else:
                                        max_index = score_list.index(max(score_list))
                                        tmp_split.append(person_interval_counter[j].pop(max_index))
                            if tmp_split != []:
                                aggregated_list.append(tmp_split)

                        person_counter_max = max(
                            [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                        # collecting splited list
                    # collecting remaining list
                    pop_index_list.append(index)
                    remaining_list = []
                    for i in range(len(person_interval_counter)):
                        for j in range(len(person_interval_counter[i])):
                            remaining_list.append(person_interval_counter[i][j])
                    if remaining_list != []:
                        aggregated_list.append(remaining_list)
                else:
                    pass
            aggregated_list_new = []
            for i in range(len(aggregated_list)):
                if i in pop_index_list:
                    continue
                else:
                    aggregated_list_new.append(aggregated_list[i])
            aggregated_list = aggregated_list_new

            ###################### aggregate end ######################

            ###################### metric calculate ######################
            index_used_set = set()
            total_index_set = set([i for i in range(len(id_total_list))])

            # mean
            final_mean_xy_total_list = []
            final_mean_r_total_list = []

            # centroid 1 choosing
            final_only_one_xy_total_list = []
            final_only_one_r_total_list = []
            final_only_one_bbox_total_list = []

            final_xy_total_gt_list = []
            final_r_total_gt_list = []
            final_id_total_list = []

            # aggregated more view points here
            consistency_xy_loss = torch.tensor(0.0, device=device)
            consistency_r_loss = torch.tensor(0.0, device=device)
            num_groups = 0
            for aggregated_sub_list in aggregated_list:
                x_sum = 0
                y_sum = 0
                r_sum = 0
                # used to caculate centroid
                x_cache = []
                y_cache = []
                r_cache = []

                for index_sub in range(len(aggregated_sub_list)):
                    index_used_set.add(aggregated_sub_list[index_sub])

                    x_cache.append(top_pred_xy_total_list[aggregated_sub_list[index_sub]][0])
                    y_cache.append(top_pred_xy_total_list[aggregated_sub_list[index_sub]][1])
                    r_cache.append(top_pred_r_total_list[aggregated_sub_list[index_sub]])

                    x_sum += top_pred_xy_total_list[aggregated_sub_list[index_sub]][0]
                    y_sum += top_pred_xy_total_list[aggregated_sub_list[index_sub]][1]
                    r_sum += top_pred_r_total_list[aggregated_sub_list[index_sub]]
                x_mean = x_sum / len(aggregated_sub_list)
                y_mean = y_sum / len(aggregated_sub_list)
                r_mean = r_sum / len(aggregated_sub_list)

                # cross-view consistency loss
                for index_sub in range(len(aggregated_sub_list)):
                    consistency_xy_loss += torch.abs(top_pred_xy_total_list[aggregated_sub_list[index_sub]][0] - x_mean)
                    consistency_xy_loss += torch.abs(top_pred_xy_total_list[aggregated_sub_list[index_sub]][1] - y_mean)
                    consistency_r_loss += angular_loss(r_mean, top_pred_r_total_list[aggregated_sub_list[index_sub]])
                num_groups += 1

                # centroid about
                to_centroid_distance_list = [math.sqrt((x_cache[i] - x_mean) ** 2 + (y_cache[i] - y_mean) ** 2) for i in
                                             range(len(x_cache))]
                rank_list = torch.argsort(torch.tensor(to_centroid_distance_list)).tolist()
                rank1_index = rank_list[0]

                total1_index = aggregated_sub_list[rank1_index]

                final_only_one_xy_total_list.append(top_pred_xy_total_list[total1_index])
                final_only_one_r_total_list.append(top_pred_r_total_list[total1_index])
                final_only_one_bbox_total_list.append(
                    [bbox_total_list[aggregated_sub_list[i]] for i in range(len(aggregated_sub_list))])

                if r_mean > math.pi:
                    r_mean = r_mean - 2 * math.pi
                if r_mean < -math.pi:
                    r_mean = 2 * math.pi + r_mean
                final_mean_xy_total_list.append(torch.stack([x_mean, y_mean]))
                final_mean_r_total_list.append(r_mean)

                id_used_here = id_total_list[aggregated_sub_list[0]]

                final_id_total_list.append(id_used_here)
                final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])
                final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])
            # aggregated more view points here
            # aggregated single points here
            single_index_list = sorted(list(total_index_set - index_used_set))

            for single_index in single_index_list:
                id_used_here = id_total_list[single_index]

                final_only_one_xy_total_list.append(top_pred_xy_total_list[single_index])
                final_only_one_r_total_list.append(top_pred_r_total_list[single_index])
                final_only_one_bbox_total_list.append([bbox_total_list[single_index]])

                final_mean_xy_total_list.append(top_pred_xy_total_list[single_index])
                final_mean_r_total_list.append(top_pred_r_total_list[single_index])

                final_id_total_list.append(id_used_here)
                final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])
                final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])

            # aggregated single points here

            statistic_counter_dict = {id: [] for id in list(set(id_total_list))}
            for index_val, id_val in enumerate(id_total_list):
                statistic_counter_dict[id_val].append(index_val)
            # sort list
            for key in statistic_counter_dict.keys():
                statistic_counter_dict[key].sort()
            gt_aggregated_list = [val for val in statistic_counter_dict.values()]
            # statistic every person's probability
            prob_counter = 0
            for i in range(len(aggregated_list)):
                if aggregated_list[i] in gt_aggregated_list:
                    prob_counter += 1

            original_final_id_total_list = final_id_total_list[:]
            try:
                final_person_ratio = prob_counter / len(aggregated_list)
            except ZeroDivisionError:
                final_person_ratio = 0.0
            no_repeat_tag = len(set(final_id_total_list)) == len(final_id_total_list)

            hit_total_one_meter.update([[final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1],
                                         final_only_one_r_total_list[i]] for i in
                                        range(len(final_only_one_xy_total_list))],
                                       [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1],
                                         final_r_total_gt_list[i]] for i in range(len(final_xy_total_gt_list))], 0, 0)
            hit_total_mean_meter.update(
                [[final_mean_xy_total_list[i][0], final_mean_xy_total_list[i][1], final_mean_r_total_list[i]] for i in
                 range(len(final_mean_xy_total_list))],
                [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1], final_r_total_gt_list[i]] for i in
                 range(len(final_xy_total_gt_list))], 0, 0)
            ###################### metric calculate end ######################
            # BEV target loss
            target_xy_loss = cal_target_xy_loss(final_only_one_xy_total_list, final_xy_total_gt_list, loss_type='l2')
            target_r_loss = cal_target_r_loss(final_only_one_r_total_list, final_r_total_gt_list)

            # cross-view consistency loss
            if num_groups > 0:
                consistency_xy_loss /= num_groups
                consistency_r_loss /= num_groups

            re_id_matrix_loss = torch.nn.MSELoss()(match_total_matrix,
                                                   match_total_matrix_gt)
            total_loss = (cfg.reid_ratio * re_id_matrix_loss + cfg.xy_ratio * cam_xy_loss + cfg.r_ratio * cam_r_loss
                          + cfg.consistency_xy_ratio * consistency_xy_loss + cfg.consistency_r_ratio * consistency_r_loss
                          + cfg.target_xy_ratio * target_xy_loss + cfg.target_r_ratio * target_r_loss)

            loss_meter.update('target_xy_loss', target_xy_loss.item())
            loss_meter.update('target_r_loss', target_r_loss.item())
            loss_meter.update('cam_xy_loss', cam_xy_loss.item())
            loss_meter.update('cam_r_loss', cam_r_loss.item())
            loss_meter.update('consistency_xy_loss', consistency_xy_loss.item())
            loss_meter.update('consistency_r_loss', consistency_r_loss.item())
            loss_meter.update('re_id_loss', re_id_matrix_loss.item())
            loss_meter.update('total_loss', total_loss.item())
            ###################### heatmap/bbox ######################
            if draw_heat:
                # draw gt coverage
                gt_xyangle_list = []
                id_list = []
                for key, val in draw_gt_id_xyangle_dict_total.items():
                    gt_xyangle_list.append(val)
                    id_list.append(key)

                # filter  subjects with cameras
                while max(id_list) > 20:
                    index_max = id_list.index(max(id_list))
                    del id_list[index_max]
                    del gt_xyangle_list[index_max]

                # adding gt camera
                # adding main camera
                gt_xyangle_list.append([0, 0, -90 / 57.3])
                id_list.append(-1)
                # adding more view cameras
                for j in range(1, num_cameras):
                    x_gt, y_gt, r_gt = x_gt_list[j], y_gt_list[j], r_gt_list[j]
                    gt_xyangle_list.append([x_gt, y_gt, r_gt])
                    id_list.append(-1 - j)

                expand_ratio = 1.6
                for i in range(len(gt_xyangle_list)):
                    gt_xyangle_list[i][0] *= expand_ratio
                    gt_xyangle_list[i][1] *= expand_ratio

                frame_str = (4 - len(str(frame_id))) * '0' + str(frame_id)

                cfg.generator.get_heatmap_from_xzangle_id(gt_xyangle_list, id_list, False, adding_border=False)
                cfg.generator.save_img(os.path.join(cfg.result_path, f"figs/{frame_str}_gt_top_view.png"))
                gt_coverage = (cfg.generator.board * 0.4).int()
                # draw gt coverage

                final_xyr_draw = [
                    [final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1],
                     final_only_one_r_total_list[i]]
                    for i in range(len(final_id_total_list))]
                final_xyr_draw.append([0, 0, -90 / 57.3])
                final_id_total_list.append(-1)

                for j in range(1, num_cameras):
                    x_pred, y_pred, r_pred = T0j_dict[j]

                    r_pred = r_pred * (180/pi)
                    if r_pred < 0:
                        r_pred += 360
                    r_pred = 360 - r_pred
                    r_pred -= 90
                    if r_pred > 180:
                        r_pred = r_pred - 360
                    r_pred = r_pred / (180/pi)

                    final_xyr_draw.append([x_pred, y_pred, r_pred])
                    final_id_total_list.append(-1 - j)

                for i in range(len(final_xyr_draw)):
                    final_xyr_draw[i][0] *= expand_ratio
                    final_xyr_draw[i][1] *= expand_ratio

                while max(final_id_total_list) > 20:
                    index_max = final_id_total_list.index(max(final_id_total_list))
                    del final_xyr_draw[index_max]
                    del final_id_total_list[index_max]

                cfg.generator.get_heatmap_from_xzangle_id(final_xyr_draw, final_id_total_list, if_cropped=False)
                cfg.generator.board += gt_coverage
                cfg.generator.save_img(os.path.join(cfg.result_path, f"figs/{frame_str}_predict_top_view_with_gt.png"))

            if draw_bbox:
                # draw different view with bbox and real top view
                bbox_view_1 = []
                bbox_view_1_id = []
                bbox_view_2 = []
                bbox_view_2_id = []
                bbox_view_3 = []
                bbox_view_3_id = []
                bbox_view_4 = []
                bbox_view_4_id = []
                bbox_view_5 = []
                bbox_view_5_id = []

                for id in range(1, len(final_only_one_bbox_total_list) + 1):
                    index = id - 1
                    bbox_list = final_only_one_bbox_total_list[index]
                    for bbox in bbox_list:
                        if bbox[0] == 1:
                            bbox_view_1.append(bbox[1:])
                            bbox_view_1_id.append(original_final_id_total_list[index])
                        elif bbox[0] == 2:
                            bbox_view_2.append(bbox[1:])
                            bbox_view_2_id.append(original_final_id_total_list[index])
                        elif bbox[0] == 3:
                            bbox_view_3.append(bbox[1:])
                            bbox_view_3_id.append(original_final_id_total_list[index])
                        elif bbox[0] == 4:
                            bbox_view_4.append(bbox[1:])
                            bbox_view_4_id.append(original_final_id_total_list[index])
                        elif bbox[0] == 5:
                            bbox_view_5.append(bbox[1:])
                            bbox_view_5_id.append(original_final_id_total_list[index])
                        else:
                            print('no such view!')
                            pass

                try:
                    view1_color = [tuple(rgb_table[id]) for id in bbox_view_1_id]
                    view2_color = [tuple(rgb_table[id]) for id in bbox_view_2_id]
                    view3_color = [tuple(rgb_table[id]) for id in bbox_view_3_id]
                    view4_color = [tuple(rgb_table[id]) for id in bbox_view_4_id]
                    view5_color = [tuple(rgb_table[id]) for id in bbox_view_5_id]
                except:
                    continue

                img1_path = dataset[0].img_path[frame_idx]
                img2_path = dataset[1].img_path[frame_idx]
                img3_path = dataset[2].img_path[frame_idx]
                img4_path = dataset[3].img_path[frame_idx]
                img5_path = dataset[4].img_path[frame_idx]

                tmp = img1_path.split('/')
                tmp[-2] = 'top_video'
                img_top_path = '/'.join(tmp)

                img1 = torchvision.io.read_image(img1_path)
                img1 = draw_bounding_box_with_color_or_labels(img1, bbox_view_1, colors=view1_color)
                img2 = torchvision.io.read_image(img2_path)
                img2 = draw_bounding_box_with_color_or_labels(img2, bbox_view_2, colors=view2_color)
                img3 = torchvision.io.read_image(img3_path)
                img3 = draw_bounding_box_with_color_or_labels(img3, bbox_view_3, colors=view3_color)
                img4 = torchvision.io.read_image(img4_path)
                img4 = draw_bounding_box_with_color_or_labels(img4, bbox_view_4, colors=view4_color)
                img5 = torchvision.io.read_image(img5_path)
                img5 = draw_bounding_box_with_color_or_labels(img5, bbox_view_5, colors=view5_color)
                img_top = torchvision.io.read_image(img_top_path)
                # top bbox
                top_bbox_id = torch.load(os.path.join(cfg.label, 'f_top_bbox_pid.pth'))[str(int(frame_str))]
                top_id_list = list(set(original_final_id_total_list))
                top_color = [tuple(rgb_table[id]) for id in top_id_list]
                top_bbox_list = []
                for bbox, id in top_bbox_id:
                    if id in top_id_list:
                        top_bbox_list.append(bbox)
                img_top = draw_bounding_box_with_color_or_labels(img_top, top_bbox_list, colors=top_color, width=2)

                img_list = [img1, img2, img3, img4, img5, img_top]
                imgs = torch.stack(img_list, dim=0)
                imgs = make_grid(imgs, padding=4, pad_value=255, nrow=3)
                torchvision.io.write_png(imgs,
                                         os.path.join(cfg.result_path, f"figs/{frame_str}_first_view_and_top_bbox.png"))

            ###################### heatmap\bbox end ######################

        test_info = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),

            'target_xy_loss': loss_meter.read('target_xy_loss'),
            'target_r_loss': loss_meter.read('target_r_loss'),
            'cam_xy_loss': loss_meter.read('cam_xy_loss'),
            'cam_r_loss': loss_meter.read('cam_r_loss'),
            'consistency_xy_loss': loss_meter.read('consistency_xy_loss'),
            'consistency_r_loss': loss_meter.read('consistency_r_loss'),
            're_id_loss': loss_meter.read('re_id_loss'),
            'total_loss': loss_meter.read('total_loss'),

            'cam_prob': camera_meter.get_xy_r_prob_dict(),
            'total_sub_prob_one': hit_total_one_meter.get_xy_r_prob_dict(),
            'total_sub_xy_one': hit_total_one_meter.get_xy_mean_error(),
            'total_sub_r_one': hit_total_one_meter.get_r_mean_error(),

            # 'total_sub_prob_mean': hit_total_mean_meter.get_xy_r_prob_dict(),
            # 'total_sub_xy_mean': hit_total_mean_meter.get_xy_mean_error(),
            # 'total_sub_r_mean': hit_total_mean_meter.get_r_mean_error(),

            'total_f1': total_f1_sum / cfg.test_num,
        }

        return test_info


def try_to(ts, device):
    if ts is not None:
        return ts.to(device)
    else:
        return None


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
