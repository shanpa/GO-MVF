import torch
from torch.utils import data
import torchvision as tv
import torchvision.transforms as transforms
import os

from utils.utils import get_matchid_from_fv_id_start_from_1
from utils.utils import crop_img_by_boxes, crop_img_by_boxes_without_norm


class VirtualDataset_O(data.Dataset):

    def __init__(self, cfg, input_dir, label):
        super(VirtualDataset_O, self).__init__()
        self.cfg = cfg
        self.img_path = []
        self.label = label

        v0_list = sorted(os.listdir(input_dir))
        v0_list = [input_dir + '/' + v0 for v0 in v0_list]
        self.img_path = v0_list

        self.fv_dict = torch.load(os.path.join(cfg.label, 'fv.pth'))
        self.fv_sk_box = torch.load(os.path.join(cfg.label, 'fv_sk_box.pth'))  # fv_sk_box[f"{frame_id}_{view_id}"] = [keypoints, boxes],
        # frame person id: lable
        # key is int
        self.fps = torch.load(os.path.join(cfg.label, 'fps.pth'))

        self.f_top_bbox_pid = torch.load(os.path.join(cfg.label, 'f_top_bbox_pid.pth'))  # f_top_bbox_id_dict[frame].append([top_bbox, pid])

        self.img_path = self.img_path[:cfg.train_num+cfg.test_num]
        self.label = self.label[:cfg.train_num+cfg.test_num]

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.label)

    def __getitem__(self, index):
        """
        :param index:
        :return:
            0. img1: [b, 3, 768, 1024]
            1. img2: None
            2. img1_cropped: [b, n, 3, 256, 128]
            3. img2_cropped: None

            pifpaf：
            4. sk1: [n, 3, 17] (skeleton coordinates from pifpaf)
            5. box1:[n, 5] camera 1 (x_min, y_min, x_max, y_max, conf)
            6. sk2: None
            7. box2: None

            8. label: [x, y, r] (gt label of camera x, y, r)
            9. frame_vid: (frame, view1_id,) eg: [2,1]

            10. col_ids_list1: [id1_list]
            11. col_ids_list2: None
            12. top_box: [n, 4]
            13. top_id: subject id of each top box

            14. view1_box: gt bbox of view1
            15. view1_ids: subject id of each view1
            16. view1_label: gt label of view1 (x, y, r)

            17. view2_box: None
            18. view2_ids: None
            19. view2_label: None

        """

        if self.cfg.channel == 1:
            img1 = tv.io.read_image(self.img_path[index]).float()
            img2 = None
        else:
            img1 = tv.io.read_image(self.img_path[index], tv.io.image.ImageReadMode.RGB).float()
            img2 = None

        frame_vid = []

        path1 = self.img_path[index]
        hor, img_path = path1.split('/')[-2], path1.split('/')[-1]
        view_id = int(hor[3])  # 1~5

        if 'mono' in self.cfg.model_name:
            frame_id = int(img_path.split('.')[0])
        else:
            frame_id = int(img_path.split('.')[0].split('_')[1])
        frame_vid.append(frame_id)
        frame_vid.append(view_id)

        frame_vid = torch.tensor(frame_vid)

        label = torch.tensor(self.label[index])

        fv_str1 = "%s_%s" % (frame_vid[0].item(), frame_vid[1].item())

        sk1, box1 = self.fv_sk_box[fv_str1]
        sk2, box2 = [], []
        # print('Virtual Dataset: ', fv_str1, fv_str2)
        img1_cropped = crop_img_by_boxes(img1, box1)
        img2_cropped = None

        sk1 = torch.tensor(sk1)
        sk2 = torch.tensor(sk2)

        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)

        top_bbox_id = self.f_top_bbox_pid[str(frame_id)]
        top_box = torch.tensor([elem[0] for elem in top_bbox_id])
        top_id = torch.tensor([elem[1] for elem in top_bbox_id])

        _, col_ids_list1 = get_matchid_from_fv_id_start_from_1(box1, self.fv_dict, fv_str1)
        _, col_ids_list2 = None, []

        col_ids_list1 = torch.tensor(col_ids_list1)
        col_ids_list2 = torch.tensor(col_ids_list2)

        view1_all_gt = self.fv_dict[fv_str1]
        view2_all_gt = None

        view1_ids = torch.tensor([p_bbox[0] for p_bbox in view1_all_gt])
        view2_ids = None

        view1_box = torch.tensor([p_bbox[1] for p_bbox in view1_all_gt])
        view2_box = None

        frame_lable_dict = self.fps[frame_vid[0].item()]

        view1_lable = torch.tensor([[float(frame_lable_dict[id.item()][0]), float(frame_lable_dict[id.item()][1]),
                                     float(frame_lable_dict[id.item()][2])] for id in view1_ids])
        view2_lable = None

        return img1, img2, img1_cropped, img2_cropped, sk1, box1, sk2, box2, label, frame_vid, col_ids_list1, col_ids_list2, \
            top_box, top_id, view1_box, view1_ids, view1_lable, view2_box, view2_ids, view2_lable
