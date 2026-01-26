import numpy as np
import os
import matplotlib.pyplot as plt
import math



class PlotCurvesGeneral:
    def __init__(self):
        self.metrics = {}  # 存储指标名 -> 数据列表

    def register(self, name: str, data: list):
        """
         register a metric and its data.

        :param name: （str）
        :param data: （list of numbers）
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(data, list):
            raise ValueError("Data must be a list.")
        self.metrics[name] = data

    def plot_all(self, save_dir: str, filename: str):
        """
        draw all registered metrics in a single plot.

        :param save_dir: （str）
        :param filename: （eg: 'curves.png'）
        """
        if not self.metrics:
            print("No metrics registered. Nothing to plot.")
            return

        num_metrics = len(self.metrics)

        cols = math.ceil(math.sqrt(num_metrics))
        rows = math.ceil(num_metrics / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))

        if num_metrics == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for idx, (name, data) in enumerate(self.metrics.items()):
            axs[idx].plot(data, color='green')
            axs[idx].set_title(name)
            # axs[idx].set_xlabel('Step')
            # axs[idx].set_ylabel('Value')
            axs[idx].grid(True)

        for idx in range(num_metrics, len(axs)):
            axs[idx].axis('off')

        plt.tight_layout()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        plt.close(fig)

        print(f"Plot saved to {full_path}")


def plot_train_loss_curves_frame(re_id_loss_frame, cam_xy_loss_frame, cam_r_loss_frame,
                                consistency_xy_loss_frame, consistency_r_loss_frame,
                                target_xy_loss_frame, target_r_loss_frame,
                                 total_loss_frame, epoch, save_dir):

    # 创建2*4的子图布局
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))

    # 分别绘制不同的loss曲线到对应的子图中
    axs[0, 0].plot(cam_xy_loss_frame, color='green')
    axs[0, 0].set_title('cam_xy_loss')
    axs[0, 0].set_xlabel('frame')
    axs[0, 0].set_ylabel('loss')

    axs[0, 1].plot(cam_r_loss_frame, color='purple')
    axs[0, 1].set_title('cam_r_loss')
    axs[0, 1].set_xlabel('frame')
    axs[0, 1].set_ylabel('loss')

    axs[0, 2].plot(consistency_xy_loss_frame, color='orange')
    axs[0, 2].set_title('consistency_xy_loss')
    axs[0, 2].set_xlabel('frame')
    axs[0, 2].set_ylabel('loss')

    axs[0, 3].plot(consistency_r_loss_frame, color='orange')
    axs[0, 3].set_title('consistency_r_loss')
    axs[0, 3].set_xlabel('frame')
    axs[0, 3].set_ylabel('loss')

    axs[1, 0].plot(target_xy_loss_frame, color='cyan')
    axs[1, 0].set_title('target_xy_loss')
    axs[1, 0].set_xlabel('frame')
    axs[1, 0].set_ylabel('loss')

    axs[1, 1].plot(target_r_loss_frame, color='magenta')
    axs[1, 1].set_title('target_r_loss')
    axs[1, 1].set_xlabel('frame')
    axs[1, 1].set_ylabel('loss')

    axs[1, 2].plot(re_id_loss_frame, color='blue')
    axs[1, 2].set_title('re_id_loss')
    axs[1, 2].set_xlabel('frame')
    axs[1, 2].set_ylabel('loss')

    axs[1, 3].plot(total_loss_frame, color='red')
    axs[1, 3].set_title('total_loss')
    axs[1, 3].set_xlabel('frame')
    axs[1, 3].set_ylabel('loss')

    # 自动调整子图参数以填充整个区域
    plt.tight_layout()

    # 检查保存图像的目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'train_loss_epoch_{epoch}.png'))
    plt.close()  # 关闭图像以释放内存


def plot_loss_curves_epoch(re_id_loss_epoch, cam_xy_loss_epoch, cam_r_loss_epoch,
                           consistency_xy_loss_epoch, consistency_r_loss_epoch,
                           target_xy_loss_epoch, target_r_loss_epoch, total_loss_epoch,
                           save_dir, train_flag=True):
    # 创建2*4的子图布局
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))

    # 分别绘制不同的loss曲线到对应的子图中

    axs[0, 0].plot(cam_xy_loss_epoch, color='green')
    axs[0, 0].set_title('cam_xy_loss')
    axs[0, 0].set_xlabel('epoch')
    axs[0, 0].set_ylabel('loss')

    axs[0, 1].plot(cam_r_loss_epoch, color='purple')
    axs[0, 1].set_title('cam_r_loss')
    axs[0, 1].set_xlabel('epoch')
    axs[0, 1].set_ylabel('loss')

    axs[0, 2].plot(consistency_xy_loss_epoch, color='orange')
    axs[0, 2].set_title('consistency_xy_loss')
    axs[0, 2].set_xlabel('epoch')
    axs[0, 2].set_ylabel('loss')

    axs[0, 3].plot(consistency_r_loss_epoch, color='orange')
    axs[0, 3].set_title('consistency_r_loss')
    axs[0, 3].set_xlabel('epoch')
    axs[0, 3].set_ylabel('loss')

    axs[1, 0].plot(target_xy_loss_epoch, color='cyan')
    axs[1, 0].set_title('target_xy_loss')
    axs[1, 0].set_xlabel('epoch')
    axs[1, 0].set_ylabel('loss')

    axs[1, 1].plot(target_r_loss_epoch, color='magenta')
    axs[1, 1].set_title('target_r_loss')
    axs[1, 1].set_xlabel('epoch')
    axs[1, 1].set_ylabel('loss')

    axs[1, 2].plot(re_id_loss_epoch, color='blue')
    axs[1, 2].set_title('re_id_loss')
    axs[1, 2].set_xlabel('epoch')
    axs[1, 2].set_ylabel('loss')

    axs[1, 3].plot(total_loss_epoch, color='red')
    axs[1, 3].set_title('total_loss')
    axs[1, 3].set_xlabel('epoch')
    axs[1, 3].set_ylabel('loss')

    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图像
    if train_flag:
        plt.savefig(os.path.join(save_dir, f'train_loss.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'test_loss.png'))
    plt.close()

color_map = {
    1: (0, 0, 1),
    2: (0, 1, 0),
    3: (1, 0, 0),
    4: (0, 1, 1),
    5: (1, 0, 1),

    -1: (0, 0, 1),
    -2: (0, 1, 0),
    -3: (1, 0, 0),
    -4: (0, 1, 1),
    -5: (1, 0, 1)
}

def draw_detections(det_poses, det_ids, gt_poses=None, gt_ids=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 设置画布
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True)
    ax.axhline(0, color='black', lw=2)
    ax.axvline(0, color='black', lw=2)

    def plot_pose(pose, id, arrow_length=2, is_gt=False):
        x, y, r = pose
        angle_deg = np.degrees(-r + np.pi / 2)
        angle_deg = (angle_deg + 360) % 360

        dx = np.cos(np.radians(angle_deg)) * arrow_length
        dy = np.sin(np.radians(angle_deg)) * arrow_length

        base_color = color_map.get(id, (0, 0, 0))
        if is_gt:
            color = base_color + (0.5,)
        else:
            color = base_color + (1.0,)


        marker = 'o' if id > 0 else '^'


        ax.plot(x, y, marker, color=color[:3], alpha=color[3], markersize=10)

        ax.arrow(x, y, dx, dy, head_width=1, head_length=arrow_length, fc=color[:3], ec=color[:3],
                 alpha=color[3])


    for pose, id in zip(det_poses, det_ids):
        plot_pose(pose, id)


    if gt_poses and gt_ids:
        for pose, id in zip(gt_poses, gt_ids):
            plot_pose(pose, id, is_gt=True)

    return fig


poses = [
    [0.0000, -0.0000, 0.0000],
    [10.7526, 9.5011, 265.0000],
    [7.7558, 0.1889, 305.0000],
    [-10.3656, 0.4345, 50.0001],
    [-10.9937, 11.9919, 110.0001],
    [0.4066, 12.4256, 180.0000],
    [-2.7132, 6.4276, 219.9995],
    [3.3066, 5.5276, 125.0000],
    [-5.2284, 4.7525, 50.0000],
    [-0.0894, 4.9963, 180.0000]
]
# 转弧度
det_poses = [[pose[0], pose[1], np.radians(pose[2])] for pose in poses]
det_ids = [-1, -2, -3, -4, -5, 1, 2, 3, 4, 5]

gt_ids = [1, 2, 3, 5]
gt_poses = [[0.0861992, 2.6342271999999998, 1.5706806282722514], [-0.5751984, 1.3626512, 2.268752181500873], [0.7009991999999999, 1.1718511999999999, 0.6108202443280978], [-0.0189528, 1.0592156, 1.5706806282722514]]

fig_handle = draw_detections(det_poses, det_ids, gt_poses, gt_ids)

# 保存图像
fig_handle.savefig('output.png')