try:
    from dataset.virtual_dataset import *
except:
    from virtual_dataset import *

# return train and test dataset
def return_more_view_dataset_mvf(cfg):
    if 'Virtual' in cfg.dataset_name:
        dataset = []
        for i in range(0, cfg.view_num):  # view
            view_label = []
            view_dir = cfg.all_input_list[i]  # dir

            with open(cfg.label + '/' + f"camera{i + 1}.txt") as f:
                for line in f:
                    if len(line) > 3:
                        tmp = line.split()
                        tmp = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                        view_label.append(tmp)  # camera x、y、r

            view_dataset = VirtualDataset_O(cfg, view_dir, view_label)

            dataset.append(view_dataset)

        return dataset

    elif 'CSRD_O' in cfg.dataset_name:
        dataset = []
        for i in range(0, cfg.view_num):  # view
            view_label = []
            view_dir = cfg.all_input_list[i]  # dir

            with open(cfg.label + '/' + f"camera{i + 1}.txt") as f:
                for line in f:
                    if len(line) > 3:
                        tmp = line.split()
                        tmp = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                        view_label.append(tmp)  # camera x、y、r

            view_dataset = VirtualDataset_O(cfg, view_dir, view_label)

            dataset.append(view_dataset)

        return dataset
