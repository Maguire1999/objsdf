import os

import pickle

import h5py
import torch
import numpy as np

# import utils.general as utils
from tqdm import tqdm
import json
import random
from PIL import Image
from torchvision.transforms import functional as F
import imageio
from glob import glob

class ReplicaDataset(torch.utils.data.Dataset):
    """
    Dataset class for replica dataset
    """
    def load_color_dict(self):
        data_info = self.root_dir.split('/')
        data_info = [s.strip() for s in data_info if s.strip()]
        scene_name = data_info[-1]
        gt_color_dict_path = './datasets/color_dict.json'
        gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
        color_dict = gt_color_dict['replica'][scene_name]
        return color_dict

    def __init__(self,
                 data_dir,
                 img_res):
        self.root_dir = data_dir
        self.img_total_num = len(glob(os.path.join(self.root_dir, "rgb", "rgb_*.png")))
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.color_dict = self.load_color_dict()
        self.set_label_colour_map("")
        self.meta = {}
        # with open(os.path.join(data_dir, "transforms_full.json"), 'r') as f:
        #     self.meta = json.load(f)
        # self.train_list = np.loadtxt(os.path.join(data_dir, 'train.txt')).astype(int).tolist()
        # self.test_list = np.loadtxt(os.path.join(data_dir, 'test.txt')).astype(int).tolist()

        img_eval_interval = 5
        self.train_list = list(range(0, self.img_total_num, img_eval_interval))
        self.test_list = list(range(img_eval_interval // 2, self.img_total_num, img_eval_interval))

        self.img_res = img_res

        img_w, img_h = 640, 480
        self.img_wh = (int(img_w ), int(img_h))

        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.sampling_idx = None

        train_imgs = []
        train_segs = []
        train_poses = []
        test_imgs = []
        test_segs = []
        test_poses = []
        all_imgs = []
        all_sems = []
        all_poses = []
        # fix_rot = np.array([1, 0, 0, 0,
        #                     0, -1, 0, 0,
        #                     0, 0, -1, 0,
        #                     0, 0, 0, 1]).reshape(4, 4)
        fix_rot = np.array([1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1]).reshape(4, 4)
        
        # self.label_mapping = None
        # with open(os.path.join(data_dir, 'label_mapping.txt'), 'r') as f:
        #     content = f.readlines()
        #     self.label_mapping = [int(a) for a in content[0].split(',')]


        self.center = None
        if os.path.exists(os.path.join(data_dir, 'center.txt')):
            self.center = np.loadtxt(os.path.join(data_dir, 'center.txt')).reshape(4, 1)
        self.center_mat = np.zeros([4, 4])
        self.scale_mat = np.eye(4)
        if self.center is not None:
            self.center_mat[:3, 3:] = self.center[:3]
            self.scale_mat[0, 0] = 1.0/self.center[-1]
            self.scale_mat[1, 1] = 1.0/self.center[-1]
            self.scale_mat[2, 2] = 1.0/self.center[-1]
        for i in tqdm(self.train_list, desc=f'Loading data train ({len(self.train_list)})'):  # img_list:#
            c2w = torch.FloatTensor(self.Ts_full[i])
            image_path = os.path.join(self.root_dir, "rgb", f"rgb_{i}.png")
            sem_image_path = os.path.join(self.root_dir, 'semantic_instance', f"semantic_instance_{i}.png")
            train_imgs.append(
                imageio.imread(image_path).transpose(2, 0, 1).reshape(3, -1).transpose(1, 0))  # reshape to HW*3
            segs = imageio.imread(sem_image_path).reshape(1, -1).transpose(1, 0)
            # todo:check remap

            unique_labels = np.unique(segs)
            for index, label in enumerate(unique_labels):
                segs[segs == label] = self.color_dict[str(label)]

            # if self.label_mapping is not None:
            #     for i in self.label_mapping:
            #         segs[segs == i] = self.label_mapping.index(i)

            train_segs.append(segs)  # reshape to HW*1
            pose_matrix = np.array(c2w + self.center_mat)
            pose_matrix = self.scale_mat @ pose_matrix
            train_poses.append(pose_matrix @ fix_rot)

        for i in tqdm(self.test_list, desc=f'Loading data test ({len(self.test_list)})'):  # img_list:#
            c2w = torch.FloatTensor(self.Ts_full[i])
            image_path = os.path.join(self.root_dir, "rgb", f"rgb_{i}.png")
            sem_image_path = os.path.join(self.root_dir, 'semantic_instance', f"semantic_instance_{i}.png")
            test_imgs.append(
                imageio.imread(image_path).transpose(2, 0, 1).reshape(3, -1).transpose(1, 0))  # reshape to HW*3
            segs = imageio.imread(sem_image_path).reshape(1, -1).transpose(1, 0)
            # todo:check remap

            unique_labels = np.unique(segs)
            for index, label in enumerate(unique_labels):
                segs[segs == label] = self.color_dict[str(label)]
            # if self.label_mapping is not None:
            #     for i in self.label_mapping:
            #         segs[segs == i] = self.label_mapping.index(i)
            test_segs.append(segs)  # reshape to HW*1
            pose_matrix = np.array(c2w + self.center_mat)
            pose_matrix = self.scale_mat @ pose_matrix
            test_poses.append(pose_matrix @ fix_rot)


        train_imgs = (np.array(train_imgs) / 255.).astype(np.float32)
        train_segs = np.array(train_segs).astype(np.float32)
        train_poses = np.array(train_poses).astype(np.float32)
        test_imgs = (np.array(test_imgs) / 255.).astype(np.float32)
        test_segs = np.array(test_segs).astype(np.float32)
        test_poses = np.array(test_poses).astype(np.float32)
        all_imgs.append(train_imgs)
        all_imgs.append(test_imgs)
        all_sems.append(train_segs)
        all_sems.append(test_segs)
        all_poses.append(train_poses)
        all_poses.append(test_poses)

        self.i_split = [np.array(i) for i in [range(len(train_imgs)), range(len(train_imgs), len(train_imgs)+len(test_imgs))]]
        self.imgs = np.concatenate(all_imgs, 0)
        self.segs = np.concatenate(all_sems, 0)
        self.poses = np.concatenate(all_poses, 0)
        # print(self.poses.max(axis=0))
        # print(self.poses.min(axis=0))
        self.n_imgs = len(self.i_split)

        self.intrinsics_all = []
        # focal = 0.5 * img_res[1] / np.tan(0.5* float(self.meta['camera_angle_x']))
        w, h = self.img_wh
        hfov = 90
        self.focal_x = 0.5 * w / np.tan(0.5 * np.radians(hfov))  # w ?
        self.focal_y = self.focal_x
        cx = (w - 1.) / 2
        cy = (h - 1.) / 2
        intrinsics_objsdf = torch.from_numpy(np.array([
            [self.focal_x, 0, cx, 0],
            [0, self.focal_y, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])).float()
        for pose in self.poses:
            self.intrinsics_all.append(intrinsics_objsdf)

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": torch.from_numpy(self.poses[idx]).float()
        }

        ground_truth = {
            "rgb": torch.from_numpy(self.imgs[idx]).float(),
            "segs": torch.from_numpy(self.segs[idx]).float()
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = torch.from_numpy(self.imgs[idx][self.sampling_idx, :])
            ground_truth["segs"] = torch.from_numpy(self.segs[idx][self.sampling_idx, :])
            sample["uv"] = uv[self.sampling_idx, :]
        return idx, sample, ground_truth
    
    def __len__(self):
        return self.n_images


    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def fix_sampling_pattern(self, sampling_idx): 
        self.sampling_idx = sampling_idx

    # train_dataset.remap_sem_gt_label(train_dataset.sem_samples["sem_img"],
    #                                  test_dataset.sem_samples["sem_img"],
    #                                  args.sem_info_path)
    # self.sem_samples = {}
    # self.sem_samples["sem_img"] = []
    # self.sem_samples["label_ins_map"] = {}
    # self.sem_samples["ins_label_map"] = {}
    # self.sem_samples["sem_img"].append(np.array(sem_img))

    def remap_sem_gt_label(self, train_sem_imgs=None, test_sem_imgs=None, sem_info_path=None,
                           save_map=False,
                           load_map=False, ins2label_path=None):
        self.sem_samples["sem_img"] = np.asarray(self.sem_samples["sem_img"])
        self.sem_samples["sem_remap"] = self.sem_samples["sem_img"].copy()

        if load_map:
            assert ins2label_path, "map file path must be provided"

            scene_id = self.root_dir.split('/')[-1]
            ins2label = json.load((open(ins2label_path, 'r')))['replica'][scene_id]
            for ins, label in ins2label.items():
                self.sem_samples["sem_remap"][self.sem_samples["sem_img"] == int(ins)] = label

            self.num_semantic_class = len(ins2label.keys())
            self.num_valid_semantic_class = self.num_semantic_class

        else:
            self.semantic_classes = np.unique(np.concatenate((np.unique(train_sem_imgs), np.unique(test_sem_imgs))).astype(np.uint8))
            self.num_semantic_class = self.semantic_classes.shape[0]
            self.num_valid_semantic_class = self.num_semantic_class

            for i in range(self.num_semantic_class):
                self.sem_samples["sem_remap"][self.sem_samples["sem_img"] == self.semantic_classes[i]] = i
                self.sem_samples["label_ins_map"][i] = self.semantic_classes[i]
                self.sem_samples["ins_label_map"][self.semantic_classes[i]] = i

            if save_map:
                with open(self.root_dir+"label2ins_map.pkl", "wb") as f:
                    pickle.dump(self.sem_samples["label_ins_map"], f)

                with open(self.root_dir+"ins2label_map.pkl", "wb") as f:
                    pickle.dump(self.sem_samples["ins_label_map"], f)

    # def select_sems(self):
    #     assert self.is_stack
    #     self.selected_rays = self.all_rays[::self.sem_interval, ...]
    #     self.selected_rgbs = self.all_rgbs[::self.sem_interval, ...]
    #     self.selected_sems = torch.tensor(self.sem_samples["sem_remap"][::self.sem_interval, ...]).unsqueeze(-1)

    def set_label_colour_map(self, sem_info_path, label2color_path=None):
        if label2color_path:  # label to assigned color
            color_f = os.path.join(label2color_path)
        else:
            color_f = os.path.join(self.root_dir, 'ins_rgb.hdf5')

        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]  # ndarray
        f.close()

        def label_color_map(sem_map):
            color_map = np.zeros(shape=(int(self.img_wh[0] * self.img_wh[1]), 3))
            for label in np.unique(sem_map):
                valid_label_list = list(range(0, ins_rgbs.shape[0]))
                if label in valid_label_list:
                    color_map[sem_map == label] = ins_rgbs[label]
            return color_map

        self.label_color_map = label_color_map
        self.ins_rgbs = ins_rgbs



if __name__ == '__main__':
    a = ReplicaDataset('/data/dzy_data/nerf/datasets/toydesk_data/processed/our_desk_2', [640, 480])
    i_split = a.i_split
    print(i_split)
    # train_split = torch.utils.data.sampler.SubsetRandomSampler(i_split[0])
    # test_split = torch.utils.data.sampler.SubsetRandomSampler(i_split[1])
    test_data = torch.utils.data.Subset(a, i_split[1])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True, collate_fn=a.collate_fn)
    for epoch in range(3):
        if epoch == 0:
            a.change_sampling_idx(1024)
        else:
            a.change_sampling_idx(-1)
        for batch_idx, (idx, sample, gt) in enumerate(test_loader):
            print(epoch)
            print(idx, sample['uv'])