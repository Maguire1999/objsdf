import os
import torch
import numpy as np

# import utils.general as utils
from tqdm import tqdm
import json
import random
from PIL import Image
from torchvision.transforms import functional as F
import imageio
from kornia import create_meshgrid
from glob import glob
from torchvision import transforms as T

# import sys
# sys.path.append('../code')

from datasets.util.ray_utils import get_rays,get_ray_directions
from datasets.util.render_util import gen_path,circle

#
# def get_rays(directions, c2w):
#     """
#     Get ray origin and normalized directions in world coordinate for all pixels in one image.
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems
#     Inputs:
#         directions: (H, W, 3) precomputed ray directions in camera coordinate
#         c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
#     Outputs:
#         rays_o: (H*W, 3), the origin of the rays in world coordinate
#         rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
#     """
#     # Rotate ray directions from camera coordinate to the world coordinate
#     rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
#     # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
#     # The origin of all rays is the camera origin in world coordinate
#     rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
#
#     rays_d = rays_d.view(-1, 3)
#     rays_o = rays_o.view(-1, 3)
#
#     return rays_o, rays_d
#
# def get_ray_directions(H, W, focal, center=None):
#     """
#     Get ray directions for all pixels in camera coordinate.
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems
#     Inputs:
#         H, W, focal: image height, width and focal length
#     Outputs:
#         directions: (H, W, 3), the direction of the rays in camera coordinate
#     """
#     grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
#
#     i, j = grid.unbind(-1)
#     # the direction here is without +0.5 pixel centering as calibration is not so accurate
#     # see https://github.com/bmild/nerf/issues/24
#     cent = center if center is not None else [W / 2, H / 2]
#     directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)
#
#     return directions



# train_dataset = dataset(args.datadir, near=args.near, far=args.far, scene_bbox_stretch=args.scene_bbox_stretch,
#                         split='train', downsample=args.downsample_train, is_stack=False, use_sem=True,
#                         dino_feature_path=args.dino_feature_path)
# test_dataset = dataset(args.datadir, near=args.near, far=args.far, scene_bbox_stretch=args.scene_bbox_stretch,
#                        split='test', downsample=args.downsample_train, is_stack=True, use_sem=True)
# train_dataset.remap_sem_gt_label(train_dataset.sem_samples["sem_img"],
#                                  test_dataset.sem_samples["sem_img"],
#                                  args.sem_info_path)



class ReplicaDatasetDMNeRF(torch.utils.data.Dataset):
    """
    Dataset class for toydesk dataset
    """
    def __init__(self,
                 data_dir,
                 img_res,
                 split = 'train',
                 is_stack = False):

        self.root_dir = data_dir
        self.split = split
        self.img_total_num = len(glob(os.path.join(self.root_dir, "rgb", "rgb_*.png")))
        self.scene_bbox_stretch = 6.5
        img_w, img_h = 640, 480
        self.img_wh = (int(img_w ), int(img_h))
        self.total_pixels = img_w * img_h
        self.use_sem = True
        self.gen_rays = True
        self.downsample = 1.0
        self.is_stack = is_stack
        self.define_transforms()
        self.read_meta()

        # self.meta = {}
        # with open(os.path.join(data_dir, "transforms_full.json"), 'r') as f:
        #     self.meta = json.load(f)
        # self.train_list = np.loadtxt(os.path.join(data_dir, 'train.txt')).astype(int).tolist()
        # self.test_list = np.loadtxt(os.path.join(data_dir, 'test.txt')).astype(int).tolist()

        # self.img_res = img_res
        # self.total_pixels = self.img_res[0] * self.img_res[1]
        # self.sampling_idx = None
        #
        # train_imgs = []
        # train_segs = []
        # train_poses = []
        # test_imgs = []
        # test_segs = []
        # test_poses = []
        # all_imgs = []
        # all_sems = []
        # all_poses = []
        # # fix_rot = np.array([1, 0, 0, 0,
        # #                     0, -1, 0, 0,
        # #                     0, 0, -1, 0,
        # #                     0, 0, 0, 1]).reshape(4, 4)
        # fix_rot = np.array([1, 0, 0, 0,
        #                     0, 1, 0, 0,
        #                     0, 0, 1, 0,
        #                     0, 0, 0, 1]).reshape(4, 4)
        #
        # self.label_mapping = None
        # with open(os.path.join(data_dir, 'label_mapping.txt'), 'r') as f:
        #     content = f.readlines()
        #     self.label_mapping = [int(a) for a in content[0].split(',')]
        #
        #
        # self.center = None
        # if os.path.exists(os.path.join(data_dir, 'center.txt')):
        #     self.center = np.loadtxt(os.path.join(data_dir, 'center.txt')).reshape(4, 1)
        # self.center_mat = np.zeros([4, 4])
        # self.scale_mat = np.eye(4)
        # if self.center is not None:
        #     self.center_mat[:3, 3:] = self.center[:3]
        #     self.scale_mat[0, 0] = 1.0/self.center[-1]
        #     self.scale_mat[1, 1] = 1.0/self.center[-1]
        #     self.scale_mat[2, 2] = 1.0/self.center[-1]
        #
        # for frame in self.meta['frames'][::1]:
        #     if frame['idx'] in self.train_list:
        #         img_fname = os.path.join(data_dir, frame['file_path'])+'.png'
        #         seg_fname = os.path.join(data_dir, frame['file_path'])+'.instance.png'
        #         train_imgs.append(imageio.imread(img_fname).transpose(2, 0, 1).reshape(3, -1).transpose(1, 0)) # reshape to HW*3
        #         segs = imageio.imread(seg_fname).reshape(1, -1).transpose(1, 0)
        #         if self.label_mapping is not None:
        #             for i in self.label_mapping:
        #                 segs[segs == i] = self.label_mapping.index(i)
        #         train_segs.append(segs) # reshape to HW*1
        #         pose_matrix = np.array(frame['transform_matrix'] + self.center_mat)
        #         pose_matrix = self.scale_mat @ pose_matrix
        #         train_poses.append( pose_matrix @ fix_rot)
        #     elif frame['idx'] in self.test_list:
        #         img_fname = os.path.join(data_dir, frame['file_path'])+'.png'
        #         seg_fname = os.path.join(data_dir, frame['file_path'])+'.instance.png'
        #         test_imgs.append(imageio.imread(img_fname).transpose(2, 0, 1).reshape(3, -1).transpose(1, 0))
        #         segs = imageio.imread(seg_fname).reshape(1, -1).transpose(1, 0)
        #         if self.label_mapping is not None:
        #             for i in self.label_mapping:
        #                 segs[segs == i] = self.label_mapping.index(i)
        #         test_segs.append(segs)
        #
        #         pose_matrix = np.array(frame['transform_matrix'] + self.center_mat)
        #         pose_matrix= self.scale_mat @ pose_matrix
        #         test_poses.append( pose_matrix @ fix_rot)
        #     else:
        #         continue
        #
        # train_imgs = (np.array(train_imgs) / 255.).astype(np.float32)
        # train_segs = np.array(train_segs).astype(np.float32)
        # train_poses = np.array(train_poses).astype(np.float32)
        # test_imgs = (np.array(test_imgs) / 255.).astype(np.float32)
        # test_segs = np.array(test_segs).astype(np.float32)
        # test_poses = np.array(test_poses).astype(np.float32)
        # all_imgs.append(train_imgs)
        # all_imgs.append(test_imgs)
        # all_sems.append(train_segs)
        # all_sems.append(test_segs)
        # all_poses.append(train_poses)
        # all_poses.append(test_poses)
        #
        # self.i_split = [np.array(i) for i in [range(len(train_imgs)), range(len(train_imgs), len(train_imgs)+len(test_imgs))]]
        # self.imgs = np.concatenate(all_imgs, 0)
        # self.segs = np.concatenate(all_sems, 0)
        # self.poses = np.concatenate(all_poses, 0)
        # # print(self.poses.max(axis=0))
        # # print(self.poses.min(axis=0))
        # self.n_imgs = len(self.i_split)
        #
        # self.intrinsics_all = []
        #
        # w, h = self.img_wh
        # hfov = 90
        # focal= 0.5 * w / np.tan(0.5 * np.radians(hfov))  # w ?
        # self.focal_y = self.focal_x
        #
        # # focal = 0.5 * img_res[1] / np.tan(0.5* float(self.meta['camera_angle_x']))
        # for pose in self.poses:
        #     self.intrinsics_all.append(torch.from_numpy(np.array([
        #                                     [focal, 0, 0.5*self.img_res[1], 0],
        #                                     [0, focal, 0.5*self.img_res[0], 0],
        #                                     [0, 0, 1, 0],
        #                                     [0, 0, 0, 1]
        #                                     ])).float())

    def read_meta(self):
        w, h = self.img_wh

        hfov = 90
        self.focal_x = 0.5 * w / np.tan(0.5 * np.radians(hfov))  # w ?
        self.focal_y = self.focal_x
        cx = (w - 1.) / 2
        cy = (h - 1.) / 2

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, cx], [0, self.focal_y, cy], [0, 0, 1]]).float().cpu()
        self.intrinsics_objsdf = torch.from_numpy(np.array([
                                        [self.focal_x, 0, cx, 0],
                                        [0, self.focal_y,cy, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]
                                        ])).float()

        # load c2w for all images in the video
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        # self.image_paths = []
        # self.sem_paths = []
        # self.colored_sem_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_sems = []
        self.all_colored_sems = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0
        # for semantic labels remapping ?
        self.sem_samples = {}
        self.sem_samples["sem_img"] = []
        self.sem_samples["label_ins_map"] = {}
        self.sem_samples["ins_label_map"] = {}


        img_eval_interval = 5
        if self.split == "train":
            self.indices = list(range(0, self.img_total_num, img_eval_interval))
        elif self.split == "test":
            self.indices = list(range(img_eval_interval // 2, self.img_total_num, img_eval_interval))

        for i in tqdm(self.indices, desc=f'Loading data {self.split} ({len(self.indices)})'):  # img_list:#
            c2w = torch.FloatTensor(self.Ts_full[i])
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, "rgb", f"rgb_{i}.png")
            img = Image.open(image_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w), normalized
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
            self.all_rgbs += [img]

            if self.use_sem:
                sem_image_path = os.path.join(self.root_dir, 'semantic_instance', f"semantic_instance_{i}.png")
                sem_img = Image.open(sem_image_path)  # type: Image
                self.sem_samples["sem_img"].append(np.array(sem_img))

                if self.downsample != 1.0:
                    sem_img = sem_img.resize(self.img_wh, Image.LANCZOS)
                sem_img = self.transform(sem_img)
                sem_img = sem_img.view(-1, h*w).permute(1, 0)  # (h*w, 1)
                self.all_sems += [sem_img]

            # if self.load_colored_sem:
                # colored_sem_img_path = os.path.join(self.root_dir, "sem", f"{i:03d}.png")
                # # self.colored_sem_paths += colored_sem_img_path
                # colored_sem_img = Image.open(colored_sem_img_path)
                # if self.downsample != 1.0:
                #     colored_sem_img = colored_sem_img.resize(self.img_wh, Image.LANCZOS)
                # colored_sem_img = self.transform(colored_sem_img)
                # colored_sem_img = colored_sem_img.view(3, -1).permute(1, 0)  # (h*w, 3)
                # self.all_colored_sems += [colored_sem_img]

            if self.gen_rays:
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            # pixel-wise in training
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (num_imgs*h*w, 3)
        else:
            # image-wise in testing
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (num_imgs,h,w,3)

        if self.use_sem:
            if not self.is_stack:
                self.all_sems = torch.cat(self.all_sems, 0)  # (num_imgs*h*w, 1)
            else:
                self.all_sems = torch.stack(self.all_sems, 0).reshape(-1, *self.img_wh[::-1], 1)  # (num_imgs,h,w,1)

        # """load"""
        # if self.load_colored_sem:
        #     if not self.is_stack:
        #         self.all_colored_sems = torch.cat(self.all_colored_sems, 0)  # (num_imgs*h*w, 3)
        #     else:
        #         self.all_colored_sems = torch.stack(self.all_colored_sems, 0).reshape(-1, *self.img_wh[::-1], 3)  # (num_imgs,h,w,3)
        # """change from sem_maps"""

        self.poses = torch.stack(self.poses)

        if self.gen_rays:
            # used in train.py for aabb
            # adaptive scene_bbox
            all_rays_o = torch.stack(self.all_rays)[..., :3]  # for all images, (N_imgs, h*w, 3)
            all_rays_o = all_rays_o.reshape(-1, 3)

            scene_min = torch.min(all_rays_o, 0)[0] - self.scene_bbox_stretch
            scene_max = torch.max(all_rays_o, 0)[0] + self.scene_bbox_stretch

            self.scene_bbox = torch.stack([scene_min, scene_max]).reshape(-1, 3)

            self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
            self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0)  # (num_imgs*h*w, 3)
            else:
                self.all_rays = torch.stack(self.all_rays, 0)  # (num_imgs,h*w, 3)

            # render images from new view-points
            center = torch.mean(self.scene_bbox, dim=0)
            radius = torch.norm(self.scene_bbox[1]-center)*0.1
            up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
            pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
            self.render_path = gen_path(pos_gen, up=up, frames=200).float().cpu()
            self.render_path[:, :3, 3] += center

    def define_transforms(self):
        self.transform = T.ToTensor()
    # def __getitem__(self, idx):
    #     img = self.all_rgbs[idx]
    #     rays = self.all_rays[idx]
    #     sems = self.all_sems[idx]
    #
    #     sample = {'rays': rays,
    #               'rgbs': img,
    #               'sems': sems}
    #
    #     return sample

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        img = self.all_rgbs[idx]
        # rays = self.all_rays[idx]
        sems = self.all_sems[idx]

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_objsdf,
            "pose" : self.poses[idx]
            # "pose": torch.from_numpy(self.poses[idx]).float()
        }

        ground_truth = {
            "rgb": img,
            "segs": sems
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = img[self.sampling_idx, :]
            ground_truth["segs"] = sems[self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]
        return idx, sample, ground_truth

    def __len__(self):
        return len(self.all_rgbs)


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

if __name__ == '__main__':
    a = ReplicaDatasetDMNeRF('/data/dzy_data/nerf/datasets/nerf_replica/replica_ins/office_3/', [640, 480],'train')
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