import os

import imageio
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import lpips
from skimage import metrics

from utils import rend_util
from utils.metrics import ConfusionMatrix, calculate_ap
import matplotlib.pyplot as plt

# data_dir = "/root/autodl-tmp/datasets/nerf_replica/replica_ins/office_3/"
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seg_metric(gt_label_map,sem_label_map,sem_logit_map,img_res,num_semantic_class = 82):
    try:
        sem_logit_map = sem_logit_map.cpu().numpy()
        gt_label_map = gt_label_map.cpu().numpy().reshape(-1, 1)
        sem_label_map = sem_label_map.cpu().numpy().reshape(-1, 1)
    except:
        pass
    H, W = img_res

    val_cm = ConfusionMatrix(num_classes=num_semantic_class)
    metric_iou = val_cm.add_batch(sem_label_map, gt_label_map, return_miou=True)

    '''mAP'''
    confusion_matrix = val_cm.confusion_matrix

    unique_pred_labels = np.unique(sem_label_map)
    unique_gt_labels = np.unique(gt_label_map)
    num_valid_labels = len(unique_gt_labels)
    siou = np.divide(np.diag(confusion_matrix), (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0)
                                                 - np.diag(confusion_matrix) + 1e-6))
    iou_metrics = siou[unique_gt_labels]

    '''confidence values'''
    # prepare confidence values
    unique_pred_labels, unique_gt_labels = torch.from_numpy(unique_pred_labels), torch.from_numpy(unique_gt_labels)
    pred_label_map = torch.from_numpy(sem_label_map).reshape(H, W)
    pred_conf_mask = torch.from_numpy(sem_logit_map).reshape(H, W, num_semantic_class)
    conf_scores = torch.zeros_like(unique_gt_labels, dtype=torch.float32)
    for i, label in enumerate(unique_gt_labels):
        if label.item() in unique_pred_labels:
            index = torch.where(pred_label_map == label)
            ssm = pred_conf_mask[index[0], index[1]]  # confidence value
            pred_obj_conf = torch.median(ssm).item()  # median confidence value for one object
            conf_scores[i] = pred_obj_conf

    iou_metrics = torch.from_numpy(iou_metrics).to(device)
    conf_scores = conf_scores.to(device)

    thre_list = [0.5, 0.75]
    ap = calculate_ap(iou_metrics, num_valid_labels, thre_list, device, confidence=conf_scores,
                      function_select='integral')
    return ap,metric_iou
    # '''mIoU'''
    # metric_ious.append(metric_iou)
    #
    # all_ap.append(ap)

    # miou = np.mean(np.array(metric_ious))
    #
    # all_ap = np.array(all_ap)
    # mean_ap = np.mean(all_ap, axis=0)
    #
    #
    # sem_label_maps = np.stack(sem_label_maps, 0)
    # sem_label_maps_flatten = sem_label_maps.reshape(-1, 1)
    # gt_label_maps_flatten = gt_label_maps[0::img_eval_interval].reshape(-1, 1)
    # metric_iou_pix = val_cm_pix.add_batch(sem_label_maps_flatten, gt_label_maps_flatten, return_miou=True)
    # # metric_ious.append(metric_iou_pix)

def render_label2img(labels, ins_map):
    unique_labels = torch.unique(labels)
    labels = labels.cpu()
    unique_labels = unique_labels.cpu()
    h, w = labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        ra_se_im_t[labels == label] = ins_map[label]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t

def metric_test(index,plot_data,img_res):
    H, W = img_res
    rgb = plot_data['rgb_eval'].reshape([H, W, 3])
    gt_imgs = plot_data['rgb_gt'].reshape([H, W, 3])

    # rgb = lin2img(plot_data['rgb_eval'], img_res)
    # gt_imgs = lin2img(plot_data['rgb_gt'], img_res)
    lpips_vgg = lpips.LPIPS(net="vgg").to("cuda")
    # rgb.shape HW3
    # rgb image evaluation part
    psnr = metrics.peak_signal_noise_ratio(rgb.cpu().numpy(), gt_imgs.cpu().numpy(), data_range=1)
    ssim = metrics.structural_similarity(rgb.cpu().numpy(), gt_imgs.cpu().numpy(), multichannel=True, data_range=1,
                                         win_size=7, channel_axis=-1)
    # ssim = metrics.structural_similarity(rgb.cpu().numpy(), gt_imgs.cpu().numpy(), multichannel=True, data_range=1)
    # ssim = metrics.structural_similarity(rgb.permute(2, 0, 1).unsqueeze(0).cpu().numpy(), gt_imgs.permute(2, 0, 1).unsqueeze(0).cpu().numpy(), multichannel=True, data_range=1)
    lpips_i = lpips_vgg(rgb.permute(2, 0, 1).unsqueeze(0).to("cuda"), gt_imgs.permute(2, 0, 1).unsqueeze(0).to("cuda"))
    # psnrs.append(psnr)
    # ssims.append(ssim)
    # lpipses.append(lpips_i.item())
    print(f"index{index} PSNR: {psnr} SSIM: {ssim} LPIPS: {lpips_i.item()}")
    return psnr,ssim,lpips_i


def plot_test(implicit_network, indices, plot_data, path, index, img_res, plot_nimgs, resolution, grid_boundary, level=0,ins_map = None):

    if plot_data is not None:
        save_path = os.path.join(path, 'test')
        os.makedirs(save_path, exist_ok=True)

        H, W = img_res

        rgb = plot_data['rgb_eval'].reshape([H, W, 3])
        rgb8 = to8b(rgb.cpu().numpy())
        filename = os.path.join(save_path, '{:03d}.png'.format(index))
        imageio.imwrite(filename, rgb8)

        # plot semantic_image
        if 'semantic_map' in plot_data.keys():

            gt_label = plot_data['seg_gt'].reshape([H, W]).long()
            gt_ins_img = render_label2img(gt_label, ins_map)
            gt_img_file = os.path.join(save_path, f'{index}_ins_gt.png')
            imageio.imwrite(gt_img_file, gt_ins_img)
            gt_ins_file = os.path.join(save_path, f'{index}_ins_gt_mask.png')
            imageio.imwrite(gt_ins_file, np.array(gt_label.cpu().numpy(), dtype=np.uint8))

            pred_label = plot_data['semantic_map'].reshape([H, W]).long()
            ins_img = render_label2img(pred_label,ins_map)
            pred_ins_file = os.path.join(save_path, f'{index}_ins_pred_mask.png')
            fileins = os.path.join(save_path, f"instance_{str(index).zfill(3)}.png")
            imageio.imwrite(pred_ins_file, np.array(pred_label.cpu().numpy(), dtype=np.uint8))
            imageio.imwrite(fileins, ins_img)



        # plot_seg_images(plot_data['semantic_map'], plot_data['seg_gt'], path, index, plot_nimgs, img_res)
        # plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, index, plot_nimgs, img_res)
        #
        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], save_path, index, plot_nimgs, img_res)
    psnr,ssim,lpips_i = metric_test(index, plot_data, img_res)
    pred_logit = plot_data['semantic_logit_map'].reshape([H, W,plot_data['semantic_logit_map'].shape[-1]])
    ap,iou = seg_metric(gt_label, pred_label, pred_logit, img_res, num_semantic_class=plot_data['num_class'])
    return psnr,ssim,lpips_i,ap,iou

def plot(implicit_network, indices, plot_data, path, epoch, img_res, plot_nimgs, resolution, grid_boundary, level=0):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot semantic_image
        if 'semantic_map' in plot_data.keys():
            plot_seg_images(plot_data['semantic_map'], plot_data['seg_gt'], path, epoch, plot_nimgs, img_res)
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res)

        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], path, epoch, plot_nimgs, img_res)

        if 'occ_map' in plot_data.keys():
            plot_opacity_maps(plot_data['occ_map'], path, epoch, plot_nimgs, img_res)


    data = []

    # plot surface
    # get semantic number
    if 'semantic_map' in plot_data.keys():
        sem_num = implicit_network.d_out
        f = torch.nn.MaxPool1d(sem_num)
        for indx in range(sem_num):
            _ = get_semantic_surface_trace(path=path,
                                            epoch=epoch,
                                            #    sdf=lambda x: -f(-implicit_network(x)[:, :6]),
                                            sdf = lambda x: implicit_network(x)[:, indx],
                                            resolution=resolution,
                                            grid_boundary=grid_boundary,
                                            level=level,
                                            idx= indx
                                            )

        surface_traces = get_surface_trace(path=path,
                                    epoch=epoch,
                                    sdf=lambda x: -f(-implicit_network(x)[:, :sem_num].unsqueeze(1)).squeeze(-1),
                                    # sdf = lambda x: implicit_network(x)[:, indx],
                                    resolution=resolution,
                                    grid_boundary=grid_boundary,
                                    level=level
                                    )
    else:
        surface_traces = get_surface_trace(path=path,
                                    epoch=epoch,
                                    sdf=lambda x: implicit_network(x)[:, 0],
                                    resolution=resolution,
                                    grid_boundary=grid_boundary,
                                    level=level
                                    )
    if surface_traces is not None:
        data.append(surface_traces[0])


    # plot cameras locations
    if plot_data is not None:
        for i, loc, dir in zip(indices, cam_loc, cam_dir):
            data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-6, 6], autorange=False),
                      yaxis=dict(range=[-6, 6], autorange=False),
                      zaxis=dict(range=[-6, 6], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1200, height=1200, showlegend=True)
    filename = '{0}/surface_{1}.html'.format(path, epoch)
    offline.plot(fig, filename=filename, auto_open=False)


def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_semantic_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0, idx=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                 grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]),
                allow_degenerate=True)
        except:
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                 grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                 grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))
        except:
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                 grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100, grid_boundary=[-2.0, 2.0], level=0, take_components=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100, grid_boundary)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))
    except:
        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))
    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    if take_components:
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                                 grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))
        except:
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                                 grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))
        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_surface_by_grid(grid_params, sdf, resolution=100, level=0, higher_res=False):
    grid_params = grid_params * [[1.5], [1.0]]

    # params = PLOT_DICT[scan_id]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()

    if higher_res:
        # get low res mesh to sample point cloud
        grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)
        z = []
        points = grid['grid_points']

        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        z = z.astype(np.float32)
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                 grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))
        except:
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                 grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))
        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()

        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()

        grid_aligned = get_grid(helper.cpu(), resolution, eps=0.01)
    else:
        grid_aligned = get_grid(None, resolution, input_min=input_min, input_max=input_max, eps=0.0)

    grid_points = grid_aligned['grid_points']

    if higher_res:
        g = []
        for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                                 grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))
        except:
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                                 grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                         grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))
        if higher_res:
            verts = torch.from_numpy(verts).cuda().float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()
        else:
            verts = verts + np.array([grid_aligned['xyz'][0][0], grid_aligned['xyz'][1][0], grid_aligned['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, normals)

        # CUTTING MESH ACCORDING TO THE BOUNDING BOX
        if higher_res:
            bb = grid_params
            transformation = np.eye(4)
            transformation[:3, 3] = (bb[1,:] + bb[0,:])/2.
            bounding_box = trimesh.creation.box(extents=bb[1,:] - bb[0,:], transform=transformation)

            meshexport = meshexport.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)

    return meshexport

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_normal_maps(normal_maps, path, epoch, plot_nrow, img_res):
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/normal_{1}.png'.format(path, epoch))

def plot_opacity_maps(occ_maps, path, epoch, plot_nrow, img_res):
    occ_map_plot = lin2img(occ_maps, img_res)

    tensor = torchvision.utils.make_grid(occ_map_plot,
                                        scale_each=False,
                                        normalize=False,
                                        nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/occ_{1}.png'.format(path, epoch))



def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = ground_true.cuda()

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

def colored_data(x, cmap='jet', d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:,:,:3]).astype(np.uint8) # H, W, C

def plot_seg_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = ground_true.cuda()
    
    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)[:, :, 0]
    tensor = colored_data(tensor)

    img = Image.fromarray(tensor)
    img.save('{0}/semantic_{1}.png'.format(path, epoch))

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
