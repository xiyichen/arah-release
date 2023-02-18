import os
import torch
# import trimesh
import glob
import json
import shutil
import argparse
import cv2
from PIL import Image
import numpy as np
import preprocess_datasets.easymocap.mytools.camera_utils as cam_utils

from scipy.spatial.transform import Rotation

from human_body_prior.body_model.body_model import BodyModel

from preprocess_datasets.easymocap.smplmodel import load_model

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Preprocessing for ZJU-MoCap.'
)
parser.add_argument('--data-dir', type=str, help='Directory that contains raw iphone data.')
parser.add_argument('--out-dir', type=str, help='Directory where preprocessed data is saved.')
parser.add_argument('--seqname', type=str, default='space-out-arah', help='Sequence to process.')

def world_to_cam(world_coord, R, t):
    cam_coord = (R@world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord

if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)
    out_dir = os.path.join(args.out_dir, seq_name)

    annots = np.load(os.path.join(data_dir, 'annots.npy'), allow_pickle=True).item()
    alignments = np.load(os.path.join(data_dir, 'alignments.npy'), allow_pickle=True).item()
    cameras = annots['cams']

    smpl_dir = os.path.join(data_dir, 'new_params')
    verts_dir = os.path.join(data_dir, 'new_vertices')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1)
    if torch.cuda.is_available():
        body_model = body_model.cuda()

    faces = np.load('body_models/misc/faces.npz')['faces']

    cam_names = ['0', '1', '2']

    all_cam_params = {'all_cam_names': cam_names}
    smpl_out_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(smpl_out_dir):
        os.makedirs(smpl_out_dir)

    for cam_idx, cam_name in enumerate(cam_names):
        cam_out_dir = os.path.join(out_dir, cam_name)
        if not os.path.exists(cam_out_dir):
            os.makedirs(cam_out_dir)

        img_in_dir = os.path.join(data_dir, 'Camera_{}'.format(cam_name))
        mask_in_dir = os.path.join(data_dir, 'mask', 'Camera_{}'.format(cam_name))

        img_files = sorted(glob.glob(os.path.join(img_in_dir, '*.png')))
        K = cameras['K'][cam_idx]
        for key in K:
            K[key] = K[key].tolist()
        D = cameras['D'][cam_idx]
        for key in D:
            D[key] = D[key].tolist()
        R = cameras['R'][cam_idx]
        for key in R:
            R[key] = R[key].tolist()
        T = cameras['T'][cam_idx]
        for key in T:
            T[key] = T[key].tolist()
        cam_params = {'K': K, 'D': D, 'R': R, 'T': T}
        all_cam_params.update({cam_name: cam_params})
        for img_file in img_files:
            print ('Processing: {}'.format(img_file))
            frame_id = int(img_file.split('/')[-1].split('.')[0])
            K = cameras['K'][cam_idx][frame_id].tolist()
            D = cameras['D'][cam_idx][frame_id].tolist()
            R = cameras['R'][cam_idx][frame_id].tolist()

            R_np = np.array(R)
            T = cameras['T'][cam_idx][frame_id]
            T_np = np.array(T).reshape(3, 1) / 1000.0
            T = T_np.tolist()
            idx = int(os.path.basename(img_file)[:-4])
            frame_index = idx

            mask_file = os.path.join(mask_in_dir, os.path.basename(img_file)[:-4] + '.png')
            smpl_file = os.path.join(smpl_dir, '{}.npy'.format(idx))
            verts_file = os.path.join(verts_dir, '{}.npy'.format(idx))

            if not os.path.exists(smpl_file):
                print ('Cannot find SMPL file for {}: {}, skipping'.format(img_file, smpl_file))
                continue

            # We only process SMPL parameters in world coordinate
            if cam_idx == 0:
                params = np.load(smpl_file, allow_pickle=True).item()

                root_orient = Rotation.from_rotvec(np.array(params['Rh']).reshape([-1])).as_matrix()
                trans = np.array(params['Th']).reshape([3, 1])

                betas = np.array(params['shapes'], dtype=np.float32)
                poses = np.array(params['poses'], dtype=np.float32)
                pose_body = poses[:, 3:66].copy()
                pose_hand = poses[:, 66:].copy()

                poses_torch = torch.from_numpy(poses)
                pose_body_torch = torch.from_numpy(pose_body)
                pose_hand_torch = torch.from_numpy(pose_hand)
                betas_torch = torch.from_numpy(betas)

                if torch.cuda.is_available():
                    poses_torch = poses_torch.cuda()
                    pose_body_torch = pose_body_torch.cuda()
                    pose_hand_torch = pose_hand_torch.cuda()
                    betas_torch = betas_torch.cuda()

                new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
                new_trans = trans.reshape([1, 3]).astype(np.float32)

                new_root_orient_torch = torch.from_numpy(new_root_orient)
                new_trans_torch = torch.from_numpy(new_trans)
                if torch.cuda.is_available():
                    new_root_orient_torch = new_root_orient_torch.cuda()
                    new_trans_torch = new_trans_torch.cuda()

                # Get shape vertices
                body = body_model(betas=betas_torch)
                minimal_shape = body.v.detach().cpu().numpy()[0]

                # Get bone transforms
                # body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)

                # body_model_em = load_model(gender='neutral', model_type='smpl')
                # verts = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()

                # vertices = body.v.detach().cpu().numpy()[0]
                # new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)
                # new_trans_torch = torch.from_numpy(new_trans).cuda()
                transf = alignments['0_' + str(int(img_file.split('/')[-1].split('.')[0])).zfill(5) + '.png']['transformation']
                smpl_vertices2d = alignments['0_' + str(int(img_file.split('/')[-1].split('.')[0])).zfill(5) + '.png']['smpl_vertices2d'][:, :2]
                
                # translation loss minimization
                translation = torch.tensor(trans.reshape(1, 3), requires_grad=True, device=poses_torch.device)
                optim_list = [
                    {"params": translation, "lr": 1e-3},
                ]
                optim = torch.optim.Adam(optim_list)
                total_iters = 1000
                for i in tqdm(range(total_iters), total=total_iters):
                    body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=translation)
                    vertices = body.v[0]
                    vertices_world = world_to_cam(vertices, torch.tensor(R, device=poses_torch.device), torch.tensor(T, device=poses_torch.device))
                    projected_v = vertices_world / vertices_world[:, -1:]
                    projected_v = torch.einsum('ij,kj->ki', torch.tensor(K, device=poses_torch.device), projected_v)[:, :2].float()
                    optim.zero_grad()
                    loss = torch.nn.functional.mse_loss(projected_v, torch.tensor(smpl_vertices2d, device=poses_torch.device).float())
                    loss.backward()
                    optim.step()
                print(loss, translation)
                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=translation)
                vertices = body.v[0]
                vertices_world = world_to_cam(vertices, torch.tensor(R, device=poses_torch.device), torch.tensor(T, device=poses_torch.device))
                projected_v = vertices_world / vertices_world[:, -1:]
                projected_v = torch.einsum('ij,kj->ki', torch.tensor(K, device=poses_torch.device), projected_v)[:, :2].float()
                
                image = cv2.imread(img_file)
                for idx, loc in enumerate(smpl_vertices2d):
                    x = int(loc[0])
                    y = int(loc[1])
                    cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
                Image.fromarray(image[:, :, ::-1]).save(os.path.join(out_dir, 'smpl_projections', img_file.split('/')[-1]))

                bone_transforms = body.bone_transforms.detach().cpu().numpy()
                Jtr_posed = body.Jtr.detach().cpu().numpy()

                out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(idx))

                np.savez(out_filename,
                         minimal_shape=minimal_shape,
                         betas=betas,
                         Jtr_posed=Jtr_posed[0],
                         bone_transforms=bone_transforms[0],
                         trans=translation.detach().cpu().numpy()[0],
                         root_orient=new_root_orient[0],
                         pose_body=pose_body[0],
                         pose_hand=pose_hand[0])

            shutil.copy(os.path.join(img_file), os.path.join(cam_out_dir, '{:06d}.jpg'.format(idx)))
            shutil.copy(os.path.join(mask_file), os.path.join(cam_out_dir, '{:06d}.png'.format(idx)))

    with open(os.path.join(out_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)
