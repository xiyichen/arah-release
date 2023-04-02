import os
import torch
# import trimesh
import glob
import json
import shutil
import argparse
import cv2
from PIL import Image
from tqdm import tqdm

import numpy as np
import preprocess_datasets.easymocap.mytools.camera_utils as cam_utils

from scipy.spatial.transform import Rotation

from human_body_prior.body_model.body_model import BodyModel

from preprocess_datasets.easymocap.smplmodel import load_model

parser = argparse.ArgumentParser(
    description='Preprocessing for ZJU-MoCap.'
)
parser.add_argument('--data-dir', type=str, help='Directory that contains raw ZJU-MoCap data.')
parser.add_argument('--out-dir', type=str, help='Directory where preprocessed data is saved.')
parser.add_argument('--seqname', type=str, default='CoreView_313', help='Sequence to process.')

def world_to_cam(world_coord, R, t):
    cam_coord = (R@world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord

def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal

def transform_pose(W2C, translate, scale):
    C2W = np.linalg.inv(W2C)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.linalg.inv(C2W)

if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)
    out_dir = os.path.join(args.out_dir, seq_name)

    annots = np.load(os.path.join(data_dir, 'annots.npy'), allow_pickle=True).item()
    cameras = annots['cams']

    smpl_dir = os.path.join(data_dir, 'new_params')
    verts_dir = os.path.join(data_dir, 'new_vertices')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1)
    
    if torch.cuda.is_available():
        body_model = body_model.cuda()

    faces = np.load('body_models/misc/faces.npz')['faces']

    if seq_name in ['CoreView_313', 'CoreView_315']:
        cam_names = list(range(1, 20)) + [22, 23]
    else:
        cam_names = list(range(1, 24))

    cam_names = [str(cam_name) for cam_name in cam_names]

    all_cam_params = {'all_cam_names': cam_names}
    smpl_out_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(smpl_out_dir):
        os.makedirs(smpl_out_dir)
    
    cam_centers = []

    for cam_idx, cam_name in enumerate(cam_names):
        if seq_name in ['CoreView_313', 'CoreView_315']:
            K = cameras['K'][cam_idx]
            D = cameras['D'][cam_idx]
            R = cameras['R'][cam_idx]
        else:
            K = cameras['K'][cam_idx].tolist()
            D = cameras['D'][cam_idx].tolist()
            R = cameras['R'][cam_idx].tolist()

        R_np = np.array(R)
        T = cameras['T'][cam_idx]
        T_np = np.array(T).reshape(3, 1) / 1000.0
        T = T_np.tolist()
    
        W2C = np.eye(4)
        W2C[:3, :3] = R_np
        W2C[:3, 3] = T_np.reshape(3,)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
        

        cam_params = {'K': K, 'D': D, 'R': R, 'T': T}
        all_cam_params.update({cam_name: cam_params})
    target_radius = 3.
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    for cam_idx, cam_name in enumerate(cam_names):
        if seq_name in ['CoreView_313', 'CoreView_315']:
            K = cameras['K'][cam_idx]
            D = cameras['D'][cam_idx]
            R = cameras['R'][cam_idx]
        else:
            K = cameras['K'][cam_idx].tolist()
            D = cameras['D'][cam_idx].tolist()
            R = cameras['R'][cam_idx].tolist()

        R_np = np.array(R)
        T = cameras['T'][cam_idx]
        T_np = np.array(T).reshape(3, 1) / 1000.0
        T = T_np.tolist()

        W2C = np.eye(4)
        W2C[:3, :3] = R_np
        W2C[:3, 3] = T_np.reshape(3,)
        W2C = transform_pose(W2C, translate, scale)
        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        R = W2C[:3, :3].tolist()
        T_new = W2C[:3, 3].reshape(3, 1)

        T_diff = T_new - T_np

        cam_params = {'K': K, 'D': D, 'R': R, 'T': T_new.tolist(), 'T_diff': T_diff.tolist()}
        all_cam_params.update({cam_name: cam_params})


        cam_out_dir = os.path.join(out_dir, cam_name)
        if not os.path.exists(cam_out_dir):
            os.makedirs(cam_out_dir)
        
        if seq_name in ['CoreView_313', 'CoreView_315']:
            img_in_dir = os.path.join(data_dir, 'Camera ({})'.format(cam_name))
            mask_in_dir = os.path.join(data_dir, 'mask_cihp/Camera ({})'.format(cam_name))
        else:
            img_in_dir = os.path.join(data_dir, 'Camera_B{}'.format(cam_name))
            mask_in_dir = os.path.join(data_dir, 'mask_cihp/Camera_B{}'.format(cam_name))

        img_files = sorted(glob.glob(os.path.join(img_in_dir, '*.jpg')))

        for img_file in img_files:
            print ('Processing: {}'.format(img_file))
            if seq_name in ['CoreView_313', 'CoreView_315']:
                idx = int(os.path.basename(img_file).split('_')[4])
                frame_index = idx - 1
            else:
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

                # solve new Th for easymocap
                body_model_em = load_model(gender='neutral', model_type='smpl')
                verts_em = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0]
                vertices_world_expected_em = world_to_cam(verts_em, torch.tensor(R), torch.tensor(T_np))
                V_wo_trans = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=torch.zeros((1,3)), return_verts=True)[0]
                Th_new_em = ((torch.tensor(np.linalg.inv(R)) @ (vertices_world_expected_em - (torch.tensor(R)@V_wo_trans.T).T - T_new.reshape(1, 3)).T).T).mean(0, keepdims=True).float()
                verts_em_new = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=Th_new_em, return_verts=True)[0]
                vertices_world_em_new = world_to_cam(verts_em_new, torch.tensor(R), torch.tensor(T_new))
                assert((vertices_world_em_new - vertices_world_expected_em).mean(axis=0).max() < 1e-5)

                # solve new Th for standard SMPL
                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)
                verts_bm = body.v[0]
                vertices_world_expected_bm = world_to_cam(verts_bm, torch.tensor(R), torch.tensor(T_np))
                V_wo_trans = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=torch.zeros((1, 3))).v[0]
                Th_new_bm = ((torch.tensor(np.linalg.inv(R)) @ (vertices_world_expected_bm - (torch.tensor(R)@V_wo_trans.T).T - T_new.reshape(1, 3)).T).T).mean(0, keepdims=True).float()
                verts_bm_new = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=Th_new_bm).v[0]
                vertices_world_bm_new = world_to_cam(verts_bm_new, torch.tensor(R), torch.tensor(T_new))
                assert((vertices_world_bm_new - vertices_world_expected_bm).mean(axis=0).max() < 1e-5)
                Th_new = (verts_em - verts_bm + Th_new_em).mean(0, keepdims=True)

                new_trans_torch = Th_new

                body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)
                vertices = body.v[0]
                vertices_world = world_to_cam(vertices, torch.tensor(R), torch.tensor(T_new))
                projected_v = vertices_world / vertices_world[:, -1:]
                projected_v = np.einsum('ij,kj->ki', torch.tensor(K), projected_v)[:, :2]
                
                image = cv2.imread(img_file)
                for idx_, loc in enumerate(projected_v):
                    x = int(loc[0])
                    y = int(loc[1])
                    cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
                Image.fromarray(image[:, :, ::-1]).save('/cluster/scratch/xiychen/CoreView_387/CoreView_387_preprocessed/CoreView_387/vis_smpl/' + img_file.split('/')[-1])
                


                bone_transforms = body.bone_transforms.detach().cpu().numpy()
                Jtr_posed = body.Jtr.detach().cpu().numpy()

                out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(idx))

                np.savez(out_filename,
                         minimal_shape=minimal_shape,
                         betas=betas,
                         Jtr_posed=Jtr_posed[0],
                         bone_transforms=bone_transforms[0],
                         trans=new_trans[0],
                         root_orient=new_root_orient[0],
                         pose_body=pose_body[0],
                         pose_hand=pose_hand[0])
            shutil.copy(os.path.join(img_file), os.path.join(cam_out_dir, '{:06d}.jpg'.format(idx)))
            shutil.copy(os.path.join(mask_file), os.path.join(cam_out_dir, '{:06d}.png'.format(idx)))

    with open(os.path.join(out_dir, 'cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)
