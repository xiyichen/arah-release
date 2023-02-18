from smpl_numpy import SMPL
import glob
import json
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
import cv2
from easymocap.smplmodel import load_model

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    if len(rot_vecs.shape) > 2:
        rot_vec_ori = rot_vecs
        rot_vecs = rot_vecs.view(-1, 3)
    else:
        rot_vec_ori = None
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    if rot_vec_ori is not None:
        rot_mat = rot_mat.reshape(*rot_vec_ori.shape[:-1], 3, 3)
    return rot_mat

def batch_rot2aa(Rs):
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return (theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)).detach().cpu().numpy()

smpl_model = SMPL(sex='male', model_dir='./smpl_models')
pare_frames = glob.glob('./smpl_pred/*.pkl')
pare_frames.sort()
alignments = np.load('./alignments.npy', allow_pickle=True).item()


for pare_frame_fn in pare_frames:
	transf = alignments[pare_frame_fn.split('/')[-1].split('.')[0] + '.png']['transformation']
	pare_frame = joblib.load(pare_frame_fn)
	pose = batch_rot2aa(torch.tensor(pare_frame['pred_pose'][0])).reshape(-1, 1)
	shape = pare_frame['pred_shape'][0]
	# joints = pare_frame['smpl_joints3d']
	# j0 = joints[:, 0, :].reshape(1, 3)
	# cam = pare_frame['orig_cam'][0]
	# cam_s = cam[0:1]
	# cam_pos = cam[2:]
	# w = 360
	# flength = w / 2.
	# tz = flength / (0.5 * w * cam_s)
	# trans = -np.hstack([cam_pos, tz])
	# print(pare_frame['smpl_joints2d'])
	# exit()

	pose = pose.reshape(-1, 3)
	Rh = Rot.from_matrix(transf[:3, :3] @ Rot.from_rotvec(pose[0, :]).as_matrix()).as_rotvec().reshape(1, 3)
	# Th = transf[3, :].reshape(1, 3) + j0 - np.einsum('bij,bj->bi', Rot.from_rotvec(Rh).as_matrix(), j0).reshape(1, 3)
	Th = transf[3, :].reshape(1, 3)
	# Th = (transf[:3, :3]@pose[0, :].reshape(3, 1)).reshape(1, 3) + transf[3, :].reshape(1, 3) - pose[0, :].reshape(1, 3)
	# Th = np.einsum('bij,bj->bi', transf[:3, :3].reshape(1, 3, 3), (trans + pose[0, :]).reshape(1, 3))[0] + transf[3, :] - pose[0, :]


	# exit()
	# Th += ...

	# body_model_em = load_model(gender='neutral', model_type='smpl')
	# verts = body_model_em(poses=torch.from_numpy(pose.reshape(1, 72).astype(np.float32)), shapes=torch.from_numpy(shape.reshape(1, 10).astype(np.float32)), Rh=torch.from_numpy(Rh.astype(np.float32)), Th=torch.from_numpy(Th.astype(np.float32)), return_verts=True)[0].detach().cpu().numpy()
	# vertices = (transf[:3, :3] @ pare_frame['smpl_vertices'][0].T).T
	# Th = Th + (verts - vertices).mean(0, keepdims=True).reshape(1, 3)

	pose = pose.reshape(1, -1)
	# pose[0, :3] = 0
	shape = shape.reshape(1, -1)
	# Rh = Rh.reshape(1, -1)
	# Th = Th.reshape(1, -1)
	np.save('./new_params/' + str(int(pare_frame_fn.split('/')[-1].split('.')[0])) + '.npy', {'poses': pose, 'Rh': Rh, 'Th': Th, 'shapes': shape, 'o': Rot.from_matrix(transf[:3, :3]).as_rotvec().reshape(1, 3), 't': transf[3, :].reshape(1, 3)})

annots = {}
annots['cams'] = {}
annots['cams']['K'] = {}
annots['cams']['D'] = {}
annots['cams']['R'] = {}
annots['cams']['T'] = {}
for cam_name in [0, 1, 2]:
	fns = glob.glob('./Camera_' + str(cam_name) + '/*.png')
	fns.sort()
	annots['cams']['K'][cam_name] = {}
	annots['cams']['D'][cam_name] = {}
	annots['cams']['R'][cam_name] = {}
	annots['cams']['T'][cam_name] = {}
	for fn in fns:
		frame_id = int(fn.split('/')[-1].split('.')[0])
		with open('./cam_params/' + str(cam_name) + '_' + str(frame_id).zfill(5) + '.json', 'r') as f:
			cam_file = json.load(f)
		focal_length = cam_file['focal_length']/2
		principal_point = cam_file['principal_point']
		K = np.zeros((3, 3))
		K[0][0] = focal_length
		K[1][1] = focal_length
		K[2][2] = 1
		K[0][2] = principal_point[0]/2
		K[1][2] = principal_point[1]/2
		o = np.array(cam_file['orientation'])
		p = np.array(cam_file['position'])
		R = o
		T = -o @ p
		annots['cams']['K'][cam_name][frame_id] = K.reshape(3, 3)
		annots['cams']['D'][cam_name][frame_id] = np.zeros((5, 1))
		annots['cams']['R'][cam_name][frame_id] = R.reshape(3, 3)
		annots['cams']['T'][cam_name][frame_id] = T.reshape(3, 1)
np.save('./annots.npy', annots)