import re, math, os, joblib, glob, torch, cv2, json
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from human_body_prior.body_model.body_model import BodyModel
import collections
from PIL import Image

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, PerspectiveCameras
)


def silhouette_renderer_from_pinhole_cam(fx, fy, width, height, cx, cy, device='cpu'):
    focal_length = torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[width - cx, height - cy]])  # In PyTorch3D, we assume that +X points left, and +Y points up and +Z points out from the image plane.
    image_size = torch.tensor([[height, width]])
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=image_size, device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=10,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    return silhouette_renderer

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True
    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    
def read_camera(cameras_txt_path):
    with open(cameras_txt_path, "r") as fid:
        line = fid.readline()
        assert line == '# Camera list with one line of data per camera:\n'
        line = fid.readline()
        assert line == '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
        line = fid.readline()
        assert re.search('^# Number of cameras: \d+\n$', line)
        num_cams = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

        elems = fid.readline().split()
        camera_id = int(elems[0])
        if elems[1] == 'SIMPLE_RADIAL':
            width, height, focal_length, cx, cy, radial = list(map(float, elems[2:]))
            fx = fy = focal_length
        elif elems[1] == 'PINHOLE':
            width, height, fx, fy, cx, cy = list(map(float, elems[2:]))
        elif elems[1] == 'OPENCV':
            width, height, fx, fy, cx, cy, k1, k2, k3, k4 = list(map(float, elems[2:]))
        else:
            raise ValueError(f'unsupported camera: {elems[1]}')
    return width, height, fx, fy, cx, cy

def read_camera_params(cam_path):
    cam_params = {'K': {}, 'D': {}, 'R': {}, 'T': {}}
    images_txt_path = os.path.join(cam_path, 'images.txt')
    cam_txt_path = os.path.join(cam_path, 'cameras.txt')
    width, height, fx, fy, cx, cy = read_camera(cam_txt_path)
    with open(images_txt_path, "r") as fid:
        line = fid.readline()
        assert line == '# Image list with two lines of data per image:\n'
        line = fid.readline()
        assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
        line = fid.readline()
        assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
        line = fid.readline()
        assert re.search('^# Number of images: \d+, mean observations per image: [-+]?\d*\.\d+|\d+\n$', line)
        num_images, mean_ob_per_img = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        num_images = int(num_images)
        mean_ob_per_img = float(mean_ob_per_img)

        for _ in tqdm(range(num_images), desc='reading camera params'):
            elems = fid.readline().split()
            assert len(elems) == 10
            line = fid.readline()
            image_id = int(elems[0]) - 1
            qw, qx, qy, qz, tx, ty, tz = list(map(float, elems[1:8]))
            K = np.zeros((3, 3))
            K[0][0] = fx
            K[1][1] = fy
            K[2][2] = 1
            K[0][2] = cx
            K[1][2] = cy
            T = np.array([tx, ty, tz], dtype=np.float32)
            R = quaternion_matrix(np.array([qw, qx, qy, qz], dtype=np.float32))[:3, :3].T
            D = np.zeros((5, 1))
            cam_params['K'][image_id] = K
            cam_params['D'][image_id] = D
            cam_params['R'][image_id] = R
            cam_params['T'][image_id] = T
    for key in cam_params:
        cam_params[key] = collections.OrderedDict(sorted(cam_params[key].items()))
    
    return cam_params


def world_to_cam(world_coord, R, t):
    cam_coord = (R@world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord

def coco_to_smpl(coco2d):
    '''
    input 2d joints in coco dataset format,
    and out 2d joints in SMPL format.
    Non-overlapping joints are set to 0s. 
    '''
    assert coco2d.shape == (17, 2)
    smpl2d = np.zeros((24, 2))
    smpl2d[1]  = coco2d[11] # leftUpLeg
    smpl2d[2]  = coco2d[12] # rightUpLeg
    smpl2d[4]  = coco2d[13] # leftLeg
    smpl2d[5]  = coco2d[14] # rightLeg
    smpl2d[7]  = coco2d[15] # leftFoot
    smpl2d[8]  = coco2d[16] # rightFoot
    smpl2d[16] = coco2d[5]  # leftArm
    smpl2d[17] = coco2d[6]  # rightArm
    smpl2d[18] = coco2d[7]  # leftForeArm
    smpl2d[19] = coco2d[8]  # rightForeArm
    smpl2d[20] = coco2d[9]  # leftHand
    smpl2d[21] = coco2d[10] # rightHand
    return smpl2d

def read_obj(path):
    vert = []
    uvs = []
    faces = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line[:2] == 'v ':
                v = line[2:].split()
                v = [float(i) for i in v]
                vert.append(np.array(v))
            elif line[:3] == 'vt ':
                uv = line[3:].split()
                uv = [float(i) for i in uv]
                uvs.append(np.array(uv))
            elif line[:2] == 'f ':
                f = line[2:].split()
                fv = [int(i.split('/')[0]) for i in f]
                ft = [int(i.split('/')[1]) for i in f]
                faces.append(np.array(fv + ft))

    vert = np.array(vert)
    uvs = np.array(uvs)
    faces = np.array(faces) - 1
    return vert, uvs, faces

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base_dir = '/cluster/scratch/xiychen/neuman_data/bike/'
cam_params = read_camera_params(os.path.join(base_dir, 'sparse'))
alignments = np.load(os.path.join(base_dir, 'alignments.npy'), allow_pickle=True)
smpl_optimized = joblib.load(os.path.join(base_dir, 'smpl_output_optimized_romp.pkl'))
body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1)
faces = np.load('body_models/misc/faces.npz')['faces']
smpl_params = {}
for key in tqdm(range(len(alignments.item().keys())), desc='reading optimized smpl params'):
    transf = alignments.item()[str(key).zfill(5) + '.png']
    poses = smpl_optimized[1]['pose'][key].reshape(24, 3)
    shape = smpl_optimized[1]['betas'][key]
    Rh = Rot.from_matrix(transf[:3, :3] @ Rot.from_rotvec(poses[0, :]).as_matrix()).as_rotvec().reshape(1, 3)
    Th = transf[3, :].reshape(1, 3)
    smpl_params[key] = {'Rh': Rh, 'Th': Th, 'poses': poses, 'shapes': shape}

cam_names = ['0']

all_cam_params = {'all_cam_names': cam_names}

img_in_dir = os.path.join(base_dir, 'Camera_0')
# mask_in_dir = os.path.join(data_dir, 'mask', 'Camera_{}'.format(cam_name))

img_files = sorted(glob.glob(os.path.join(img_in_dir, '*.png')))
K = cam_params['K']
for key in K:
    K[key] = K[key].tolist()
D = cam_params['D']
for key in D:
    D[key] = D[key].tolist()
R = cam_params['R']
for key in R:
    R[key] = R[key].tolist()
T = cam_params['T']
for key in T:
    T[key] = T[key].tolist()
cam_params = {'K': K, 'D': D, 'R': R, 'T': T}
all_cam_params.update({'0': cam_params})
with open(os.path.join(base_dir, 'cam_params.json'), 'w') as f:
    json.dump(all_cam_params, f)

_, uvs, faces = read_obj(
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'body_models/misc/smpl_uv.obj')
)
faces = torch.tensor(faces, device=device)
smpl_out_dir = os.path.join(base_dir, 'models')
arah_optimized_dir = os.path.join(base_dir, 'arah_optimized_smpl_projections')
if not os.path.exists(smpl_out_dir):
    os.makedirs(smpl_out_dir)
if not os.path.exists(arah_optimized_dir):
    os.makedirs(arah_optimized_dir)
for img_file in img_files:
    print('Processing: {}'.format(img_file))
    keypoints_coco = np.load(os.path.join(base_dir, 'keypoints', img_file.split('/')[-1].split('.')[0] + '.png.npy'), allow_pickle=True)
    joints_target = keypoints_coco[:, :2]
    joints_target[keypoints_coco[:, 2]<0.3] = 0
    joints_target = torch.from_numpy(coco_to_smpl(joints_target[:, :2])).float()
    joints_mask = (joints_target.sum(dim=1) != 0)
    frame_id = int(img_file.split('/')[-1].split('.')[0])
    K = cam_params['K'][frame_id]
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    width = 1265
    height = 711
    renderer = silhouette_renderer_from_pinhole_cam(fx, fy, width, height, cx, cy, device=device)
    D = cam_params['D'][frame_id]
    R = cam_params['R'][frame_id]
    R_np = np.array(R)
    T = cam_params['T'][frame_id]
    T_np = np.array(T).reshape(3, 1) / 1000.0
    T = T_np.tolist()
    frame_index = int(os.path.basename(img_file)[:-4])
    root_orient = Rot.from_rotvec(np.array(smpl_params[frame_index]['Rh']).reshape([-1])).as_matrix()
    trans = np.array(smpl_params[frame_index]['Th']).reshape([3, 1])
    
    betas = np.array(smpl_params[frame_index]['shapes'], dtype=np.float32).reshape(1, -1)
    poses = np.array(smpl_params[frame_index]['poses'], dtype=np.float32).reshape(1, -1)
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

    new_root_orient = Rot.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
    new_trans = trans.reshape([1, 3]).astype(np.float32)


    new_root_orient_torch = torch.from_numpy(new_root_orient)
    new_trans_torch = torch.from_numpy(new_trans)
    if torch.cuda.is_available():
        new_root_orient_torch = new_root_orient_torch.cuda()
        new_trans_torch = new_trans_torch.cuda()

    # Get shape vertices
    body = body_model(betas=betas_torch)
    minimal_shape = body.v.detach().cpu().numpy()[0]

    
    rotation = torch.tensor(new_root_orient_torch.reshape(1, 3), device=poses_torch.device)
    translation = torch.tensor(trans.reshape(1, 3), requires_grad=True, device=poses_torch.device)

    optim_list = [
        {"params": translation, "lr": 5e-3},
    ]
    optim = torch.optim.Adam(optim_list)
    total_iters = 15000
    expected_v = torch.tensor(np.load('/cluster/scratch/xiychen/neuman_data/bike/v2d/' + str(int(frame_id)) + '.npy'), device=poses_torch.device).float()
    for i in tqdm(range(total_iters), total=total_iters):
        body = body_model(root_orient=rotation, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=translation)
        vertices = body.v[0]
        vertices_world = world_to_cam(vertices, torch.tensor(R, device=poses_torch.device), torch.tensor(T, device=poses_torch.device))
        projected_v = vertices_world / vertices_world[:, -1:]
        projected_v = torch.einsum('ij,kj->ki', torch.tensor(K, device=poses_torch.device), projected_v)[:, :2].float()
        
        joints = body.Jtr[0]
        joints_world = world_to_cam(joints, torch.tensor(R, device=poses_torch.device), torch.tensor(T, device=poses_torch.device))
        projected_j = joints_world / joints_world[:, -1:]
        projected_j = torch.einsum('ij,kj->ki', torch.tensor(K, device=poses_torch.device), projected_j)[:, :2].float()
        # world_verts = torch.tensor(body.v, device=device)
        # mesh = Meshes(
        #     verts=world_verts.reshape(1, -1, 3),
        #     faces=faces.reshape(1, -1, 6)[:, :, :3]
        # )
        # silhouette = renderer(meshes_world=mesh, R=torch.tensor(R_np, device=device).reshape(1, 3, 3), T=torch.tensor(T, device=device).reshape(1, 3))
        # silhouette = torch.rot90(silhouette[0, ..., 3], k=2)
        # print(silhouette.shape)
        
        # print(silhouette.sum())
        # exit()
        

        optim.zero_grad()
        # print(projected_j.shape)
        # exit()
        # loss = torch.nn.functional.mse_loss(projected_j[joints_mask], torch.tensor(joints_target, device=poses_torch.device).float()[joints_mask])
        loss_fn = torch.nn.MSELoss()
        # loss = loss_fn(projected_j[joints_mask], joints_target[joints_mask])
        loss = loss_fn(projected_v, expected_v)
        
        # loss += torch.nn.functional.mse_loss(projected_j[joints_mask], torch.tensor(joints_target[joints_mask], device=poses_torch.device).float())
        
        loss.backward()
        optim.step()
    print(loss, translation)

    body = body_model(root_orient=rotation, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=translation)
    bone_transforms = body.bone_transforms.detach().cpu().numpy()
    Jtr_posed = body.Jtr.detach().cpu().numpy()

    vertices = body.v[0]
    vertices_world = world_to_cam(vertices, torch.tensor(R, device=poses_torch.device), torch.tensor(T, device=poses_torch.device))
    projected_v = vertices_world / vertices_world[:, -1:]
    projected_v = torch.einsum('ij,kj->ki', torch.tensor(K, device=poses_torch.device), projected_v)[:, :2].float()
                
    image = cv2.imread(img_file)
    for idx_, loc in enumerate(projected_v):
        x = int(loc[0])
        y = int(loc[1])
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
    Image.fromarray(image[:, :, ::-1]).save(os.path.join(arah_optimized_dir, img_file.split('/')[-1]))
    # exit()
    
    out_filename = os.path.join(smpl_out_dir, '{:06d}.npz'.format(frame_id))

    np.savez(out_filename,
            minimal_shape=minimal_shape,
            betas=betas,
            Jtr_posed=Jtr_posed[0],
            bone_transforms=bone_transforms[0],
            trans=translation.detach().cpu().numpy()[0],
            root_orient=new_root_orient[0],
            pose_body=pose_body[0],
            pose_hand=pose_hand[0])
