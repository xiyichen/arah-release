import numpy as np
import json
import copy


def get_tf_cams(cam_dict, target_radius=3.):
    cam_centers = []
    for cam_name in {x: cam_dict[x] for x in cam_dict.keys() if x not in ['all_cam_names']}:
        R = np.array(cam_dict[cam_name]['R'])
        T = np.array(cam_dict[cam_name]['T']).reshape(3,)
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3, 3] = T
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=3.):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    translate, scale = get_tf_cams(in_cam_dict, target_radius=target_radius)
  
    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return np.linalg.inv(C2W)

    out_cam_dict = copy.deepcopy(in_cam_dict)
    for cam_name in {x: out_cam_dict[x] for x in out_cam_dict.keys() if x not in ['all_cam_names']}:
        R = np.array(out_cam_dict[cam_name]['R'])
        T = np.array(out_cam_dict[cam_name]['T']).reshape(3,)
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3, 3] = T
        W2C = transform_pose(W2C, translate, scale)
        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        out_cam_dict[cam_name]['R'] = W2C[:3, :3].tolist()
        out_cam_dict[cam_name]['T'] = W2C[:3, 3].tolist()


    with open(out_cam_dict_file, 'w') as fp:
        json.dump(out_cam_dict, fp)


if __name__ == '__main__':
    in_cam_dict_file = '/cluster/scratch/xiychen/CoreView_387_preprocessed/CoreView_387/cam_params_unnormalized.json'
    out_cam_dict_file = '/cluster/scratch/xiychen/CoreView_387_preprocessed/CoreView_387/cam_params.json'
    normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=3.)
