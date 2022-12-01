### This code has been borrowed from Thibault Groueix.
import numpy as np
import torch

def get_vertex_normalised_area(mesh):
    """Cross product normalised area of each face
    """
    num_vertices = mesh.vertices.shape[0]
    print("num_vertices", num_vertices)
    a = mesh.vertices[mesh.faces[:, 0]]
    b = mesh.vertices[mesh.faces[:, 1]]
    c = mesh.vertices[mesh.faces[:, 2]]
    cross = np.cross(a - b, a - c)
    area = np.sqrt(np.sum(cross ** 2, axis=1))
    prop = np.zeros((num_vertices))
    prop[mesh.faces[:, 0]] = prop[mesh.faces[:, 0]] + area
    prop[mesh.faces[:, 1]] = prop[mesh.faces[:, 1]] + area
    prop[mesh.faces[:, 2]] = prop[mesh.faces[:, 2]] + area
    return prop / np.sum(prop)

def get_3D_rot_matrix(axis, angle):
    if axis == 0:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    if axis == 1:
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [- np.sin(angle), 0,  np.cos(angle)]])
    
def uniform_rotation_axis_matrix(axis=0, range_rot=360):
    scale_factor = 360.0 / range_rot
    theta = np.random.uniform(- np.pi/scale_factor, np.pi/scale_factor)
    rot_matrix = get_3D_rot_matrix(axis, theta)
    return torch.from_numpy(np.transpose(rot_matrix)).float()


def uniform_rotation_axis(points, axis=0, normals=False, range_rot=360):
    rot_matrix = uniform_rotation_axis_matrix(axis, range_rot)

    if isinstance(points, torch.Tensor):
        points[:, :3] = torch.mm(points[:, :3], rot_matrix)
        if normals:
            points[:, 3:6] = torch.mm(points[:, 3:6], rot_matrix)
        return points, rot_matrix
    elif isinstance(points, np.ndarray):
        points = points.copy()
        points[:, :3] = points[:, :3].dot(rot_matrix.numpy())
        if normals:
            points[:, 3:6] = points[:, 3:6].dot(rot_matrix.numpy())
        return points, rot_matrix
    else:
        print("Pierre-Alain was right.")

def center_bounding_box(points):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # Center bounding box of first 3 dimensions
    if isinstance(points, torch.Tensor):
        points = points.squeeze()
        transpo = False
        if points.size(0) == 3:
            transpo = True
            points = points.transpose(1, 0).contiguous()
        min_vals = torch.min(points, 0)[0]
        max_vals = torch.max(points, 0)[0]
        points = points - (min_vals + max_vals) / 2
        if transpo:
            points = points.transpose(1, 0).contiguous()
        return points, (min_vals + max_vals) / 2, (max_vals - min_vals)/2
    elif isinstance(points, np.ndarray):
        min_vals = np.min(points, 0)
        max_vals = np.max(points, 0)
        points = points - (min_vals + max_vals) / 2
        return points, (min_vals + max_vals) / 2, (max_vals - min_vals)/2
    else:
        print(type(points))
        print("Pierre-Alain was right.")

def add_random_translation(points, scale=0.03):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # Uniform random translation on first 3 dimensions
    a = torch.DoubleTensor(3)
    points[:, 0:3] = points[:, 0:3] + \
        (a.uniform_(-1, 1) * scale).unsqueeze(0).expand(-1, 3)
    return points
