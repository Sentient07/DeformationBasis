from random import shuffle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import torch
import json
import numpy as np
import trimesh
from glob import glob
from pathlib import Path
import os.path as osp
from utils.my_utils import *

from utils import local_config
from utils.data_aug_utils import *


def unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)


class Base_NodalSample(Dataset):
    def __init__(
        self,
        dir_path,
        template_path,
        is_train,
        data_augment=False,
        range_rot=180,
        n_surf_points=3000,
        gt_maps=None,
        use_bary_pts: bool = False,
        wo_preprocess: bool =False
    ):

        self.dir_path = dir_path
        self.use_bary_pts = use_bary_pts
        self.is_train = is_train
        self.n_surf_points = n_surf_points
        self.wo_preprocess = wo_preprocess

        self.gt_maps = gt_maps
        self.data = self._get_data()
        self.pair_len = len(self.data)
        self.data_augment = data_augment
        self.range_rot = range_rot

        self.template_data = torch.load(template_path)
        self.nodes_th = self.template_data["nodes"]
        self.temp_verts_th = self.template_data["verts"]
        self.temp_verts_np = self.temp_verts_th.numpy()
        self.temp_faces_np = self.template_data["faces"].numpy()
        self.template_mesh = trimesh_from_vf(
            self.temp_verts_np, self.temp_faces_np)
        self.temp_all_edges = self.temp_verts_np[self.template_mesh.edges]
        self.temp_all_edge_len = np.linalg.norm(
            self.temp_all_edges[:, 0, :] - self.temp_all_edges[:, 1, :],
            axis=1,
            keepdims=True,
        )
        self.prop = get_vertex_normalised_area(self.template_mesh)


    def _get_data(self):
        raise NotImplementedError()

    def __len__(self):
        return self.pair_len

    def subsample_baryc(self, ):
        cur_vert = numpied(cur_vert)
        shape_mesh = trimesh_from_vf(cur_vert, self.temp_faces_np)
        template_sur_samples, face_ids = trimesh.sample.sample_surface(
            self.template_mesh, self.n_surf_points
        )
        temp_triangles = self.template_mesh.vertices[
            self.template_mesh.faces[face_ids]
        ]
        temp_sur_samp_bary = trimesh.triangles.points_to_barycentric(
            temp_triangles, template_sur_samples
        )

        shape_triangles = np.array(
            shape_mesh.vertices[shape_mesh.faces[face_ids]])
        shape_sur_samples = trimesh.triangles.barycentric_to_points(
            shape_triangles, temp_sur_samp_bary
        )
        # samp_pts_torch = torched(shape_sur_samples, device='cpu')
        surf_subsamp_ind_src = face_ids
        return surf_subsamp_ind_src

    def subsample_diffconnec(self, idx):
        mesh_name = Path(self.data[idx]).stem
        cur_p2p = np.array((self.gt_maps[mesh_name]))
        corr_ind_temp = np.squeeze(cur_p2p)[:, 0]
        corr_ind_src = np.squeeze(cur_p2p)[:, 1]
        rand_samp_ind = np.random.choice(corr_ind_src.shape[0],
                                         size=self.n_surf_points,
                                         replace=False)
        # Take random same random samples from corresp
        subsamp_ind_src = corr_ind_src[rand_samp_ind]
        subsamp_ind_temp = corr_ind_temp[rand_samp_ind]
        return subsamp_ind_src, subsamp_ind_temp

    def subsample_mesh(self, cur_vert, idx):
        if self.use_bary_pts and self.is_train:
            assert self.gt_maps is None
            subsamp_ind_src = self.subsample_baryc()
            subsamp_ind_temp = subsamp_ind_src

        if self.gt_maps is not None:
            assert not bool(self.use_bary_pts), "Can't use bary supervision"
            subsamp_ind_src, subsamp_ind_temp = self.subsample_diffconnec(idx)
        else:
            proba = self.prop
            subsamp_ind_src = np.random.choice(cur_vert.shape[0],
                                               size=self.n_surf_points,
                                               p=proba)
            subsamp_ind_temp = subsamp_ind_src
            # samp_pts_torch = cur_vert[surf_subsamp_ind_src]

        return subsamp_ind_src, subsamp_ind_temp

    def _get_surface_points_ind(self, inp1_mesh, idx):
        # sur_pts_np = self.data[index1]
        sur_pts_np = np.array(inp1_mesh.vertices)
        if not self.wo_preprocess:
            sur_pts_np = np.array(all_preprocess(inp1_mesh, self.template_mesh,
                                                 is_train=self.is_train).vertices)
        sur_pts_th = torch.from_numpy(sur_pts_np).float()
        if self.data_augment:
            sur_pts_th, _ = uniform_rotation_axis(sur_pts_th.float(), axis=1, normals=None,
                                                  range_rot=self.range_rot)

        sur_pts_th, _, _ = center_bounding_box(sur_pts_th.float())

        if self.is_train:
            sur_pts_th = add_random_translation(
                sur_pts_th.double(), scale=0.03).float()

        if self.n_surf_points > 0:
            subsamp_ind_src, subsamp_ind_temp = self.subsample_mesh(sur_pts_th, idx)
        else:
            subsamp_ind_src = np.arange(len(sur_pts_th))
            subsamp_ind_temp = subsamp_ind_src

        subsamp_ind_src_th = torched(subsamp_ind_src, device="cpu", dtype=torch.int64)
        subsamp_ind_temp_th = torched(subsamp_ind_temp, device="cpu", dtype=torch.int64)

        return sur_pts_th, subsamp_ind_src_th, subsamp_ind_temp_th

    def __getitem__(self, idx):  # permuted for input1

        # Get surface and far points
        cur_npz = np.load(self.data[idx])
        sur_pts_np = np.reshape(np.array(cur_npz["verts"]), (-1, 3))
        cur_faces = self.template_mesh.faces
        cur_mesh = trimesh_from_vf(sur_pts_np, cur_faces)
        sur_pts_th, subsamp_ind_src_th, subsamp_ind_temp_th = self._get_surface_points_ind(
            cur_mesh, idx)

        gt_map = 0

        return_items = (sur_pts_th, subsamp_ind_src_th, subsamp_ind_temp_th, sur_pts_th.shape[0],
                        gt_map)
        return return_items


class SURREAL1K_NodalSample(Base_NodalSample):
    def __init__(
        self,
        dir_path,
        template_path,
        is_train,
        data_augment=False,
        range_rot=180,
        n_surf_points=3000,
        gt_maps=None,
        use_bary_pts=False,
    ):
        super().__init__(
            dir_path,
            template_path,
            is_train,
            data_augment=data_augment,
            range_rot=range_rot,
            n_surf_points=n_surf_points,
            gt_maps=gt_maps,
            use_bary_pts=use_bary_pts,
        )

    def _get_data(self):
        self.all_npzs_sorted = sorted(glob(osp.join(self.dir_path, "*.npz")))
        shuffle(self.all_npzs_sorted)
        self.all_npzs_sorted = self.all_npzs_sorted
        return self.all_npzs_sorted

    
class SHREC20a_Base(Base_NodalSample):
    def __init__(self, dir_path, template_path, is_train,
                 d_type='pose_t', data_augment=False, range_rot=360,
                 n_surf_points=2500,
                 use_bary_pts=False):

        gt_map_dir = local_config.SHREC20a_GTMAP_LRES
        self.is_train = is_train
        assert d_type in ('pose_t', 'shape_t')
        self.d_type = d_type
        processed_gt_map = self.get_templ_map_from_json(gt_map_dir)
        self.template_mesh = trimesh.load(osp.join(local_config.SHREC20a_PLY, 'scan00.ply'),
                                          process=False)
        super().__init__(dir_path, template_path, is_train,
                         data_augment=False, n_surf_points=n_surf_points,
                         gt_maps=processed_gt_map, use_bary_pts=False,
                         wo_preprocess=True)

    def get_templ_map_from_json(self, gt_map_dir):
        templ = 'scan%02d_' % 0
        self.gt_map = json.load(open(gt_map_dir))
        refined_gt_maps = {k[len(templ):]:np.c_[v, np.arange(len(v))] for k,v in self.gt_map.items()}
        return refined_gt_maps

    def _get_data(self):
        train_shapes = ['scan01', 'scan02', 'scan03', 'scan04', 'scan06', 'scan07']
        test_shapes = ['scan08', 'scan09', 'scan10', 'scan11']
        if self.is_train:
            data = [osp.join(self.dir_path, i+".npz") for i in train_shapes]
        else:
            data = [osp.join(self.dir_path, i+".npz") for i in test_shapes]

        return data

    def __getitem__(self, idx):  # permuted for input1

        # Get surface and far points
        cur_file = self.data[idx]
        cur_npz = np.load(cur_file)
        sur_pts_np = np.reshape(np.array(cur_npz["verts"]), (-1, 3))
        cur_faces = np.array(cur_npz['faces'])
        cur_mesh = trimesh_from_vf(sur_pts_np, cur_faces)
        # sur_pts_th, subsamp_ind_src_th, subsamp_ind_temp_th = self._get_surface_points_ind(cur_mesh,
        #                                                                                    idx)
        sur_pts_th = torched(sur_pts_np, device='cpu')
        subsamp_ind_src_th = torch.arange(sur_pts_np.shape[0])
        subsamp_ind_temp_th = torch.arange(self.template_mesh.vertices.shape[0])
        gt_map = torched(np.array(self.gt_map['scan00_%s'%(Path(cur_file).stem)]), device='cpu', dtype=torch.long)
        return_items = (sur_pts_th, subsamp_ind_src_th, subsamp_ind_temp_th,
                        sur_pts_th.shape[0],
                        gt_map)
        return return_items


class ShapeNetTrain(Base_NodalSample):
    def __init__(self, dir_path, template_path, is_train,
                 d_type='pose_t', data_augment=False, range_rot=360,
                 n_surf_points=2500,
                 use_bary_pts=False):
        self.is_train = is_train
        self.all_shapes = self.get_shapes(dir_path)
        self.template_path = template_path
        self.len = len(self.all_shapes)
        self.n_surf_points = n_surf_points
        self.temp_samp_ind_th = torch.arange(torch.load(self.template_path)['verts'].cpu().shape[0]).long()
        print("loaded ShapeNet train : %d shapes" % self.len)

    def get_shapes(self, mesh_dir):
        all_shapes = []
        gt_pts = sorted(glob(osp.join(mesh_dir, '*.xyz')))
        for gt_pt in gt_pts:
            gt_pc = trimesh.load(gt_pt, process=False)
            all_shapes.append(gt_pc)
        return all_shapes
    
    def subsample_pts(self, cur_v):
        sample_ind = np.random.choice(cur_v.shape[0],self.n_surf_points)
        sampled_v = cur_v[sample_ind]
        return sampled_v, sample_ind

    def __getitem__(self, index):
        cur_v = self.all_shapes[index]
        cur_v_samp, sample_ind  = self.subsample_pts(cur_v)
        cur_v_samp_th = torch.from_numpy(cur_v_samp).float()
        sample_ind_th = torch.from_numpy(sample_ind).long()
        return_items = (cur_v_samp_th, sample_ind_th, self.temp_samp_ind_th,
                        cur_v_samp_th.shape[0], 0)
        return return_items

    def __len__(self):
        return self.len

class Shrec20_a_test(Dataset):
    def __init__(self, mesh_dir):
        self.all_shapes = self.get_shapes(mesh_dir)
        self.len = len(self.all_shapes)
        print("loaded SHREC20 : %d shapes" % self.len)

    def get_shapes(self, mesh_dir):
        test_shapes = ['scan00', 'scan08', 'scan09', 'scan10', 'scan11']
        all_shapes = sorted([osp.join(mesh_dir, "%s.ply" % s) for s in test_shapes])
        return all_shapes

    def __getitem__(self, index):
        return self.all_shapes[index]

    def __len__(self):
        return self.len
    
    
class Scape_r_test(Dataset):
    def __init__(self, mesh_dir):
        self.all_shapes = self.get_shapes(mesh_dir)
        self.len = len(self.all_shapes)
        print("loaded SCAPE : %d shapes" % self.len)

    def get_shapes(self, mesh_dir):
        return sorted([osp.join(mesh_dir, "mesh%03d.ply" % i) for i in range(52, 72)])

    def __getitem__(self, index):
        return self.all_shapes[index]

    def __len__(self):
        return self.len
    

class PanopticTest(Dataset):
    def __init__(self, mesh_dir):
        self.all_shapes = self.get_shapes(mesh_dir)
        self.len = len(self.all_shapes)
        print("loaded Panoptic : %d shapes" % self.len)

    def get_shapes(self, mesh_dir):
        return [i for i in sorted(glob(osp.join(local_config.PANOPTIC_PCS, '*.ply'))) if 'Reconstruction' not in Path(i).stem]

    def __getitem__(self, index):
        return self.all_shapes[index]

    def __len__(self):
        return self.len
    

class Shrec19_r_test(Dataset):
    def __init__(self, mesh_dir):
        self.all_shapes = self.get_shapes(mesh_dir)
        self.len = len(self.all_shapes)
        print("loaded SHREC19 : %d shapes" % self.len)

    def get_shapes(self, mesh_dir):
        return sorted(
            [
                osp.join(mesh_dir, "%d%s" % (i, local_config.SHREC19_EXT))
                for i in range(1, 45)
            ],
            key=lambda y: int(Path(y).stem),
            reverse=True
        )

    def __getitem__(self, index):
        return self.all_shapes[index]

    def __len__(self):
        return self.len


class ShapeNetTest(Dataset):
    def __init__(self, mesh_dir):
        self.dir_path = mesh_dir
        self.all_shapes = self.get_shapes()
        self.len = len(self.all_shapes)
        print("loaded ShapeNet Test : %d shapes" % self.len)

    def __getitem__(self, index):
        return self.all_shapes[index]
    
    def get_shapes(self):
        all_shapes = []
        mesh_dir = self.dir_path
        gt_pts = sorted(glob(osp.join(mesh_dir, '*.xyz')))
        for gt_pt in gt_pts:
            all_shapes.append(gt_pt)
        return all_shapes
    

    def __len__(self):
        return self.len

def dataset_name_wrapper(dataset_name, is_train=True):

    dataset_class = SURREAL1K_NodalSample
    
    if dataset_name.lower() in ('shrec20_a'):
        dataset_class = SHREC20a_Base
        dataset_dir = local_config.SHREC20a_NPZ

    elif dataset_name.lower() in ("surreal1k"):
        dataset_dir = local_config.SURREAL1K_TR

    elif dataset_name.lower() in ("surreal_val"):
        dataset_dir = local_config.SURREAL1K_VAL

    elif dataset_name.lower() in ("chairs"):
        dataset_class = ShapeNetTrain
        dataset_dir = local_config.CHAIRS_FULL if is_train else local_config.CHAIRS_VAL
    elif dataset_name.lower() in ("planes"):
        dataset_class = ShapeNetTrain
        dataset_dir = local_config.PLANE_FULL if is_train else local_config.PLANE_VAL
    elif dataset_name.lower() in ("tables"):
        dataset_class = ShapeNetTrain
        dataset_dir = local_config.TABLES_TRAIN if is_train else local_config.TABLES_VAL
    else:
        raise ValueError("Unknown dataset")

    return dataset_class, dataset_dir


def test_dataset_wrapper(dataset_name):
    shrec19_o_dir = local_config.SHREC19_O

    if dataset_name.lower() in ("scape_r_n05"):
        dataset_class = Scape_r_test
        dataset_dir = local_config.SCAPE_RM_N05
        
    elif dataset_name.lower() in ("shrec19_o"):
        dataset_class = Shrec19_r_test
        dataset_dir = shrec19_o_dir
        
    elif dataset_name.lower() in ("panoptic"):
        dataset_class = PanopticTest
        dataset_dir = local_config.PANOPTIC_PCS
    
    elif dataset_name.lower() in ('shrec20_a_test'):
        dataset_class = Shrec20_a_test
        dataset_dir = local_config.SHREC20a_PLY

    elif dataset_name.lower() in ('chairs'):
        dataset_class = ShapeNetTest
        dataset_dir = local_config.CHAIRS_VAL
        
    elif dataset_name.lower() in ('planes'):
        dataset_class = ShapeNetTest
        dataset_dir = local_config.PLANE_VAL

    elif dataset_name.lower() in ('tables'):
        dataset_class = ShapeNetTest
        dataset_dir = local_config.TABLES_VAL
        
    elif dataset_name.lower() in ("panoptic"):
        dataset_class = PanopticTest
        dataset_dir = local_config.PANOPTIC_PCS
        
    return dataset_class, dataset_dir
    


def my_collate_fn(batch):
    new_batch = [i[3:] for i in batch]
    padded_vert_ind = []
    for j in range(3):
        padded_vert_ind.append(pad_sequence([i[j] for i in batch],
                                            batch_first=True))

    padded_vert_ind.extend(
        torch.utils.data.dataloader.default_collate(new_batch))

    return padded_vert_ind
