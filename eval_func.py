# eval_func.py
import torch
import numpy as np
import os
import os.path as osp
import json
import trimesh
from tqdm import tqdm
from itertools import combinations, permutations
from sklearn.neighbors import NearestNeighbors

from MatchEval.Meshes import get_dist, get_all_p2p_vts
from pathlib import Path
from utils.my_utils import safe_make_dirs, flatten_list_of_list
from utils import local_config
from pdb import set_trace as strc


def custom_correspondence_fast(source, source_reconstructed,
                               target, target_reconstructed):

    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neigh.fit(source_reconstructed.vertices)
    nn_ind = neigh.kneighbors(source.vertices, return_distance=False)
    closest_points = target_reconstructed.vertices[nn_ind]
    closest_points = np.mean(closest_points, 1, keepdims=False)
    neigh.fit(target.vertices)
    idx_knn = neigh.kneighbors(closest_points, return_distance=False)
    return np.arange(len(target.vertices))[np.squeeze(idx_knn)]


def run_matching(mesh_dir, test_combo, all_recon_mesh, ext='.ply', save_dir=None):
    unique_names = list(set(flatten_list_of_list(test_combo)))
    unique_meshes = [osp.join(mesh_dir, i + ext) for i in unique_names]
    all_src_mesh = {Path(i).stem: trimesh.load(i, process=False)
                    for i in unique_meshes}
    p2p_json = {}
    for _, (s, t) in enumerate(tqdm(test_combo)):
        source_m = all_src_mesh[s]
        target_m = all_src_mesh[t]
        source_recon = all_recon_mesh[s]
        tar_recon = all_recon_mesh[t]

        cp = custom_correspondence_fast(source_m, source_recon, target_m, tar_recon)
        p2p_json[s+"_"+t] = cp.tolist()

        if bool(save_dir):
            file_name = str(s) + "_" + str(t) + '.txt'
            with open(osp.join(save_dir, file_name), 'w') as fo:
                for item in cp:
                    fo.write("%s\n" % (item+1))

    return p2p_json

def match_shapenet(mesh_dir, dataset_test, exp_id, all_recon_mesh):
    m_names = [Path(i).stem for i in os.listdir(mesh_dir)]
    test_combo = [i for i in combinations(m_names, 2)]
    p2p_dict = run_matching(mesh_dir, test_combo, all_recon_mesh, ext='.xyz')
    save_dir = './Results/MLS/%s'%dataset_test
    safe_make_dirs(save_dir)
    p2p_json_name = osp.join(save_dir, '%s.json' % exp_id)
    json.dump(p2p_dict, open(p2p_json_name, 'w'))
    print("Result for %s is : \n" % exp_id)
    return p2p_dict

def match_panoptic(mesh_dir, dataset_test, exp_id, all_recon_mesh):
    m_names = [Path(i).stem for i in os.listdir(mesh_dir) if 'Reconstruction' not in i]
    test_combo = [i for i in permutations(m_names, 2)]
    p2p_dict = run_matching(mesh_dir, test_combo, all_recon_mesh, ext='.ply')
    save_dir = './Results/MLS/%s'%dataset_test
    safe_make_dirs(save_dir)
    p2p_json_name = osp.join(save_dir, '%s.json' % exp_id)
    json.dump(p2p_dict, open(p2p_json_name, 'w'))
    print("Result for %s is : \n" % exp_id)
    return p2p_dict

def match_shrec_19_o(mesh_dir, method_name, eval_pairs, all_recon_mesh, dataset_test,
                     ext='.off'):
    test_combo_full = [i.rstrip().split(',')
                       for i in open(eval_pairs).readlines()]
    save_dir = './Results/MLS/%s/%s/'%(dataset_test, method_name)
    safe_make_dirs(save_dir)
    run_matching(mesh_dir, test_combo_full, all_recon_mesh,
                 ext=ext, save_dir=save_dir)

def match_shrec_20(mesh_dir, method_name, all_recon_mesh):
    test_shapes = ['scan08', 'scan09', 'scan10', 'scan11']
    eval_pairs = [(s,'scan00') for s in test_shapes]
    p2p_json = run_matching(mesh_dir, eval_pairs, all_recon_mesh, ext='.ply')
    save_dir = './Results/MLS/SHREC20'
    safe_make_dirs(save_dir)
    p2p_json_name = osp.join(save_dir, '%s.json' % method_name)
    json.dump(p2p_json, open(p2p_json_name, 'w'))
    return p2p_json

def match_scape_r(mesh_dir, method_name, all_recon_mesh,
                  dataset_test, permutation=False):
    m_name = 'mesh%03d'
    if permutation:
        test_combo = [(m_name % i[0], m_name % i[1])
                      for i in permutations(list(range(52, 72)), 2)]
    else:
        test_combo = [(m_name % i[0], m_name % i[1])
                      for i in combinations(list(range(52, 72)), 2)]
    p2p_dict = run_matching(mesh_dir, test_combo, all_recon_mesh)
    save_dir = './Results/MLS/%s'%dataset_test
    safe_make_dirs(save_dir)
    p2p_json_name = osp.join(save_dir, '%s.json' % method_name)
    json.dump(p2p_dict, open(p2p_json_name, 'w'))
    print("Result for %s is : \n" % method_name)
    return p2p_dict


def run_eval_vts(pred_map, geod_mat_dir, mesh_dir, vts_dir, test_pairs,
                 mesh_ext='.ply'):
    vts_dict = get_all_p2p_vts(vts_dir, prefix='', test_pairs=test_pairs)
    geod = get_dist(test_pairs, vts_dict, pred_map, geod_mat_dir, mesh_dir=mesh_dir,
                    normalise=True, strict=True,
                    is_matlab=False, mesh_ext=mesh_ext)
    return geod


def eval_scape_r(pred_map, mesh_dir):
    geod_dir = local_config.SCAPE_RM_GEOD_DIR
    test_pairs = [i.split('_') for i in pred_map.keys()]
    geod = run_eval_vts(pred_map, geod_dir,
                        mesh_dir, local_config.SCAPE_RM_VTS,
                        test_pairs, mesh_ext='.ply')
    return geod

def run_evaluation(dataset_test, exp_id, all_recon_mesh,
                   permutation=False):
    sh19_pairs = local_config.SHREC19_EVAL_PAIRS
    if dataset_test == "scape_r_n05":
        pred_map = match_scape_r(local_config.SCAPE_RM_N05, exp_id,
                                 all_recon_mesh, 'SCAPE_R_N05',
                                 permutation=permutation)
        geod_ar = eval_scape_r(pred_map, mesh_dir=local_config.SCAPE_RM)
        test_loss_dict = {"scape_r_n05_err": np.mean(geod_ar)}

    elif dataset_test == 'shrec20_a_test':
        match_shrec_20(local_config.SHREC20a_PLY, exp_id, all_recon_mesh)
        test_loss_dict = {"shrec20_a_err": 0.}

    elif dataset_test == 'shrec19_o':
        mesh_dir = local_config.SHREC19_O
        ext=local_config.SHREC19_EXT
        match_shrec_19_o(mesh_dir, exp_id, sh19_pairs, all_recon_mesh,
                         'SHREC19_O', ext=ext)
        test_loss_dict = {"shrec19_o_err": 0.}
        
    elif dataset_test == 'chairs':
        match_shapenet(local_config.CHAIRS_VAL, dataset_test, exp_id, all_recon_mesh)
        test_loss_dict = {"match_chairs": 0.}
        
    elif dataset_test == 'planes':
        match_shapenet(local_config.PLANE_VAL, dataset_test, exp_id, all_recon_mesh)
        test_loss_dict = {"match_planes": 0.}
        
    elif dataset_test == 'tables':
        match_shapenet(local_config.TABLES_VAL, dataset_test, exp_id, all_recon_mesh)
        test_loss_dict = {"match_tables": 0.}
        
    elif dataset_test == 'panoptic':
        match_panoptic(local_config.PANOPTIC_PCS, dataset_test, exp_id, all_recon_mesh)
        test_loss_dict = {"match_panoptic": 0.}
    else:
        raise ValueError("Unknown dataset_test: {}".format(dataset_test))
        
    return test_loss_dict
