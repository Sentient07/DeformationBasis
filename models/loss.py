# loss.py
import torch
import numpy as np

from torch.linalg import det
from utils.my_utils import b3nTobn3, convert_to_batch, eye_like
from utils.batched_mls_function import f_x_interpolate_batched, get_Phi_batched, compute_flow_approx
from pdb import set_trace as strc

def loss_registration(predicted_flow, surface_pts, temp_pts):
    predicted_flow = b3nTobn3(predicted_flow)
    surface_pts = b3nTobn3(surface_pts)
    temp_pts = b3nTobn3(temp_pts)
    gt_flow = surface_pts - temp_pts
    loss_recon = ((predicted_flow - gt_flow)**2).mean()
    return loss_recon


def loss_registration_alternative(predicted_flow, surface_pts, temp_pts):
    predicted_flow = b3nTobn3(predicted_flow)
    surface_pts = b3nTobn3(surface_pts)
    temp_pts = b3nTobn3(temp_pts)
    deformed_template = temp_pts + predicted_flow
    loss_recon = (((surface_pts - deformed_template)**2)).mean()
    return loss_recon


def loss_registration_scaled(predicted_flow, surface_pts, temp_pts):
    """
    NOTE: Does not work
    """
    predicted_flow = b3nTobn3(predicted_flow)
    surface_pts = b3nTobn3(surface_pts)
    surface_pts_scaling = surface_pts**2 + 1e-6
    temp_pts = b3nTobn3(temp_pts)
    deformed_template = temp_pts + predicted_flow
    loss_recon = (((surface_pts - deformed_template)**2) /
                  surface_pts_scaling).mean()
    return loss_recon


def loss_edgelen(deformed_verts, edge_pair, gt_edge_len):
    batch_size = deformed_verts.size(0)
    flow_edge_src = deformed_verts[torch.arange(batch_size).unsqueeze(1).cuda(),
                                   edge_pair[:, :, 0]]
    flow_edge_tar = deformed_verts[torch.arange(batch_size).unsqueeze(1).cuda(),
                                   edge_pair[:, :, 1]]

    pred_edge_len = torch.linalg.norm(flow_edge_tar - flow_edge_src,
                                      dim=2, keepdim=True).squeeze()
    rand_ind = np.random.choice(pred_edge_len.shape[1], 1000, replace=False)
    deformed_edge_len = torch.abs(gt_edge_len.squeeze()[:, rand_ind] - pred_edge_len[:, rand_ind]).mean()
    return deformed_edge_len


def loss_volp(anlyt_g):
    # eye_deform = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(anlyt_g.device)
    eye_deform = eye_like(anlyt_g).to(anlyt_g.device)
    cur_vol = ((torch.linalg.det(eye_deform + anlyt_g) - 1)** 2)
    return cur_vol


def loss_deformation_volume_old(anlyt_g, radius):
    const = 4 / 3 * np.pi * radius**3
    det_mat = torch.abs(torch.linalg.det(anlyt_g))
    det_mat = torch.clip(det_mat, -5e2, 5e2)
    change_vol = det_mat.sum(dim=2).mean()
    return change_vol

def loss_nodal_rigidity(nodal_g, radius):
    const = 4 / 3 * np.pi * radius**3
    # F(x) = x + U(x)
    nodal_g_deform = eye_like(nodal_g) + nodal_g
    anlyt_grad_inner = torch.einsum(
        "bqnmp,bqnpl->bqnml",
        nodal_g_deform.transpose(-2, -1),
        nodal_g_deform.transpose(-2, -1),
    )
    cur_rigidity = const * torch.linalg.norm(
        torch.abs(anlyt_grad_inner - eye_like(anlyt_grad_inner) + 1e-7),
        ord="fro",
        dim=(-1, -2),
        keepdim=True,
    )
    return cur_rigidity.sum(dim=2).mean()


def loss_arap(nodal_g):
    # F(x) = x + U(x)
    nodal_g_deform = eye_like(nodal_g) + nodal_g
#     anlyt_grad_inner = torch.einsum("bpm,bmp->bmp", nodal_g_deform, nodal_g_deform)
    # anlyt_grad_inner = torch.bmm(nodal_g_deform.transpose(2,1), nodal_g_deform)
    anlyt_grad_inner = torch.matmul(nodal_g_deform.transpose(-1,-2), nodal_g_deform)
    cur_rigidity = torch.linalg.norm(torch.abs(anlyt_grad_inner - eye_like(anlyt_grad_inner) + 1e-7),
                                     ord="fro", dim=(-1, -2))
    return cur_rigidity


def compute_jacobian_batched(X, X_is, F_is, radius, return_indv=False):
    inp = X.clone()
    inp.requires_grad_(True)
    out_b = f_x_interpolate_batched(inp, X_is, F_is, radius)
    u = out_b[:,:,0]
    v = out_b[:,:,1]
    w = out_b[:,:,2]
    grad_outputs = torch.ones_like(u)
    grad_u = torch.autograd.grad(u, [inp], grad_outputs=grad_outputs, create_graph=True)[0]
    grad_v = torch.autograd.grad(v, [inp], grad_outputs=grad_outputs, create_graph=True)[0]
    grad_w = torch.autograd.grad(w, [inp], grad_outputs=grad_outputs, create_graph=True)[0]
    if return_indv:
        return grad_u, grad_v, grad_w
    return torch.stack((grad_u, grad_v, grad_w), dim=-1).transpose(3,2)


def loss_nodal_deform(predicted_flow, surface_pts, temp_pts_surface,
                      nodal_pts, mls_radius, return_gt=False):
    sparse_flow_th = surface_pts - temp_pts_surface
    nodal_pts = convert_to_batch(nodal_pts, predicted_flow.shape[0])
    Phi = get_Phi_batched(temp_pts_surface, nodal_pts, mls_radius)
    F_is = torch.linalg.lstsq(Phi, sparse_flow_th)[0]
    loss_nodal_d = ((F_is - predicted_flow)**2).mean()

    if return_gt:
        return loss_nodal_d, F_is

    return loss_nodal_d


def loss_nodal_deform_from_ind(predicted_flow, surface_pts,
                               temp_pts_surface, Phi_full,
                               interest_inds,
                               return_gt=False):
    predicted_flow = b3nTobn3(predicted_flow)
    sparse_flow_th = surface_pts - temp_pts_surface
    Phi_inds = Phi_full[interest_inds, :].float()
    F_is = torch.linalg.lstsq(Phi_inds, sparse_flow_th)[0]
    loss_nodal_d = ((F_is - predicted_flow)**2).mean()

    if return_gt:
        return loss_nodal_d, F_is

    return loss_nodal_d


@torch.no_grad()
def get_laplacian_wobatch(deformed_vert, face_2d):
    from pytorch3d.structures.utils import list_to_padded, padded_to_packed
    from pytorch3d.ops import cot_laplacian
    L, inv_areas = cot_laplacian(deformed_vert, face_2d)
    norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
    idx = norm_w > 0
    norm_w[idx] = 1.0 / norm_w[idx]
    Lx = L.mm(deformed_vert) * norm_w
    return Lx
    
@torch.no_grad()
def get_laplacian(deformed_vert, face_3d):
    from pytorch3d.structures.utils import list_to_padded, padded_to_packed, list_to_packed
    from pytorch3d.ops import cot_laplacian
    n_batch = deformed_vert.shape[0]
    n_verts = deformed_vert.shape[1]
    n_faces = face_3d.shape[0]
    
    verts_packed = padded_to_packed(list_to_padded([i for i in deformed_vert]))
    faces_packed = padded_to_packed(list_to_padded([face_3d.to(deformed_vert.device).to(torch.long) for i in range(n_batch)]))
    faces_offset = (torch.arange(n_batch)*n_verts).to(face_3d.device).long().repeat_interleave(n_faces)
    face_offsetted = faces_packed+faces_offset.unsqueeze(1)
    L, inv_areas = cot_laplacian(verts_packed, face_offsetted)
    
    indices = torch.arange(L.shape[1]).repeat(2,1).view(2,-1).to(L.device)
    lsum = torch.sparse_coo_tensor(indices, torch.sparse.sum(L, dim=1).to_dense().squeeze(), (L.shape[-2], L.shape[-1]))
    L_norm = L - lsum
    
    norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
    idx = norm_w > 0
    norm_w[idx] = 1.0 / norm_w[idx]
    return L_norm, norm_w

def loss_laplacian(deformed_vert, face_3d, temp_L):
    n_batch = deformed_vert.shape[0]
    curve_gt = torch.norm(temp_L.view(-1, 3), p=2, dim=1).float()
    L_norm = get_laplacian(deformed_vert, face_3d)[0]
    Lx = L_norm.mm(deformed_vert.view(-1, 3))
    repeated_gt = curve_gt.repeat(n_batch)
    loss = ((torch.norm(Lx, p=2, dim=1).float() - repeated_gt)**2).mean()
    return loss