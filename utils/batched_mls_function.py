
import torch
import numpy as np

from pdb import set_trace as strc
from torch.linalg import *

from utils.my_utils import eye_like, b3nTobn3, convert_to_batch


def get_weights_batched(X, X_is, radius):
    """
    X_is : (B, N, 3)
    X    : (B, M, 3)

    returns : (B, M, N)
    """
    assert X.size(0) == X_is.size(0)
    dist = ((X.unsqueeze(2) - X_is.unsqueeze(1)) ** 2).sum(dim=-1)
    dist_mat = torch.maximum(torch.zeros_like(dist).to(dist.device), (1 - (dist / (radius**2))) ** 3)
    return dist_mat


def get_moment_mat_batched(X_is, dist_vec):
    """
    X_is : (B, N, 3)
    dist_vec : (B, M, N)
    Returns : (B, M, 4, 4)
    """
    ones_coord = torch.ones((X_is.shape[0], X_is.shape[1], 1)).to(X_is.dtype).to(X_is.device)
    p_xis = torch.cat((ones_coord, X_is), dim=-1)
    P_XIs = torch.einsum('bijk,bikl->bijl', p_xis.unsqueeze(-1), p_xis.unsqueeze(2))
    M_X = torch.einsum('bij,bjkl->bijkl', dist_vec, P_XIs)
    return M_X.sum(dim=2)


def get_Phi_batched(X, X_is, radius):
    dist_vec = get_weights_batched(X, X_is, radius=radius)  # (200,)
    M = get_moment_mat_batched(X_is, dist_vec)  # (4,4)
    p_x = torch.cat((torch.ones((X.shape[0], X.shape[1], 1)).to(X.device).to(X.dtype), X), dim=-1)
    P_Xis = torch.cat((torch.ones((X_is.shape[0], X_is.shape[1], 1)).to(X_is.device).to(X_is.dtype),
                       X_is, ),dim=-1,)

    M_inv = torch.linalg.solve(M, eye_like(M))

    first_term = torch.einsum('bij,bijk->bik', p_x, M_inv)
    second_term = torch.einsum('bpn,bmn->bmpn', P_Xis.transpose(2, 1), dist_vec)
    Phi = torch.einsum('bij,bijk->bik', first_term, second_term)
    return Phi


def f_x_interpolate_batched(X, X_is, F_is, radius):
    Phi = get_Phi_batched(X, X_is, radius)
    F_x = torch.bmm(Phi, F_is)
    return F_x


def compute_flow_approx(predicted_flow, nodal_points, interest_points, mls_radius):
    # Interpolate for surface pts
    n_batch = predicted_flow.size(0)
    F_is = b3nTobn3(predicted_flow)
    X = b3nTobn3(interest_points)
    X_is = b3nTobn3(convert_to_batch(nodal_points, n_batch))
    target_flow_approx = f_x_interpolate_batched(X, X_is, F_is,
                                                 radius=mls_radius)
    return target_flow_approx


def compute_flow_approx_from_ind(predicted_flow, interest_inds, Phi_full):
    # Interpolate for surface pts
    F_is_src = b3nTobn3(predicted_flow)
    Phi_inds = Phi_full[interest_inds, :].float()
    target_flow_approx = torch.bmm(Phi_inds, F_is_src)
    return target_flow_approx


def grad_weights_batched(X, X_is, radius):

    assert X.size(0) == X_is.size(0)

    X_minus_Xi = X.unsqueeze(2) - X_is.unsqueeze(1)
    X_minus_Xi_square = X_minus_Xi**2

    grad_residue = -6 / (radius**6)

    term_1 = (
        grad_residue
        * X_minus_Xi[:, :, :, 0]
        * (
            (
                radius**2
                - X_minus_Xi[:, :, :, 2] ** 2
                - X_minus_Xi[:, :, :, 0] ** 2
                - X_minus_Xi[:, :, :, 1] ** 2
            )
            ** 2
        )
    )
    term_2 = (
        grad_residue
        * X_minus_Xi[:, :, :, 1]
        * (
            (
                radius**2
                - X_minus_Xi[:, :, :, 2] ** 2
                - X_minus_Xi[:, :, :, 0] ** 2
                - X_minus_Xi[:, :, :, 1] ** 2
            )
            ** 2
        )
    )
    term_3 = (
        grad_residue
        * X_minus_Xi[:, :, :, 2]
        * (
            (
                radius**2
                - X_minus_Xi[:, :, :, 2] ** 2
                - X_minus_Xi[:, :, :, 0] ** 2
                - X_minus_Xi[:, :, :, 1] ** 2
            )
            ** 2
        )
    )

    grad_vector = torch.stack((term_1, term_2, term_3), dim=-1)
    out_tensor = (
        torch.zeros_like(grad_vector).to(
            grad_vector.device).to(grad_vector.dtype)
    )
    anlyt_grad = torch.where(
        X_minus_Xi_square.sum(dim=3).unsqueeze(
            3) < radius**2, grad_vector, out_tensor
    )
    return anlyt_grad


def grad_moment_mat_batched(X_is, grad_weights, M_inv):
    p_xis = torch.cat(
        (
            torch.ones((X_is.shape[0], X_is.shape[1], 1))
            .to(X_is.dtype)
            .to(X_is.device),
            X_is,
        ),
        dim=-1,
    )
    P_XIs = torch.einsum("bijk,bikl->bijl", p_xis.unsqueeze(-1), p_xis.unsqueeze(2))
    dM_xyz = []
    for i in range(3):
        dM_i = torch.einsum("bijk,bni->bnijk", P_XIs, grad_weights[:, :, :, i]).sum(
            dim=2
        )
        M_inv_dM_i = torch.einsum(
            "bnjk,bnkl->bnjl", -M_inv, dM_i.transpose(2, 3))
        M_inv_dM_i_M_inv = torch.einsum(
            "bnjk,bnkl->bnjl", M_inv_dM_i, M_inv.transpose(2, 3)
        )
        dM_xyz.append(M_inv_dM_i_M_inv)
    dM_net = torch.stack(tuple(dM_xyz), dim=-1)
    return dM_net


def grad_each_f_x_forward_batched(p_x, M_inv, P_Xis, dist_vec, F_is):
    first_term = torch.einsum("bij,bijk->bik", p_x, M_inv)
    second_term = torch.einsum(
        "bpn,bmn->bmpn", P_Xis.transpose(2, 1).contiguous(), dist_vec
    )
    Phi = torch.einsum("bij,bijk->bik", first_term, second_term)
    F_x = torch.bmm(Phi, F_is)
    return F_x


def grad_f_x_forward_batched(p_x, M_inv, P_Xis, dist_vec):
    first_term = torch.einsum("bij,bijk->bik", p_x, M_inv)
    second_term = torch.einsum("bpn,bmn->bmpn", P_Xis.transpose(2, 1).contiguous(), dist_vec)
    D_Phi_j = torch.einsum("bij,bijk->bik", first_term, second_term)
    return D_Phi_j


def compute_first_term_batched(grad_p_x, M_inv, P_Xis, dist_vec):
    first_term = torch.einsum('bijp,bijk->bpik', grad_p_x, M_inv)
    second_term = torch.einsum("bpn,bmn->bmpn", P_Xis.transpose(2, 1), dist_vec)
    D_Phi_j = torch.einsum('bpij,bijk->bikp', first_term, second_term)
    return D_Phi_j


def compute_second_term_batched(p_x, grad_M_inv, P_Xis, dist_vec):
    first_term = torch.einsum('bij,bijkp->bikp', p_x, grad_M_inv)
    second_term = torch.einsum("bpn,bmn->bmpn", P_Xis.transpose(2, 1), dist_vec)
    D_Phi_j = torch.einsum('bxjd,bxjn->bxnd', first_term, second_term)
    return D_Phi_j


def compute_third_term_batched(p_x, M_inv, P_Xis, grad_weights):
    first_term = torch.einsum('bij,bijk->bik', p_x, M_inv)
    second_term = torch.einsum('bjn,bxnd->bxnjd', P_Xis.transpose(2, 1), grad_weights)
    D_Phi_j = torch.einsum('bxj,bxnjd->bxnd', first_term, second_term)
    return D_Phi_j


def grad_compute_residues(X, X_is, radius):
    # get dist vec
    dist_vec = get_weights_batched(X, X_is, radius)
    # Get Moment from dist vec and nodes
    M = get_moment_mat_batched(X_is, dist_vec)
    # Create basis column vector
    p_x = torch.cat(
        (torch.ones((X.shape[0], X.shape[1], 1)).to(X.device).to(X.dtype), X), dim=-1
    )
    # Constant Term -> Nodal points
    P_Xis = torch.cat(
        (
            torch.ones((X_is.shape[0], X_is.shape[1], 1))
            .to(X_is.device)
            .to(X_is.dtype),
            X_is,
        ),
        dim=-1,
    )
    # 1) Term 1 : Grad P_x = Eye padded on top with 0s
    grad_eye = (
        torch.eye(p_x.shape[-1] - 1)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(p_x.shape[0], p_x.shape[1], -1, -1)
        .to(p_x.device)
    )
    grad_p_x = torch.cat(
        (
            torch.zeros((grad_eye.shape[0], grad_eye.shape[1], 1, 3))
            .to(X.device)
            .to(X.dtype),
            grad_eye,
        ),
        dim=-2,
    )

    # 2) Term 3 : Grad W(X). This is needed for Grad(M(x))
    grad_weights = grad_weights_batched(X, X_is, radius=radius)
    # 3) Term 2 : Grad(M(x))
    # M_inv = pinv(M)  # (B, 4, 4)
    # M_inv = 
    M_inv = torch.linalg.solve(M, eye_like(M))
    grad_M_inv = grad_moment_mat_batched(X_is, grad_weights, M_inv)

    return_elem = (dist_vec, p_x, P_Xis, M_inv,
                   grad_weights, grad_p_x, grad_M_inv)
    return return_elem


def grad_compute_product_rule(dist_vec, p_x, P_Xis, M_inv, grad_weights, grad_p_x, grad_M_inv):

    term_1 = compute_first_term_batched(grad_p_x, M_inv, P_Xis, dist_vec)

    term_2 = compute_second_term_batched(p_x, grad_M_inv, P_Xis, dist_vec)

    term_3 = compute_third_term_batched(p_x, M_inv, P_Xis, grad_weights)
    D_Phi = term_1 + term_2 + term_3
    # Combine the terms
    return D_Phi


def grad_Phi_b(X, X_is, radius):
    dist_vec, p_x, P_Xis, M_inv, grad_weights, grad_p_x, grad_M_inv = grad_compute_residues(X, X_is, radius)
    D_Phi = grad_compute_product_rule(dist_vec, p_x, P_Xis, M_inv, grad_weights, grad_p_x, grad_M_inv)
    return D_Phi

def grad_f_x_interpolate_batched(X, X_is, F_is, radius):
    # Get the residues
    D_Phi = grad_Phi_b(X, X_is, radius)
    D_F_x = torch.einsum('bimd,bmp->bipd', D_Phi, F_is)
    return D_F_x


def grad_f_x_batched_from_ind(interest_ind,
                              dist_vec_f, p_x_f, P_Xis_f, M_inv_f,
                              grad_weights_f, grad_p_x_f, grad_M_inv_f,
                              F_is):
    # Get the residues
    nb = interest_ind.shape[0]
    dist_vec_s = dist_vec_f[interest_ind]
    p_x_s = p_x_f[interest_ind]
    P_Xis_s = P_Xis_f.unsqueeze(0).expand(nb, -1, -1)
    M_inv_s = M_inv_f[interest_ind]
    # M_inv_fb = M_inv_f.unsqueeze(0).expand(nb, -1, -1, -1)
    # M_inv_s = M_inv_fb[torch.arange(nb).unsqueeze(1), interest_ind]
    grad_weights_s = grad_weights_f[interest_ind]
    grad_p_x_s = grad_p_x_f[interest_ind]
    grad_M_inv_s = grad_M_inv_f[interest_ind]
    return grad_compute_product_rule(dist_vec_s, p_x_s, P_Xis_s, M_inv_s,
                                     grad_weights_s, grad_p_x_s, grad_M_inv_s,
                                     F_is)




def compute_grad_f_x_spatial(spatial_points, nodal_points_grad, predicted_flow, mls_radius):
    # X = convert_to_batch(self.spatial_points, nodal_points_grad.size(0))
    # FIXME : hacky way
    X = spatial_points[:nodal_points_grad.size(0), ...]
    X_is = b3nTobn3(nodal_points_grad)
    F_is = b3nTobn3(predicted_flow)
    grad_f_x_spatial = grad_f_x_interpolate_batched(X, X_is, F_is,
                                                    mls_radius)
    return grad_f_x_spatial


def compute_grad_f_x_spatial_from_ind(interest_ind,
                                      dist_vec_f, p_x_f, P_Xis_f, M_inv_f,
                                      grad_weights_f, grad_p_x_f, grad_M_inv_f,
                                      F_is):
    grad_f_x_spatial = grad_f_x_batched_from_ind(interest_ind,
                                                 dist_vec_f, p_x_f, P_Xis_f, M_inv_f,
                                                 grad_weights_f, grad_p_x_f, grad_M_inv_f,
                                                 F_is)
    return grad_f_x_spatial
