# nodeGradAnalytic.py

import torch
from utils.my_utils import check_4_nonplanar_node, numpied, b3nTobn3
from .batched_mls_function import grad_Phi_b, grad_compute_product_rule

class GradAnalytic(object):

    def __init__(self, surface_pts, nodal_pts, radius, check_planar=False):
        self.surface_pts = surface_pts.double().unsqueeze(0)
        self.nodal_pts = nodal_pts.double().unsqueeze(0)
        self.radius = radius
        if check_planar:
            print(" [*] Checking if nodes aren't planar to each other")

            check_4_nonplanar_node(fps_pts=numpied(self.nodal_pts),
                                   sur_pts_scaled=numpied(self.nodal_pts),
                                   radius=self.mls_radius)
        
        self.D_Phi_sur = grad_Phi_b(self.surface_pts, self.nodal_pts, self.radius)
        self.D_Phi_node = grad_Phi_b(self.nodal_pts, self.nodal_pts, self.radius)


    def get_grad_nodes(self, predicted_flow):
        F_is = b3nTobn3(predicted_flow).double()
        n_batch = F_is.shape[0]
        Phi_expanded = self.D_Phi_node.expand(n_batch, -1, -1, -1)
        grad_f_x_nodal = torch.einsum('bimd,bmp->bipd', Phi_expanded, F_is)
        return grad_f_x_nodal.to(predicted_flow.dtype)

    def get_grad_surf(self, predicted_flow):
        F_is = b3nTobn3(predicted_flow).double()
        n_batch = F_is.shape[0]
        Phi_expanded = self.D_Phi_sur.expand(n_batch, -1, -1, -1)
        grad_f_x_sur = torch.einsum('bimd,bmp->bipd', Phi_expanded, F_is)
        return grad_f_x_sur.to(predicted_flow.dtype)
