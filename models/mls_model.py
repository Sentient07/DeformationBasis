# mls_model.py
from dataclasses import replace
from lib2to3.pytree import convert
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import os
import trimesh
import pytorch_lightning as pl
import os.path as osp

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from argparse import ArgumentParser
from termcolor import colored
from pdb import set_trace as strc
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from collections import ChainMap
from trimesh.transformations import rotation_matrix

from utils import local_config 
from utils.my_utils import *
from utils.batched_mls_function import *
from utils.GradAnalytic import GradAnalytic
from utils.argument_parsers import get_model_parser

from .encoders import *
from .get_models import *
from .loss import *

from eval_func import *

from utils.batched_mls_function import *


class Nodal_Deformer(pl.LightningModule):
    def __init__(
        self,
        encoder_name,
        decoder_name,
        bottleneck_size,
        latent_size,
        mls_radius,
        lrate,
        **kwargs,
    ):
        super().__init__()

        self.use_cached = False

        self.bottleneck_size = bottleneck_size
        self.mls_radius = mls_radius
        self.kwargs = kwargs

        self.batch_size = self.kwargs["batch_size"]

        # Initialise template
        self.init_template(self.kwargs.get("template_path"))

        # Init Model
        self.pe_enc = self.kwargs.get("pe_enc", False)
        self.pe_dec = self.kwargs.get("pe_dec", False)
        pe_dim = self.kwargs.get("pe_dim", 64)
        self.pe_dim = pe_dim
        self.decoder_name = decoder_name
        coord_dim_enc = 2*(pe_dim//3) * 3 if self.pe_enc else 3
        coord_dim_dec = 2*(pe_dim//3) * 3 if self.pe_dec else 3
        
        self.encoder = get_encoder(encoder_name, bottleneck_size,
                                   latent_size,
                                   input_dim=coord_dim_enc,
                                   pe_dim=pe_dim
                                   )
        self.decoder = get_decoder(decoder_name, bottleneck_size, 
                                   coord_dim=coord_dim_dec,
                                   latent_dim=self.nodal_pts.shape[0])

        # Init CD loss
        self.distChamfer = get_chamfer_loss()
        
        
        self.lr = lrate
        # Coefficients
        self.unsup = kwargs.get('unsup', False)
        self.volp_coeff = kwargs.get("vol_coeff", 0.0)
        self.arap_coeff = kwargs.get("arap_coeff", 0.0)
        self.geo_coeff = kwargs.get("geo_coeff", 0.)
        self.lvcon_coeff = kwargs.get('lvcon_coeff', 0.5)
        self.use_fo = self.arap_coeff > 0.0 or self.volp_coeff > 0.0 or self.lvcon_coeff > 0.0
        self.use_lv_loss = self.lvcon_coeff > 0.0
        
        self.n_intermediate = kwargs.get("n_intermediate", 16)

        # If computing ARAP, check if the system is solvable;
        # Also initialise var independent residuals
        if self.use_fo:
            self.init_nodal_grad()

        ## Inference
        self.wo_test_preprocess = self.kwargs.get('wo_test_preprocess', False)
        self.save_test_mesh = self.kwargs.get('save_test_mesh', False)
        self.dataset_val_2 = self.kwargs.get("dataset_val_2").lower()
        self.cor_freq = self.kwargs.get("eval_corr_freq", 50)
        self.dataset_test = self.kwargs.get("dataset_test").lower()
        self.HR_inference = bool(self.kwargs.get("HR_inference", 1))
        self.resume_recon = self.kwargs["resume_recon"]
        self.cd_iter = self.kwargs["reg_num_steps"]

        refine_suffix = "" if self.cd_iter > 10 else "_wocd"
        self.exp_id = self.kwargs["exp_name"] + "_" + self.kwargs["id"] + refine_suffix

        ### CD constraints
        self.cd_w_arap = self.kwargs.get("cd_w_arap", False)
        self.cd_w_volp = self.kwargs.get("cd_w_volp", False)
        self.cd_w_fo = self.cd_w_arap or self.cd_w_volp

        self.save_hyperparameters()


    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optim, min_lr=1e-7)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_cd",
                "frequency": 10,
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        return get_model_parser(parent_parser)

    def init_template(self, template_path):
        self.template_data = torch.load(template_path)
        self.nodal_pts = self.template_data["nodes"].cuda().float()
        self.nodal_pts_np = numpied(self.nodal_pts)
        self.temp_verts_th = self.template_data["verts"].cuda().float()
        self.temp_verts_np = self.template_data["verts"].numpy()
        self.temp_faces_np = self.template_data["faces"].numpy()
        self.temp_mesh = trimesh_from_vf(self.temp_verts_np, self.temp_faces_np)
        self.temp_mesh_faces = np.array(self.temp_mesh.faces)
        self.temp_mesh_faces_th = torched(self.temp_mesh_faces, dtype=torch.long)
        self.temp_all_edges = self.temp_verts_np[self.temp_mesh.edges]
        self.temp_all_edge_ind_th = torched(numpied(self.temp_mesh.edges), dtype=torch.long)
        
        self.temp_all_edge_len = np.linalg.norm(
            self.temp_all_edges[:, 0, :] - self.temp_all_edges[:, 1, :],
            axis=1,
            keepdims=True,
        )
        self.temp_all_edge_len_th = torched(self.temp_all_edge_len)
        if "verts_hr" in self.template_data:
            self.temp_vertex_HR_th = self.template_data["verts_hr"].cuda().float()
            self.temp_mesh_HR_faces = self.template_data["faces_hr"].long().cpu().numpy()
        if "Phi" in self.template_data:
            self.use_cached = True
            self.Phi_f = torched(self.template_data["Phi"])
            self.dist_vec_f = torched(self.template_data["dist_vec"])
            self.p_x_f = torched(self.template_data["p_x"])
            self.P_Xis_f = torched(self.template_data["P_Xis"])
            self.M_inv_f = torched(self.template_data["M_inv"])
            self.grad_weights_f = torched(self.template_data["grad_weights"])
            self.grad_p_x_f = torched(self.template_data["grad_p_x"])
            self.grad_M_inv_f = torched(self.template_data["grad_M_inv"])
            self.mls_residue = (self.dist_vec_f, self.p_x_f, self.P_Xis_f, self.M_inv_f,
                                 self.grad_weights_f, self.grad_p_x_f, self.grad_M_inv_f)

    def init_nodal_grad(self):
        self.AnlyGradInst = GradAnalytic(self.temp_verts_th, self.nodal_pts, self.mls_radius)


    def get_nodal_flow(self, latent_vec):
        # One nodal point, so make it into batch
        batched_nodes = bn3Tob3n(convert_to_batch(self.nodal_pts, latent_vec.shape[0]))
        batched_nodes = fourier_encode(batched_nodes,
                                       embedding_size=self.pe_dim).transpose(1,2) if self.pe_dec else batched_nodes
        decoded_points = self.decoder(latent_vec, batched_nodes).contiguous()
        return decoded_points

    def get_surface_flow(self, x, temp_sur_all=None):
        lat_vec = self.encoder(x)
        if temp_sur_all is None:
            temp_sur_all = convert_to_batch(self.temp_verts_th, lat_vec.shape[0])
        return self.get_approx_flow_from_latvec(lat_vec, temp_sur_all)

    def get_approx_flow_from_latvec(self, lat_vec, temp_pts_surf,
                                    return_nodal_flow=False):
        predicted_flow = self.get_nodal_flow(lat_vec)
        batched_nodes = convert_to_batch(self.nodal_pts, lat_vec.shape[0])
        surface_flow_approx = compute_flow_approx(predicted_flow, batched_nodes, temp_pts_surf,
                                                  mls_radius=self.mls_radius)
        if return_nodal_flow:
            return surface_flow_approx, predicted_flow
        return surface_flow_approx

    def forward_step(self, target_sur_samp, temp_sur_samp, samp_index):
        n_batch = target_sur_samp.shape[0]
        
        # Conversion for pointnet
        target_sur_samp = bn3Tob3n(target_sur_samp)
        # pointnetFwd pass
        lat_vec = self.encoder(target_sur_samp)
        # Decoding step
        predicted_flow = self.get_nodal_flow(lat_vec)
        all_samp_ind = torch.arange(self.temp_verts_th.shape[0]).unsqueeze(0).expand(n_batch, -1).to()
        temp_sur_all = self.temp_verts_th.unsqueeze(0).expand(n_batch, -1,-1).to(predicted_flow.device)

        if self.use_cached:
            sample_flow_approx = compute_flow_approx_from_ind(predicted_flow, samp_index, self.Phi_f)
            surface_flow_approx = compute_flow_approx_from_ind(predicted_flow, all_samp_ind, self.Phi_f)
            
        else:
            sample_flow_approx_og = compute_flow_approx(predicted_flow, self.nodal_pts, temp_sur_samp, self.mls_radius)
            surface_flow_approx_og = compute_flow_approx(predicted_flow, self.nodal_pts, temp_sur_all, self.mls_radius)
        return sample_flow_approx, predicted_flow, surface_flow_approx
    
    def get_intermediate_flow(self, tar_sur_samp):
        samp_range = torch.arange(tar_sur_samp.shape[0]).to(tar_sur_samp.device)
        full_combo = torch.combinations(samp_range)
        samp_combo = full_combo[np.random.choice(full_combo.shape[0], self.n_intermediate, replace=False)]
        lat_vec = self.encoder(bn3Tob3n(tar_sur_samp))
        latvec_src = lat_vec[samp_combo[:, 0]]
        latvec_tar = lat_vec[samp_combo[:, 1]]
        
        alpha = torch.rand(latvec_src.shape[0]).to(latvec_src.device)
        mid_pts = alpha.unsqueeze(-1)*latvec_src + (1-alpha.unsqueeze(-1))*latvec_tar
        batched_nodes = convert_to_batch(self.nodal_pts, mid_pts.shape[0])
        predicted_flow = b3nTobn3(self.decoder(mid_pts, bn3Tob3n(batched_nodes)))
        return predicted_flow
        

    def training_step(self, batch, batch_idx):
        tar_sur_all, tar_samp_ind, temp_samp_ind, n_tar, gt_map = batch
        n_batch = len(tar_sur_all)
        batch_ind = torch.arange(n_batch).unsqueeze(-1).long()
        tar_sur_samp = tar_sur_all[batch_ind, tar_samp_ind]
        
        temp_sur_batched = convert_to_batch(self.temp_verts_th.unsqueeze(0), n_batch)
        if len(gt_map.size())>1:
            temp_samp_ind = gt_map[:,:,1].long()
        temp_sur_samp = temp_sur_batched[batch_ind, temp_samp_ind]
        
        samp_flow_approx, predicted_flow, sur_flow_approx = self.forward_step(tar_sur_samp,
                                                                              temp_sur_samp,
                                                                              temp_samp_ind)
        
        gt_flow_full = None
        all_losses = self.compute_losses(
                                         tar_sur_samp, temp_sur_samp,
                                         predicted_flow, samp_flow_approx,
                                         sur_flow_approx, gt_flow_full,
                                         gt_map)

        net_loss = 0.0
        loss_log_dict = {}
        for k, v in all_losses.items():
            loss_log_dict[k] = v.item()
            net_loss += v
        self.log_dict(loss_log_dict, on_step=False, on_epoch=True)
        return net_loss

    def compute_losses(self,
                       tar_sur_samp, temp_sur_samp,
                       predicted_flow, samp_flow_approx,
                       surface_flow_approx, gt_flow_full, gt_map):
        n_batch = samp_flow_approx.shape[0]
        all_losses_dict = {}
        

        
        temp_sur_all = self.temp_verts_th.unsqueeze(0).expand(n_batch,-1,-1).to(predicted_flow.device)
        deformed_verts = temp_sur_all + surface_flow_approx
        deformed_verts_sub = temp_sur_samp + samp_flow_approx
        # CD geometric loss
        if self.unsup or self.geo_coeff > 0.0:
            deformed_verts_sub = temp_sur_samp + samp_flow_approx
            dist1, dist2 = self.distChamfer(deformed_verts_sub,
                                            b3nTobn3(tar_sur_samp))
            cd_net = (torch.mean(dist1)) + (torch.mean(dist2))
            all_losses_dict['tr_geo_loss'] = cd_net
        else:
            if len(gt_map.size())>1:
                tar_corr_pts = tar_sur_samp[torch.arange(tar_sur_samp.shape[0]).unsqueeze(-1), gt_map[:,:,0].long()]
                batched_temp = convert_to_batch(self.temp_verts_th, tar_corr_pts.shape[0])
                tmp_corr_pts = batched_temp[torch.arange(batched_temp.shape[0]).unsqueeze(-1), gt_map[:,:,1].long()]
                gt_flow_sampl = b3nTobn3(tar_corr_pts) - b3nTobn3(tmp_corr_pts)
                all_losses_dict["tr_reg_loss"] = ((gt_flow_sampl - samp_flow_approx) ** 2).mean()
                

            else:
                gt_flow_sampl = b3nTobn3(tar_sur_samp) - b3nTobn3(temp_sur_samp)
                all_losses_dict["tr_reg_loss"] = ((gt_flow_sampl - samp_flow_approx) ** 2).mean()
                
            dist1, dist2 = self.distChamfer(deformed_verts_sub,
                                            b3nTobn3(tar_sur_samp))
            cd_net = (torch.mean(dist1)) + (torch.mean(dist2))
            all_losses_dict['tr_geo_loss'] = self.geo_coeff * cd_net

        if self.use_fo:
            F_is = b3nTobn3(predicted_flow)
            grad_f_x_nodal = self.AnlyGradInst.get_grad_nodes(F_is)        
            all_losses_dict["tr_volp_loss"] = self.volp_coeff * loss_volp(grad_f_x_nodal).mean()
            all_losses_dict["tr_arap_loss"] = self.arap_coeff * loss_arap(grad_f_x_nodal).mean()
            
        # get latent loss
        if self.use_lv_loss:
            F_midpt = self.get_intermediate_flow(tar_sur_samp)
            grad_f_x_nodal_lv = self.AnlyGradInst.get_grad_nodes(F_midpt)
            all_losses_dict["tr_volp_loss_lv"] = self.lvcon_coeff * self.volp_coeff * loss_volp(grad_f_x_nodal_lv).mean()
            all_losses_dict["tr_arap_loss_lv"] = self.lvcon_coeff * self.arap_coeff * loss_arap(grad_f_x_nodal_lv).mean()
            
        return all_losses_dict


    def validation_step(self, batch, batch_idx, dataloader_idx):
        do_match_val = (dataloader_idx==1 and (self.current_epoch+1)%self.cor_freq ==0)
        net_loss = 0.
        loss_dict = {}
        if dataloader_idx == 0:
            net_loss = self.recon_validation(batch)
            loss_dict['val_cd_%d'%batch_idx] = net_loss
            return loss_dict

        if do_match_val:
            recon_m, recon_loss = self.run_test_reconstruction(batch, num_cd_iter=1)
            self.log_dict(recon_loss, on_step=True)
            return recon_m
    
    def recon_validation(self, batch):
        tar_sur_all, tar_samp_ind, temp_samp_ind, n_tar, _ = batch
        
        n_batch = len(tar_sur_all)
        n_batch = tar_sur_all.shape[0]
        batch_ind = (torch.arange(n_batch).unsqueeze(-1).long().to(tar_samp_ind.device))
        tar_sur_samp = tar_sur_all[batch_ind, tar_samp_ind]
        temp_sur_all = convert_to_batch(self.temp_verts_th.unsqueeze(0), n_batch)
        temp_sur_samp = temp_sur_all[batch_ind, temp_samp_ind]
        
        _, _, sur_flow_approx = self.forward_step(tar_sur_samp,
                                                  temp_sur_samp,
                                                  temp_samp_ind)

        deformed_template = temp_sur_all + sur_flow_approx
        dist1, dist2 = self.distChamfer(deformed_template, b3nTobn3(tar_sur_all))
        cd_net = (torch.mean(dist1)) + (torch.mean(dist2))
        return cd_net.item()

    def validation_epoch_end(self, outputs):
        val_cd_all = []
        for out in outputs[0]:
            for k,v in out.items():
                if 'val_cd_' in k:
                    val_cd_all.append(v)
            
        self.log_dict({"val_cd": np.mean(val_cd_all)}, on_step=False, on_epoch=True)
        

        if (self.current_epoch+1)%self.cor_freq ==0:
            all_recon_mesh = dict(ChainMap(*outputs[1]))
            test_log_dict = run_evaluation(self.dataset_val_2,
                                           self.exp_id,
                                           all_recon_mesh)
            self.log_dict(test_log_dict, on_step=False, on_epoch=True)

        
    def run_test_reconstruction(self, test_batch, num_cd_iter):
        test_path = Path(test_batch[0])
        test_recon = osp.join(test_path.parent, test_path.stem + "FinalReconstruction.ply")
        if osp.isfile(test_recon) and self.resume_recon:
            return {test_path.stem: trimesh.load(test_recon, process=False)}, {0:0}
        recon_m, recon_loss = self._correspondence_run(test_batch[0],
                                                        num_cd_iter)
        return recon_m, recon_loss

    def test_step(self, test_batch, batch_idx):
        recon_m, recon_loss = self.run_test_reconstruction(test_batch,
                                                           num_cd_iter=self.cd_iter)
        if not self.resume_recon:
            self.log_dict(recon_loss, on_step=True)
        return recon_m

    def test_epoch_end(self, outputs):
        all_recon_mesh = dict(ChainMap(*outputs))
        test_log_dict = run_evaluation(self.dataset_test,
                                       self.exp_id,
                                       all_recon_mesh,
                                       permutation=True)
        self.log_dict(test_log_dict, on_step=False, on_epoch=True)

    def preprocess_inp_mesh(self, input):
        input, scalefactor = scale(input, self.temp_mesh)
        input = uniformize(input)
        input = clean(input)
        input, translation = center(input)
        return input, scalefactor, translation

    def _correspondence_run(self, input_p, num_cd_iter):
        input = trimesh.load(input_p, process=False)
        if not self.wo_test_preprocess:
            input, scalefactor, translation = self.preprocess_inp_mesh(input)
        else:
            input, scalefactor, translation = input, 1., np.array([0., 0., 0.])
        losses_dict = {}
            
        meshReg, recon_m_before_cd, losses_dict = self._test_reconstruct(input,
                                                                         scalefactor,
                                                                         translation,
                                                                         num_cd_iter)

        meshReg_verts = meshReg.vertices if meshReg.vertices.ndim == 2 else meshReg.vertices[0]
        if self.save_test_mesh:
            init_name = input_p[:-4] + "InitialReconstruction.ply"
            final_name = input_p[:-4] + "FinalReconstruction.ply"
            _ = recon_m_before_cd.export(init_name)
            _ = trimesh_from_vf(meshReg_verts, meshReg.faces).export(final_name)

        recon_dict = {Path(input_p).stem: meshReg}
        return recon_dict, losses_dict

    def _test_reconstruct(self, input, scalefactor, translation, num_cd_iter):

        template_surface_pts = convert_to_batch(self.temp_verts_th, 1)

        if input.vertices.shape[0] > 10000:
            rand_ind = np.random.choice(input.vertices.shape[0], 10000, replace=False)
            torched_inp = convert_to_batch(np.array(input.vertices)[rand_ind], 1)
        else:
            torched_inp = convert_to_batch(np.array(input.vertices), 1)

            
        interpolated_flow = self.get_surface_flow(bn3Tob3n(torched_inp),
                                                  template_surface_pts)

        pointsReconstructed = template_surface_pts + interpolated_flow
        dist1, dist2 = self.distChamfer(b3nTobn3(torched_inp).contiguous(),
                                        b3nTobn3(pointsReconstructed))
        recon_m_before_cd = trimesh_from_vf((numpied(b3nTobn3(pointsReconstructed)[0])+ translation) / scalefactor,
                                            self.temp_mesh_faces)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        losses_dict = {"test_loss_init_cd": loss_net.item()}

        if self.HR_inference:
            faces_tosave = self.temp_mesh_HR_faces
        else:
            faces_tosave = self.temp_mesh.faces

        refined_recon, loss_dict = self._test_cd_opt(torched_inp,
                                                     losses_dict=losses_dict,
                                                     num_cd_iter=num_cd_iter)

        meshReg = trimesh_from_vf(
            (refined_recon[0].data.cpu().numpy() + translation) / scalefactor,
            faces_tosave,
        )

        return meshReg, recon_m_before_cd, loss_dict


    @torch.enable_grad()
    def _test_cd_opt(self, points, losses_dict, num_cd_iter):
        points = points.data

        latent_code = self.encoder(bn3Tob3n(points))

        lrate = self.lr
        # define parameters to be optimised and optimiser
        latent_vector = nn.Parameter(latent_code.data, requires_grad=True)
        optimizer = Adam([latent_vector], lr=lrate)

        if self.HR_inference:
            temp_vert_batch = convert_to_batch(self.temp_vertex_HR_th, 1)
        else:
            temp_vert_batch = convert_to_batch(self.temp_verts_th, 1)

        i = 0
        loss = 10

        while np.log(loss) > -9.5 and i < num_cd_iter:
            optimizer.zero_grad()

            template_surface_pt = convert_to_batch(self.temp_verts_th, 1)

            approximated_flow, predicted_flow = self.get_approx_flow_from_latvec(
                latent_vector, template_surface_pt, return_nodal_flow=True
            )
            pointsReconstructed = template_surface_pt + approximated_flow

            # back to (B, N, 3)
            dist1, dist2 = self.distChamfer(
                b3nTobn3(points), b3nTobn3(pointsReconstructed)
            )
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))

            # CD
            loss_cd = loss_net.item()
            vol_preserv_loss = torch.Tensor([0.0]).float()
            if self.cd_w_fo:
                grad_f_x_nodal = self.AnlyGradInst.get_grad_nodes(b3nTobn3(predicted_flow))
            
            if self.cd_w_volp:
                vol_preserv_loss = self.volp_coeff * loss_volp(grad_f_x_nodal).mean()
                loss_net += vol_preserv_loss
                
            if self.cd_w_arap:
                arap_loss = self.arap_coeff * loss_arap(grad_f_x_nodal).mean()
                arap_loss = torch.clamp(arap_loss, min=0.0, max=0.001)
                loss_net += arap_loss

            loss_net.backward()
            optimizer.step()
            loss_net = loss_net.item()
            if num_cd_iter > 10:
                print(
                    f"\r["
                    + f": "
                    + colored(f"{i}", "red")
                    + "/"
                    + colored(f"{int(num_cd_iter)}", "red")
                    + "] reg loss:  "
                    + colored(f"{loss_net}", "yellow"),
                    end="",
                )
            i = i + 1
            vol_preserv_loss = vol_preserv_loss.item()

        final_approximated_flow = self.get_approx_flow_from_latvec(latent_vector, temp_vert_batch)
        pointsReconstructed = temp_vert_batch + final_approximated_flow

        # print(colored(f"loss reg : {loss_net} after {i} iterations", "red"))
        losses_dict["test_loss_final_cd"] = loss_cd

        return pointsReconstructed, losses_dict
