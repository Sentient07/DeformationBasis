import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as strc


from utils.my_utils import convert_to_batch, bn3Tob3n, fourier_encode


class PointNet(nn.Module):
    def __init__(self, bottleneck_size=1024, latent_size=1024, pe_dim=64):
        """Encoder"""

        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, latent_size, 1)
        self.lin1 = nn.Linear(latent_size, latent_size)
        self.lin2 = nn.Linear(latent_size, bottleneck_size)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(latent_size)
        self.bn4 = torch.nn.BatchNorm1d(latent_size)
        self.bn5 = torch.nn.BatchNorm1d(bottleneck_size)

        self.latent_size = latent_size

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_size)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


class SlimPointNet(nn.Module):
    def __init__(self, bottleneck_size=1024, latent_size=512, input_dim=3, pe_dim=64):
        """Encoder"""

        super(SlimPointNet, self).__init__()
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.conv1 = torch.nn.Conv1d(input_dim, latent_size//8, 1)
        self.conv2 = torch.nn.Conv1d(latent_size//8, latent_size//4, 1)
        self.conv3 = torch.nn.Conv1d(latent_size//4, latent_size//2, 1)
        self.lin1 = nn.Linear(latent_size//2, latent_size)
        self.lin2 = nn.Linear(latent_size, bottleneck_size)
        self.dp = nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm1d(latent_size//8)
        self.bn2 = torch.nn.BatchNorm1d(latent_size//4)
        self.bn3 = torch.nn.BatchNorm1d(latent_size//2)
        self.bn4 = torch.nn.BatchNorm1d(latent_size)
        self.bn5 = torch.nn.BatchNorm1d(bottleneck_size)

        self.latent_size = latent_size

    def forward(self, x):
        pc = x.detach().clone()
        x = fourier_encode(x, embedding_size=self.pe_dim).transpose(2,1) if self.input_dim > 3 else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_size//2)
        x = self.dp(F.relu(self.bn4(self.lin1(x).unsqueeze(-1))))
        x = self.lin2(x.squeeze(2)).unsqueeze(-1)
        return x.squeeze(2)


class SlimPointNetWoBN(nn.Module):
    def __init__(self, bottleneck_size=1024, latent_size=512, input_dim=3, pe_dim=64):
        """Encoder"""

        super(SlimPointNetWoBN, self).__init__()
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.conv1 = torch.nn.Conv1d(input_dim, latent_size//8, 1)
        self.conv2 = torch.nn.Conv1d(latent_size//8, latent_size//4, 1)
        self.conv3 = torch.nn.Conv1d(latent_size//4, latent_size//2, 1)
        self.lin1 = nn.Linear(latent_size//2, latent_size)
        self.lin2 = nn.Linear(latent_size, bottleneck_size)
        self.dp = nn.Dropout(p=0.3)

        self.latent_size = latent_size

    def forward(self, x):
        pc = x.detach().clone()
        x = fourier_encode(x, embedding_size=self.pe_dim).transpose(2,1) if self.input_dim > 3 else x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_size//2)
        x = self.dp(F.relu(self.lin1(x).unsqueeze(-1)))
        x = self.lin2(x.squeeze(2)).unsqueeze(-1)
        return x.squeeze(2)
    
    
class DeeperPointNet(nn.Module):
    def __init__(self, bottleneck_size=1024, latent_size=512, input_dim=3, pe_dim=64):
        """Encoder"""
        print("Bottleneck %d " %bottleneck_size)
        print("Latent dim %d " %latent_size)
        
        super(DeeperPointNet, self).__init__()
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.conv1 = torch.nn.Conv1d(input_dim, latent_size//16, 1)
        self.conv2 = torch.nn.Conv1d(latent_size//16, latent_size//8, 1)
        self.conv3 = torch.nn.Conv1d(latent_size//8, latent_size//4, 1)
        self.conv4 = torch.nn.Conv1d(latent_size//4, latent_size//2, 1)
        self.conv5 = torch.nn.Conv1d(latent_size//2, latent_size, 1)
        self.conv6 = torch.nn.Conv1d(latent_size, 2*latent_size, 1)
        
        self.lin1 = nn.Linear(2*latent_size, bottleneck_size)
        self.lin2 = nn.Linear(bottleneck_size, bottleneck_size)
        self.dp = nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm1d(latent_size//16)
        self.bn2 = torch.nn.BatchNorm1d(latent_size//8)
        self.bn3 = torch.nn.BatchNorm1d(latent_size//4)
        self.bn4 = torch.nn.BatchNorm1d(latent_size//2)
        self.bn5 = torch.nn.BatchNorm1d(latent_size)
        self.bn6 = torch.nn.BatchNorm1d(latent_size*2)
        self.bn7 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn8 = torch.nn.BatchNorm1d(bottleneck_size)

        self.latent_size = latent_size

    def forward(self, x):
        pc = x.detach().clone()
        nbatch = x.shape[0]
        x = fourier_encode(x, embedding_size=self.pe_dim).transpose(2,1) if self.input_dim > 3 else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x, _ = torch.max(x, 2)
        x = x.view(nbatch, -1)
        x = self.dp(F.relu(self.bn7(self.lin1(x).unsqueeze(-1))))
        x = F.relu(self.bn8(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)
    

class Folding(nn.Module):
    def __init__(self, bottleneck_size=2500, latent_dim=None, coord_dim=3):
        """
        bottleneck_size is input LV dim. 
        Don't care about latent_dim. Also don't delete.
        """

        self.bottleneck_size = bottleneck_size+coord_dim
        super(Folding, self).__init__()
        print("bottleneck_size", bottleneck_size)
        self.conv1 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(
            self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def morph_points_MLSFull(self, latent_vec, temp_vert):
        return self.morph_points_MLS(latent_vec, rand_grid=temp_vert)[0]

    def morph_points_MLS(self, latent_vec, rand_grid):
        rand_grid = convert_to_batch(rand_grid, latent_vec.size(0))
        if rand_grid.requires_grad is True:
            rand_grid = rand_grid.transpose(1, 2).retain_grad()
        else:
            rand_grid = rand_grid.transpose(1, 2).contiguous()
        y = latent_vec.unsqueeze(2).expand(latent_vec.size(
            0), latent_vec.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()

        return y

    def morph_points_MLSFull_HR(self, latent_vec, temp_vertex_HR_th):

        outs = []
        # Process in batch
        div = 20
        batch = int(temp_vertex_HR_th.shape[0] / div)
        for i in range(div - 1):
            rand_grid = temp_vertex_HR_th[batch * i:batch * (i + 1)].view(latent_vec.size(0),
                                                                          batch, 3).transpose(1, 2).contiguous()
            y = latent_vec.unsqueeze(2).expand(latent_vec.size(0),
                                               latent_vec.size(1),
                                               rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self(y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = temp_vertex_HR_th[batch * i:].view(
            latent_vec.size(0), -1, 3).transpose(1, 2).contiguous()
        y = latent_vec.unsqueeze(2).expand(latent_vec.size(
            0), latent_vec.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        outs.append(self(y))
        torch.cuda.synchronize()
        return torch.cat(outs, 2).contiguous()

    def forward(self, lat_vec, nodal_points):
        nodal_points = nodal_points.transpose(2, 1)
        x = self.morph_points_MLS(lat_vec, nodal_points)
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2 * self.th(self.conv4(x))
        return x


class FoldingWoBN(nn.Module):
    def __init__(self, bottleneck_size=2500, latent_dim=None, coord_dim=3):
        """
        bottleneck_size is input LV dim. 
        Don't care about latent_dim. Also don't delete.
        """

        self.bottleneck_size = bottleneck_size+coord_dim
        super(FoldingWoBN, self).__init__()
        print("bottleneck_size", bottleneck_size)
        self.conv1 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(
            self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()

    def morph_points_MLSFull(self, latent_vec, temp_vert):
        return self.morph_points_MLS(latent_vec, rand_grid=temp_vert)[0]

    def morph_points_MLS(self, latent_vec, rand_grid):
        rand_grid = convert_to_batch(rand_grid, latent_vec.size(0))
        if rand_grid.requires_grad is True:
            rand_grid = rand_grid.transpose(1, 2).retain_grad()
        else:
            rand_grid = rand_grid.transpose(1, 2).contiguous()
        y = latent_vec.unsqueeze(2).expand(latent_vec.size(
            0), latent_vec.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()

        return y

    def morph_points_MLSFull_HR(self, latent_vec, temp_vertex_HR_th):

        outs = []
        # Process in batch
        div = 20
        batch = int(temp_vertex_HR_th.shape[0] / div)
        for i in range(div - 1):
            rand_grid = temp_vertex_HR_th[batch * i:batch * (i + 1)].view(latent_vec.size(0),
                                                                          batch, 3).transpose(1, 2).contiguous()
            y = latent_vec.unsqueeze(2).expand(latent_vec.size(0),
                                               latent_vec.size(1),
                                               rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self(y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = temp_vertex_HR_th[batch * i:].view(
            latent_vec.size(0), -1, 3).transpose(1, 2).contiguous()
        y = latent_vec.unsqueeze(2).expand(latent_vec.size(
            0), latent_vec.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        outs.append(self(y))
        torch.cuda.synchronize()
        return torch.cat(outs, 2).contiguous()

    def forward(self, lat_vec, nodal_points):
        nodal_points = nodal_points.transpose(2, 1)
        x = self.morph_points_MLS(lat_vec, nodal_points)
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = 2 * self.th(self.conv4(x))
        return x

class LitePointNet(nn.Module):
    def __init__(self, bottleneck_size=64, latent_size=128, input_dim=3, pe_dim=64):
        """Encoder"""

        super(LitePointNet, self).__init__()
        self.input_dim = input_dim
        self.pe_dim = pe_dim
        self.conv1 = torch.nn.Conv1d(input_dim, latent_size//8, 1)
        self.conv2 = torch.nn.Conv1d(latent_size//8, latent_size//4, 1)
        self.conv3 = torch.nn.Conv1d(latent_size//4, latent_size//2, 1)
        self.lin1 = nn.Linear(latent_size//2, latent_size//2)
        self.lin2 = nn.Linear(latent_size//2, bottleneck_size)
        # self.dp = nn.Dropout(p=0.5)

        self.latent_size = latent_size

    def forward(self, x):
        pc = x.detach().clone()
        x = fourier_encode(x, embedding_size=self.pe_dim).transpose(2,1) if self.input_dim > 3 else x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_size//2)
        x = F.relu(self.lin1(x).unsqueeze(-1))
        x = self.lin2(x.squeeze(2)).unsqueeze(-1)
        return x.squeeze(2)
    