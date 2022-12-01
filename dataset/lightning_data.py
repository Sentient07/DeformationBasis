# lightning_data.py
from threading import local
import torch
import numpy as np
import pytorch_lightning as pl
import os.path as osp

from torch.utils.data import DataLoader
from .dataset import *
from utils import local_config


class HumanDataModule(pl.LightningDataModule):
    name = 'SURREAL'

    def __init__(
        self,
        dataset_train: str,
        dataset_val: str,
        template_path: str,
        workers: int = 16,
        batch_size: int = 32,
        batch_size_test: int = 5,
        seed: int = 1234,
        n_surf_points: int = 2000,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        data_augment: bool = True,
        range_rot: int = 180,
        use_bary_pts: bool = False,
        dataset_val_2: str = None,
        dataset_test: str = 'scape_r_n05',
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        print("Data Augmentation : %s, degree %d " % (str(data_augment), range_rot))
        assert osp.isfile(template_path), 'Template path not found'

        default_collate = torch.utils.data.dataloader.default_collate
        self.collate_tr = my_collate_fn
        self.collate_val = my_collate_fn
        train_class, train_dir = dataset_name_wrapper(dataset_train, is_train=True)
        val_class, val_dir = dataset_name_wrapper(dataset_val, is_train=False)
        val_class_2, val_dir_2 = test_dataset_wrapper(dataset_val_2)
        test_class, test_dir = test_dataset_wrapper(dataset_test)

        self.trainset = train_class(dir_path=train_dir,
                                    template_path=template_path,
                                    is_train=True,
                                    data_augment=data_augment,
                                    range_rot=range_rot,
                                    n_surf_points=n_surf_points,
                                    use_bary_pts=use_bary_pts)

        self.valset = val_class(dir_path=val_dir,
                                template_path=template_path,
                                is_train=False,
                                data_augment=data_augment,
                                range_rot=range_rot,
                                n_surf_points=n_surf_points,
                                use_bary_pts=False)

        self.valset_2 = val_class_2(mesh_dir=test_dir)

        self.testset = test_class(mesh_dir=test_dir)

    def train_dataloader(self):
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            collate_fn=self.collate_tr
        )
        return loader

    def val_dataloader(self, **kwargs):
        recon_val_loader = DataLoader(self.valset,
                                        batch_size=min(len(self.valset), self.batch_size),
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        drop_last=False,
                                        pin_memory=self.pin_memory,
                                        persistent_workers=self.num_workers>0,
                                        collate_fn=self.collate_val
        )
        corresp_val_loader = DataLoader(self.valset_2,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        drop_last=False,
                                        pin_memory=False,
                                        )
        return [recon_val_loader, corresp_val_loader]

    def test_dataloader(self):
        loader = DataLoader(
            self.testset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        return loader
