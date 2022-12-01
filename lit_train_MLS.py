
import argparse
import os.path as osp

import pytorch_lightning as pl
import tensorboard
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset.lightning_data import HumanDataModule
from models.mls_model import Nodal_Deformer
from utils import argument_parsers
from utils.my_utils import str2bool
from pytorch_lightning.loggers import TestTubeLogger
from pathlib import Path
from pdb import set_trace as strc

pl.seed_everything(1234)
# torch.backends.cuda.matmul.allow_tf32=False

if __name__ == '__main__':
    parser = argument_parsers.get_init_parser()
    parser = Nodal_Deformer.add_model_specific_args(parser)
    args = parser.parse_args()
    data_module = HumanDataModule.from_argparse_args(args)

    log_root = "./Logs/%s" % args.exp_name
    tt_logger = TestTubeLogger(log_root, name=args.id)
    checkpoint_dir = (Path(tt_logger.save_dir)
                      / tt_logger.experiment.name
                      / "checkpoints"
                      )

    checkpoint_callback = ModelCheckpoint(monitor="val_cd",
                                          dirpath=checkpoint_dir,
                                          filename="{epoch:02d}-{val_cd:.6f}",
                                          save_last=True)
    ckpt_path = args.model if bool(args.model) else None
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    is_deterministic = True
    trainer = pl.Trainer.from_argparse_args(args, gpus=str(args.gpus),
                                            benchmark=True,
                                            deterministic=is_deterministic,
                                            callbacks=[
                                                checkpoint_callback, lr_monitor],
                                            logger=tt_logger,
                                            default_root_dir=log_root,
                                            max_epochs=args.nepoch,
                                            check_val_every_n_epoch=10)
    if not args.only_test:
        model = Nodal_Deformer(**vars(args))
        trainer.fit(model, data_module, ckpt_path=ckpt_path,)
    ckpt = ckpt_path if bool(ckpt_path) else osp.join(
        checkpoint_dir, 'last.ckpt')
    model_test = Nodal_Deformer.load_from_checkpoint(ckpt, **vars(args))
    trainer.test(model=model_test, datamodule=data_module,
                 ckpt_path=ckpt)
