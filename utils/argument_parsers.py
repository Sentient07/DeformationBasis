from argparse import ArgumentParser

from numpy import require
from utils.my_utils import str2bool


def get_init_parser():
    parser = ArgumentParser()
    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        default=16, help='input batch size')
    parser.add_argument('--batch_size_test', type=int,
                        default=5, help='input batch size test')
    parser.add_argument('--gpus', type=int, default=1, help='GPU to use')
    parser.add_argument('--strategy', type=str,
                        default=None)
    parser.add_argument('--nepoch', type=int, default=1200,
                        help='number of epochs to train for')
    parser.add_argument('--only_test', action='store_true', help='Only test')
    parser.add_argument('--unsup', action='store_true', help='Unsup Loss')
    

    # DataLoader
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=32)
    
    parser.add_argument('--n_surf_points', type=int,
                        default=3000, help='Num points for Mapping network')

    parser.add_argument('--use_bary_pts', action='store_true',
                        help='Use surface pts within triangle')

    # Data
    parser.add_argument('--dataset_train', type=str, dest="dataset_train",
                        default="surreal1k",
                        choices=('surreal1k',
                                 'shrec20_a', 'chairs', 'planes',
                                 'tables'),
                        help='training dataset')
    parser.add_argument('--dataset_val', type=str, dest="dataset_val",
                        default="surreal_val",
                        choices=('surreal_val', 'scape_r_n05', 'shrec20_a',
                                 'chairs', 'planes', 'tables'),
                        help='val dataset')
    parser.add_argument('--dataset_val_2', type=str, dest="dataset_val_2",
                        default="scape_r_n05", choices=('scape_r_n05',
                                                        'chairs', 'planes', 'tables'),
                        help='val dataset 2')
    parser.add_argument('--dataset_test', type=str,
                        default="scape_r_n05",
                        choices=('scape_r_n05', 
                                 'shrec19_o',
                                 'shrec20_a_test', 
                                 'chairs', 'planes', 'tables',
                                 'panoptic'
                                 ), 
                        help='test datasets')
    parser.add_argument("--template_path",
                        dest="template_path", default='data/temp_data.pth', type=str)
    parser.add_argument('--data_augment', type=str2bool, nargs='?', default=True,
                        help='Data Augmentation')
    parser.add_argument('--range_rot', type=int, default=360,
                        help='Range of rotation')

    # Save dirs and reload
    parser.add_argument('--id', type=str, help='training ID')
    parser.add_argument('--exp_name', required=True, type=str, help='training name')
    parser.add_argument('--dir_name', type=str, default="",  help='dirname')
    parser.add_argument('--model', type=str, default='', help='optional reload model path')
    
    parser.add_argument('--n_intermediate', type=int, default=16, help='Num intermediate shapes')

    return parser


def get_model_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--bottleneck_size", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=1024)

    # MLS
    parser.add_argument('--mls_radius', type=float, default=0.20, help='Radius for weighting')
    parser.add_argument("--encoder",  dest="encoder_name", type=str, default='SlimPointNet',
                        choices=('PointNet', 'SlimPointNet', 'LitePointNet'), help='Encoder architectures')
    parser.add_argument('--decoder', dest="decoder_name", default='Folding',
                        choices=('Folding', 'FoldingWoBN'),
                        type=str, help='Decoder architectures')

    parser.add_argument('--lrate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--pe_enc', action='store_true', help='Positional encoding input')
    parser.add_argument('--pe_dim', type=int, default=64, help='Positional encoding dim')
    parser.add_argument('--pe_dec', action='store_true', help='Positional encoding folding')
    parser.add_argument('--wo_test_preprocess', action='store_true', help='Positional encoding folding')
    
    # Coefficients
    parser.add_argument('--vol_coeff', type=float, default=5e-3,
                        help='Coefficient for volume preservation')
    parser.add_argument('--arap_coeff', type=float, default=1e-3,
                        help='Coefficient for ARAP')
    parser.add_argument('--geo_coeff', type=float,  default=0.0,
                        help='Coefficient for Chamfer Distance')
    parser.add_argument('--lap_coeff', type=float,  default=0.,
                        help='Laplacian coeff')
    parser.add_argument('--lvcon_coeff', type=float,  default=0.5,
                        help='Latent constraint coefficient')

    # Eval parameters
    parser.add_argument('--save_test_mesh', action='store_true',
                        help='Save test meshes')
    parser.add_argument('--resume_recon', action='store_true',
                        help='Resumes reconstruction if meshes are saved')
    parser.add_argument('--eval_corr_freq', type=int, default=300,
                        help='Eval Matching every k epoch')
    parser.add_argument('--HR_inference', type=int, default=0,
                        help='Use high Resolution template')
    parser.add_argument('--reg_num_steps', type=int,
                        default=2000, help='number of refinement steps')
    parser.add_argument('--cd_w_volp', action='store_true',
                        help='Refinement with volume preservation')
    parser.add_argument('--cd_w_arap', action='store_true',
                        help='Refinement with arap')


    return parser
