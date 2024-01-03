# DeformationBasis
Official code and model release for our NeurIPS 2022 paper [Reduced Representation of Deformation Fields for Effective Non-rigid Shape Matching](https://arxiv.org/abs/2211.14604).

## Setup
This code is primarily written in Python 3.7 using Pytorch but with additional dependencies that are not packaged using pip/conda.

1. Setup the conda environment with Pytorch, Trimesh, etc.. from `my_env.yml`.

2. Install Chamfer's Distance from [here](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/36f82e31d754caf7d409c83bdcf0d82d87d2fd55).

3. Install pymesh as follows,

```
pip install http://imagine.enpc.fr/~langloip/data/pymesh2-0.2.1-cp37-cp37m-linux_x86_64.whl
```

4. Setup respective dataset, ground truth directories in local_config.py. Please follow the instructions in the comments provided.

5. For evaluation, we use `CorrespondenceEvaluator` package. To setup please do as follows,

```
git clone https://github.com/Sentient07/CorrespondenceEvaluator.git && cd CorrespondenceEvaluator && pip install -e .
```

## Training
We provide the template data including pre-computed basis function, nodes, etc in `./data/` directory. To train the model, run the following command,

```
python lit_train_MLS.py --exp_name test --id 1 --pe_enc --cd_w_volp --cd_w_arap
```

Please refer to arguments in utils/argument_parsers.py for more details.

## Evaluation

Once trained use `--only_test` and `--model` arguments to restore the model and evaluate it, e.g.,

```
python lit_train_MLS.py --exp_name test --id 1 --pe_enc --cd_w_volp --cd_w_arap --only_test --model /path/to/checkpoint.ckpt
```


## Pretrained models

Pre-trained models used in our quantitative experiments can be found here: https://nuage.lix.polytechnique.fr/index.php/s/oP2zBQy7ScHRxN6 

## :hourglass_flowing_sand: Coming Soon...

- [x] Pre-trained weights to reproduce.

- [ ] Dataset used in all our experiments.

- [x] Pre-processing code to obtain basis function $`\Phi`$, its gradient, etc.. (Check `batched_mls_function.py`) 

- [ ] Code for shape interpolation.

## Citation

If you find this code useful, please cite our paper,

```

@article{Sundararaman2022DeformBasis,
  title={Reduced Representation of Deformation Fields for Effective Non-rigid Shape Matching},
  author={Sundararaman, Ramana and Marin, Riccardo and Rodola, Emanuele and Ovsjanikov, Maks},
  year={2022},
  journal={Advances in Neural Information Processing Systems},
 	volume={35},
}

```
