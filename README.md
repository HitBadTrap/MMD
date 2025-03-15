# Realistic Full-Body Motion Generation from Sparse Tracking with State Space Model (ACM MM 2024, Oral)

## Enviroment Setup
All our experiments were conducted on a single A-100 40G GPU.

The code was tested on python 3.9.19, torch 2.2.1, and mamba-ssm 1.2.0.

Download the [human_body_prior](https://github.com/nghorbani/human_body_prior/tree/master/src) lib and [body_visualizer](https://github.com/nghorbani/body_visualizer/tree/master/src) lib and put them in this repo. The repo should look like
```
agrol
├── body_visualizer
├──── mesh/
├──── tools/
├──── ...
├── human_body_prior/
├──── body_model/
├──── data/
├──── ...
├── dataset/
├── prepare_data/
└── ...
```

## Dataset Preparation
Please download the AMASS dataset from [here](https://amass.is.tue.mpg.de/)(SMPL+H G).
```
python prepare_data.py --support_dir /path/to/your/smplh/dmpls --save_dir ./dataset/AMASS/ --root_dir /path/to/your/amass/dataset
```
The generated dataset should look like this
```
./dataset/AMASS/
├── BioMotionLab_NTroje
├──── train/
├──── test/
├── CMU/
├──── train/
├──── test/
└── MPI_HDM05/
├──── train/
└──── test/
```

## Evaluation
```
python test.py --model_path /path/to/your/model --timestep_respacing ddim5 --support_dir /path/to/your/smpls/dmpls --dataset_path ./dataset/AMASS/
```

## Training
```
python train.py --save_dir /path/to/save/your/model --dataset amass --weight_decay 1e-4 --batch_size 128 --lr 3e-4 --latent_dim 128 --save_interval 1 --log_interval 1 --device 0 --input_motion_length 96 --diffusion_steps 1000 --num_workers 4 --motion_nfeat 132 --arch diffusion_DiffMotionUNet --layers 12 --sparse_dim 54 --train_dataset_repeat_times 1000 --lr_anneal_steps 225000 --overwrite --if_bidirectional --if_channel
```

## Pretrained Weights
The pretrained weights for MMD can be downloaded from this [link](https://pan.baidu.com/s/1HOjsEcTmFuW_XYV53awy3g?pwd=4wxb).

To test the pretrained model:
```
python test.py --model_path mmd_pretrained_weights/diff_motion_unet.pt --timestep_respacing ddim5 --support_dir /path/to/your/smpls/dmpls --dataset_path ./dataset/AMASS/
```

To visualize the generated motions, add these commands behind:
```
--vis --output_dir /path/to/save/your/videos
```

## Trouble Shooting

If you encounter this error during visualization:
```
ValueError: Cannot use face colors with a smooth mesh
```
You can fix it by changing the line 88 in your `body_visualizer/mesh/mesh_viewer.py` to:
```
mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
```

## Reference

* [AGRoL](https://github.com/facebookresearch/AGRoL)


## Citation
If you want to cite our work, please use this:

```
@inproceedings{dong2024realistic,
  title={Realistic Full-Body Motion Generation from Sparse Tracking with State Space Model},
  author={Dong, Kun and Xue, Jian and Niu, Zehai and Lan, Xing and Lv, Ke and Liu, Qingyuan and Qin, Xiaoyu},
  booktitle={ACM Multimedia 2024}
}

```


Open an issue or mail me directly in case of any queries or suggestions.
