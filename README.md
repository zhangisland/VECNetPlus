# VECNet++: Learning Video Exposure Correction: Large-Scale Real-World Dataset and Method

> Journal extension of "Learning Exposure Correction in Dynamic Scenes" (ACMMM24 Oral)


## Requirements
```bash
conda create -n vecnet python=3.10
conda activate vecnet
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Prepare Datasets
We construct two datasets in the journal extension version: DIMEPlus and ProCaptureVideo.
- You can download [DIMEPlus dataset](https://dx.doi.org/10.21227/w9yn-y292) in IEEE Dataport.
- Our collected ProCaptureVideo dataset is provided in [ProCaptureVideo README](datasets/ProCaptureVideo.txt).

We use the `.txt` file as the dataloader index.

```bash
python datasets/prepare_DIMEPlus.py --local_dataset_path <YOUR_DATASET_PATH_TO_LQ> --phase train

e.g., python datasets/prepare_DIMEPlus.py --local_dataset_path datasets/DIMEPlus/train/LQ --phase train
```
The generated txt file will be stored in [dataPath](dataPath/) folder, format like:
`lq_path|gt_path|exposure_label`



## Train
> [!WARNING]
> Not support DDP now.

```bash
python configs/config.py
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/train.yaml --opts SYSTEM.fast_validation False
```
The training results will be saved in `exp_runs`



## Test
```bash
# define the pretrained model path in config 
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/test.yaml  
# or from the command line
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/test.yaml --opts RESUME.ckpt_path <PRETRAINED_PATH>
```
The testing results will be saved in `exp_inference`

# Acknowledgements
If you find our work useful, please cite:
```bibtex
@article{zhangLearningExposure2026,
  title = {Learning Video Exposure Correction: Large-Scale Real-World Dataset and Method},
  author = {Zhang, Siru and Fu, Huiyuan and Liu, Jin and Wang, Bo and Wang, Chuanming and Sun, Chongqi and Ma, Huadong},
  year = 2026,
}

@inproceedings{liuLearningExposure2024,
  title = {Learning Exposure Correction in Dynamic Scenes},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  author = {Liu, Jin and Wang, Bo and Wang, Chuanming and Fu, Huiyuan and Ma, Huadong},
  year = 2024,
  pages = {3858--3866},
}
```
