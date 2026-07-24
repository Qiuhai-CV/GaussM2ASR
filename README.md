# GaussM2ASR

Official PyTorch implementation of "Adaptive Anisotropic Gaussian Splatting
for Multi-contrast MRI Arbitrary-Scale Super-Resolution with Anatomy
Guidance."

This release contains the training and testing code for GaussM2ASR.

## Environment

The released configuration was developed with Python 3.10, PyTorch 2.0.1, and
CUDA 11.8.

```bash
conda create -n gaussm2asr python=3.10
conda activate gaussm2asr
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
BASICSR_EXT=True python setup_basicsr.py develop
```

## Data Preparation

The default YAML files expect the following paths relative to this directory:

```text
datasets/BraTS/train/HR/
datasets/BraTS/train/guide/
datasets/BraTS/test/HR/
datasets/BraTS/test/guide/
```

For every target image under `HR`, the anatomy guidance image is resolved by
replacing `HR` with `guide` and `T2` (or `t2`) with `T1` (or `t1`) in its
path. Adjust `all_gt_list` in the YAML files if your dataset uses a different
layout.

## CUDA Renderer

The default configurations use `cuda_rendering: True` and `if_dmax: True`.
Compile the two local CUDA extensions once from the repository root:

```bash
python setup_gscuda.py build_ext --inplace
```

This requires an NVIDIA GPU, a CUDA Toolkit with `nvcc`, and a C++ compiler
compatible with the installed PyTorch build. The generated extension files are
platform-specific and are ignored by Git. To use the slower PyTorch renderer,
set `cuda_rendering: False` in the selected YAML file.

## Training

### Stage 1: pretraining

Stage 1 uses the GT image as the input. The supplied
`options/train/paper/train_GaussM2ASR_stage1.yml` sets `scale: 1` and both
training and validation `scale_list` values to `[1.0, 1.0]`.

The `train.stage` field controls AGGP updates automatically:
- With `stage: stage1`, AGGP is trainable and the training loop executes
  `self.optimizer_fea2gs.step()` and `self.optimizer_fea2gs.zero_grad()`.
- With `stage: stage2`, AGGP is fixed and those exact two calls are skipped.

No manual code commenting or uncommenting is required.

```bash
python basicsr/train.py -opt options/train/paper/train_GaussM2ASR_stage1.yml
```

### Stage 2: fine-tuning

The Stage 2 YAML sets `train.stage: stage2`. Set
`path_fea2gs.pretrain_network_fea2gs` in
`options/train/paper/train_GaussM2ASR_stage2.yml` to the checkpoint, then run:

```bash
python basicsr/train.py -opt options/train/paper/train_GaussM2ASR_stage2.yml
```

## Testing

Set `path.pretrain_network_g` and `path_fea2gs.pretrain_network_fea2gs` in
the selected test YAML to the checkpoints being evaluated.

Stage 1 testing uses `options/test/paper/test_GaussM2ASR_stage1.yml`, which
sets `scale: 1` and `scale_list: [1.0, 1.0]`.

```bash
python basicsr/test.py -opt options/test/paper/test_GaussM2ASR_stage1.yml
python basicsr/test.py -opt options/test/paper/test_GaussM2ASR_stage2.yml
```

Predictions and evaluation logs are saved under `results/`.

## License and Acknowledgements

See [LICENSE/README.md](LICENSE/README.md) for license texts and upstream
acknowledgements.
