# Progressive Subsampling for Oversampled Data - Application to Quantitative MRI

This repository is the official implementation of our paper accepted in: Medical Image Computing and Computer Assisted Intervention (MICCAI) 2022 [MICCAI version](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_40) and [arxiv version](https://arxiv.org/abs/2203.09268).

Please consider citing:
```
@article{blumberg2022,
    author={Stefano B. Blumberg and Hongxiang Lin and Francesco Grussu and Yukun Zhou and Matteo Figini and Daniel C. Alexander},
    title={Progressive Subsampling for Oversampled Data - Application to Quantitative {MRI}},
    journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
    year = {2022}
}
```

## Requirements

### Base Installation

For the GPU version:

```bash
$ conda create -n PROSUB python=3.6.8 # Optional create new conda environment
$ . activate PROSUB # Optional activate conda environment
$ conda install -c conda-forge cudatoolkit=9.2 cudnn=7.6.5 # For tensorflow GPU, change values of cuda and cudnn
$ pip install -r requirements.txt # May have to add -I flag depending on system
```

### Installing AutoKeras

Install autokeras from source version 1.0.16.post1 and modify few lines.

```bash
$ git clone https://github.com/keras-team/autokeras.git --branch 1.0.16.post1 --single-branch
$ cd autokeras; python setup.py install; cd ..
$ git apply autokeras_diff.patch
```

### Our Configuration
- CentOS 8
- cuda 10.1
- cudnn 7.6.5
- Titan V GPU

## Data Preparation

To use MUDI data from paper:
- Register and download [link](https://www.developingbrain.co.uk/data)
- Create a dictionary of numpy arrays, with keys: '0011' as np.array no_samplesx1344 data from subject 11 et.c

Example to create dummy data:
```python
import numpy as np; import pickle
no_samples=1000
no_measurements=1344
data_keys = ('0011','0012','0013','0014','0015')
data = {subj: np.random.rand(no_samples,no_measurements) for subj in data_keys}
with open('data.pkl','wb') as f:  pickle.dump(data, f)
```


## Results

Main results from Table 1 in the paper.

Whole brain Mean-Squared-Error between $ N=1344 $ reconstructed measurements and $ N $ ground-truth measurements, on leave-one-out cross validation on five MUlti-DIffusion (MUDI) challenge subjects.  The SARDU-Net won the MUDI challenge.

|      Model       |      M = 500               |            250             |	           100	          |             50             |             40             |             30             |             20             |            10              |
| ---------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| SARDU-Net-v1     | 1.45 +-0.14        | 1.72  +-0.15        |      4.73 +- 0.57           |  5.15 +- 0.63           |  6.10 +- 0.79           |  21.0 +- 6.07           |  19.8 +- 9.26           |  22.8 +- 6.57           |
| SARDU-Net-v2     |  0.88 +- 0.10           |  0.89 +- 0.01           |  1.36 +- 0.14           |  1.66 +- 0.10           |  1.95 +- 0.12           |  2.27 +- 0.20           |  3.01 +- 0.45           |  4.41 +- 1.39           |
| SARDU-Net-v2-BOF |  0.83 +- 0.10           |  0.86 +- 0.10           |  1.30 +- 0.12           |  1.67 +- 0.12           |  1.86 +- 0.18           |  2.15 +- 0.23           |  2.61 +- 0.24           |  3.74 +- 0.66           |
| SARDU-Net-v2-NAS |  0.82 +- 0.13           |  0.99 +- 0.12           |  1.34 +- 0.26           |  1.76 +- 0.24           |  2.23 +- 0.22           |  6.00 +- 7.14           |  2.82 +- 0.41           |  4.27 +- 1.66           |
|                  |                            |                            |
|     PROSUB       | **0.49** +- 0.07 |  **0.61** +- 0.11  |  **0.89** +- 0.11  |  **1.35** +- 0.11  |  **1.53** +- 0.05  |  **1.87** +- 0.19  |  **2.50** +- 0.40  |  **3.48** +- 0.55  |


## Training and Evaluation for Results

To replicate the results:

### SARDU-Net (Baselines)

For SARDU-Net-v1, SARDU-Net-v2, SARDU-Net-v2-BOF, use the [original SARDU-Net code](https://github.com/fragrussu/sardunet/tree/67747b09956d3b5d78d504792831aa1857b018ed).

### SARDU-Net-v2-NAS (Baseline)

```bash
# Final result is the average across the splits i.e. across SPLIT_NO

# Replace with your variables used in main.py:
data_fil=<ADD>
out_base=<ADD>

# Choose one of the two below:
#num_units_init='1063 781 781 1063' # for M=500
num_units_init='417 333 781 1063' # for M<500

T_values='4 8 12 16 20 24 28 32 36'
M_values='1344 500 250 100 50 40 30 20 10'

for SPLIT_NO in {0..4}
do
    case $SPLIT_NO in
        0) data_train_subjs="0012 0013 0014"; data_val_subjs="0015"; data_test_subjs="0011"; split_name="0011out";;
        1) data_train_subjs="0013 0014 0015"; data_val_subjs="0011"; data_test_subjs="0012"; split_name="0012out";;
        2) data_train_subjs="0014 0015 0011"; data_val_subjs="0012"; data_test_subjs="0013"; split_name="0013out";;
        3) data_train_subjs="0015 0011 0012"; data_val_subjs="0013"; data_test_subjs="0014"; split_name="0014out";;
        4) data_train_subjs="0011 0012 0013"; data_val_subjs="0014"; data_test_subjs="0015"; split_name="0015out";;
    esac

    python main.py \
        --data_fil $data_fil \
        --data_train_subjs $data_train_subjs \
        --data_val_subjs $data_val_subjs \
        --data_test_subjs $data_test_subjs \
        --proj_name SARDUNet-v2-NAS_run \
        --out_base $out_base \
        --seed 0 \
        --T_values $T_values \
        --M_values $M_values \
        --noepoch 200 \
        --epochs_decay 20 \
        --tuner greedy \
        --num_layers_nas 1 2 3 \
        --network_name sardunet-nas \
        --dropout_nas 0.2 0.0 0.1 0.3 0.4 \
        --data_normalization original \
        --num_units_nas {128..2048..128} \
        --num_units_init $num_units_init
done
```

### PROSUB (Ours)

```bash
# Final result is the average across the splits i.e. across SPLIT_NO

# Replace with your variables used in main.py:
data_fil=<ADD>
out_base=<ADD>

T_values='4 8 12 16 20 24 28 32 36'
M_values='1344 500 250 100 50 40 30 20 10'
num_units_init='1063 781 781 1063'

# Cross-validation splits
for SPLIT_NO in {0..4}
do
    case $SPLIT_NO in
        0) data_train_subjs="0012 0013 0014"; data_val_subjs="0015"; data_test_subjs="0011"; split_name="0011out";;
        1) data_train_subjs="0013 0014 0015"; data_val_subjs="0011"; data_test_subjs="0012"; split_name="0012out";;
        2) data_train_subjs="0014 0015 0011"; data_val_subjs="0012"; data_test_subjs="0013"; split_name="0013out";;
        3) data_train_subjs="0015 0011 0012"; data_val_subjs="0013"; data_test_subjs="0014"; split_name="0014out";;
        4) data_train_subjs="0011 0012 0013"; data_val_subjs="0014"; data_test_subjs="0015"; split_name="0015out";;
    esac

    python main.py \
        --data_fil $data_fil \
        --data_train_subjs $data_train_subjs \
        --data_val_subjs $data_val_subjs \
        --data_test_subjs $data_test_subjs \
        --proj_name PROSUB_run \
        --out_base $out_base \
        --seed 0 \
        --T_values $T_values \
        --M_values $M_values \
        --noepoch 200 \
        --epochs_decay 20 \
        --tuner greedy \
        --num_layers_nas 1 2 3 \
        --network_name prosub \
        --dropout_nas 0.0 \
        --data_normalization original-measurement \
        --num_units_nas {128..2048..128} \
        --num_units_init $num_units_init
done
```

## Contact and Feedback

stefano.blumberg.17@ucl.ac.uk
