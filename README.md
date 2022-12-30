
# Progressive Subsampling for Oversampled Data - Application to Quantitative MRI

## Citation and Contact

The paper is accepted in: Medical Image Computing and Computer Assisted Intervention (MICCAI) 2022

Please consider citing:<br/>
```
@article{blumberg2022,
    author={Stefano B. Blumberg and Hongxiang Lin and Francesco Grussu and Yukun Zhou and Matteo Figini and Daniel C. Alexander},
    title={Progressive Subsampling for Oversampled Data - Application to Quantitative {MRI}},
    journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
    year = {2022}
}
```
If you have any comments or suggestions, contact:
stefano.blumberg.17@ucl.ac.uk

## Installation and Requirements
See [requirements file](./requirements/README_requirements.md)

## Data Preparation

To use MUDI data from paper:<br/>
    - Register and Download [link](https://www.developingbrain.co.uk/data)<br/>
    - Create a dictionary of numpy arrays, with keys: '0011' as np.array no_samplesx1344 data from subject 11 et.c

Create Dummy Data:<br/>
```python
import numpy as np; import pickle
no_samples=1000
no_measurements=1344
data_keys = ('0011','0012','0013','0014','0015')
data = {subj: np.random.rand(no_samples,no_measurements) for subj in data_keys}
with open('data.pkl','wb') as f:  pickle.dump(data, f)
```


## Results in Paper
To replicate the results in table 1 please check and run:<br/>
    - SARDU-Net-v1, SARDU-Net-v2, SARDU-Net-v2-BOF (Baselines) -- [code](https://github.com/fragrussu/sardunet/tree/67747b09956d3b5d78d504792831aa1857b018ed)<br/>
    - SARDU-Net-v2-NAS (Baseline) -- [run](./scripts/run_SARDU-Net-v2-NAS.bash)<br/>
    - PROSUB (Ours) -- [run](./scripts/run_PROSUB.bash)
