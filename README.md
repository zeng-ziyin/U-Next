# Small but Mighty: Enhancing 3D Point Clouds Semantic Segmentation with U-Next Framework

### (1) Setup
This code has been tested with Python 3.8, Tensorflow 2.4, CUDA 11.0 and cuDNN 8.0.5 on Ubuntu 20.04.

- Setup python environment
```
conda create -n unext python=3.8
source activate unext
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Data Prepare
```
cd utils/
python data_prepare_$data_you_want_use$.py
```
If you want to use Toronto3D, run this before the previous command line
```
python process_toronto3d.py
```

### (3) Train & Test
If you don't want to use MDS, run this
```
python main_$data_you_want_use$.py --mode train & test
```

If you want to use MDS, run this
```
python main_$data_you_want_use$_ds.py --mode train & test
```
