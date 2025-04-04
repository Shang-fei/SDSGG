## Installation


### Requirements:
- Python == 3.8
- PyTorch == 1.10.1
- torchvision == 0.11.2 (cuda:11.3)
- yacs
- matplotlib
- OpenCV
- h5py
- pillow
- pycocotools
- gcc version == 9.5.0

### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create -n SDSGG python==3.8
conda activate SDSGG
#Install cuda-11.3 in PATH
export CUDA_HOME=PATH/cuda_11.3

#install pytorch 1.10.1
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides
pip install pycocotools

cd SDSGG

# re-build it
python setup.py build develop


