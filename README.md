# Nvidia-Jetson-AI-Pylon-Demo
This is a sample that uses pylon SDK on an Nvidia Jetson platfrom to apply an AI in transforming 2D images to a 3D depthmap with python

The 2D to 3D transfomations AI data set comes from  Intelligent Systems Lab Org
Which can be found at the link below 
https://github.com/isl-org/MiDaS

Dependincies for using this sample will include
OpenCV
Pytorch
numpy
pypylon
matplotlib

To apply a this to a Jetson platform you will need to configure Pytorch to work with Cuda
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

Make sure your version of pytorch is supported by your jetpack version

As of the making of this README (2023) currently deplyoed JetPack version is 5.1

When checking pytorch, it requires
https://pytorch.org/blog/deprecation-cuda-python-support/

PyTorch Version 2.0 	
Python >=3.8, <=3.11	
CUDA 11.7 
CUDNN 8.5.0.96

So we need to check your version of each of these which can be done with following commands

pip3 show torch
python3 --version
nvcc --version
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

If you need to update your python version you can try

sudo apt update && sudo apt upgrade
sudo apt install python3.8

Then you can check if your version as updated
python3 --version

If yur python version is still not pointing to 3.8, you can use the comands below to update your python3 default

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
update-alternatives --config python3

If you build your Jetson with the standard Cuda install package you may have a miss match in your version number
You will either need to reduce your PyTorch version to match this or update your cuda package

In this example you need to you need to have 11.7, whcih can be found in the link below
https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_arm64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_arm64.debsudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/sudo apt-get updatesudo apt-get -y install cuda

Now that we have the correc version of python and cuda we can install pytorch

https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

sudo apt-get -y update; 
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;

export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

export TORCH_INSTALL=path/to/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL

If everything is setup correctly at this point you should be able to open a python3 sheel and run

import torch
torch.cuda.is_available()

And this should define if pytorch can see the cuda setup
