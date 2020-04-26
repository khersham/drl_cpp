# Purpose
This repo contains C++ code to train a vanilla policy gradient on half-cheetah model in Mujoco. 
The example code utilizes the latest PyTorch (1.5.0) and Mujoco 2.0. 
The example shown here is heavily inspired from the Python implementation of [Spinningup](https://spinningup.openai.com/en/latest/). I am very grateful that OpenAI has open sourced such excellent guide for newcomer in Deep RL. 

#How to install
First download the Mujoco 2.0 and Libtorch 1.5.0 to the extern folder. 
```bash
mkdir extern
pushd extern
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip && mv mujoco200_linux mujoco200
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.0.zip
unzip libtorch-shared-with-deps-1.5.0.zip
popd
```
Then create a build directory and call CMake.
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${PWD}/../extern/libtorch" ..
make
```
You can add `-DCUDA_TOOLKIT_ROOT_DIR=<PATH-TO-CUDA>` if you have installed CUDA in non standard folder.

#How to run
After the build has finished, call the main binary:
```bash
./main/main <MUJOCO-KEY-PATH> <PATH-TO-MUJOCO-XML> gpu
```
You need to provide your own Mujoco license key and the path to half-cheetah XML model file. 
The last argument is optional, use `gpu` if you want to train the model with CUDA. For our simple
model the usage of GPU is actually slower.
