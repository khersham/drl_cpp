# Purpose
This repo contains C++ code to train different Deep Reinforcement Learning (DRL) algorithms by 
utilizing the latest [PyTorch](https://pytorch.org/) (1.5.0) and [Mujoco](http://www.mujoco.org/index.html) 2.0. 
My intention is to deconstruct the whole mystery behind DRL, from simulator to implementation of 
Vanilla Policy Gradient. 
I will also explain the non-trivial steps in parallelizing the training steps with MPI. 
The implementation is written in C++, which is not that usual for Machine Learning and Deep Learning 
implementation. 
By giving up convenience in Python, we are forced to think carefully for many coding steps 
and we will uncover more ah-ha moment during the implementation. 
The code shown here is heavily inspired from the Python implementation of 
[Spinningup](https://spinningup.openai.com/en/latest/). 
I am very grateful that OpenAI has open sourced such excellent guide for newcomer in DRL. 

# Prerequisite
You should know a little bit of C++ and the syntax of PyTorch. 
The only new class the we will use a lot is `torch::Tensor`.
I do not assume that you are familiar with the C++ frontend of PyTorch,
and I will explain in more details if I think that certain steps are not clear.

# Simulator and Environment
We will implement our own Half-Cheetah simulator with Mujoco. 
You need a model in XML format for the simulator to run. 
For our Half-Cheetah implementation, we can download the XML from [OpenAI Gym](https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/half_cheetah.xml). 
The XML describe the positions of joins and actuators, and also some important metadata like timesteps.


Essentially Mujoco is very straight forward to operate:
1. You create the variable model `m`.
2. You load the model XML file `m = mj_loadXML(path-to-file);` to `m`.
3. You initialize the data file `d` with `d = mj_makeData(m);`.
4. To control the actuators, you pass a C-array of double to `d->ctrl`.
5. To simulate or step forward in time, you call the function `mj_step(m,d);`.
6. You can get the current observation from `d->qpos` (position) and `d->qvel` (velocity).
7. Finally you can reset the simulator with `mj_resetData(m,d);`.

To encapsulate the interaction with Mujoco simulator, we create a Cheetah class like in OpenAI Gym.
The most important method in Cheetah class is `step`:
```c++
std::tuple<torch::Tensor, torch::Tensor, bool> Cheetah::step(const torch::Tensor& action) {
    auto xpos_init = data->qpos[0];
    simulator(model, data, action, 5);
    auto xpos_final = data->qpos[0];
    auto reward_ctrl = -0.1*(action*action).sum().item<float>();
    auto reward_run = (xpos_final - xpos_init) / model->opt.timestep;
    auto reward = (reward_ctrl + reward_run)*torch::ones({1}, options)[0];
    auto obs_ = obs();
     
    return {obs_, reward, false};
}
```
where we use the first `qpos` observation as a measure of reward, i.e. how far the cheetah has traveled.
We also penalize the reward should the actuators try to generate volative movement. 
The function `simulator` passes the action array to Mujoco and moves the time 5 steps forward.
Please check the complete implementation in `src/drlmodel.cpp` file.

# Storage Buffer
Almost all DRL algorithms require a storage buffer to store the path of observations, actions taken and 
reward obtained. 
You need to tweak the storage for different DRL implementation. 
For our VPG implementation, first create a buffer to store observations, actions, rewards, advantage,
reward-to-go and the logarithm of policy.  
```c++
VPGBuffer::VPGBuffer(const int64_t& obs_dim_, 
                     const int64_t& act_dim_, 
                     const int64_t& max_size_,
                     const float& gamma_, 
                     const float& lam_):
    gamma{gamma_}, lam{lam_}, max_size{max_size_} {
    obs_buf = torch::zeros({max_size, obs_dim_}, options);
    act_buf = torch::zeros({max_size, act_dim_}, options);
    adv_buf = torch::zeros(max_size, options);
    rew_buf = torch::zeros(max_size, options);
    ret_buf = torch::zeros(max_size, options);
    val_buf = torch::zeros(max_size, options);
    logp_buf = torch::zeros(max_size, options);
}
```
Perhaps we should remind ourselves about the important equations in VPG:



## How to install for VPG
```bash
cd vpg
mkdir extern
pushd extern
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip && mv mujoco200_linux mujoco200
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.0+cpu.zip
unzip libtorch-shared-with-deps-1.5.0+cpu.zip
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
Make sure to download Libtorch with CUDA support in the step above.

## How to train
After the build has finished, call the main binary:
```bash
./main/main <MUJOCO-KEY-PATH> <PATH-TO-MUJOCO-XML> gpu
```
You need to provide your own Mujoco license key and the path to half-cheetah XML model file. 
The last argument is optional, use `gpu` if you want to train the model with CUDA. For our simple
model the usage of GPU is actually slower.

## How to watch the video
I have included a sample code for inference, which is heavily modified from the Mujoco's sample code.
After the training steps, run:
```bash
./inference/video <MUJOCO-KEY-PATH> <PATH-TO-MUJOCO-XML> model.pt 
```
Enjoy the show!
