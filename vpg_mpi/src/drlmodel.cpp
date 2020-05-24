#include <random>
#include <algorithm>

#include "drlmodel.h"

namespace drlmodel{

//function to get current qpos and qvel of simulator
torch::Tensor current_state(const mjModel* m, const mjData* d){
    torch::TensorOptions options = torch::TensorOptions()
                                    .dtype(torch::kFloat64)
                                    .layout(torch::kStrided);
    
    torch::TensorOptions options_float = torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .layout(torch::kStrided);

    torch::Tensor q = torch::from_blob(d->qpos, {m->nq}, options).clone();
    torch::Tensor v = torch::from_blob(d->qvel, {m->nv}, options).clone();
    torch::Tensor obs = torch::cat({q, v}, {0}).to(options_float);
    
    return obs;
}


//Reset simulator
void reset_mujoco(const mjModel* m, mjData* d){
    mj_resetData(m, d);
    int dim = m->nq;   
 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis_uniform(-0.1, 0.1);
    std::normal_distribution<double> dis_normal(0.0,1.0);
    
    for (int i=0; i != dim; ++i) {
        d->qpos[i] = dis_uniform(gen);
        d->qvel[i] = dis_normal(gen)*0.1;
    }

}

//Simulator only needs to access mujoco, accept cpu tensor only
void simulator(const mjModel* m, mjData* d, const torch::Tensor& action, const int& frameskip){
    auto act_acc = action.accessor<float,1>();
    auto act_size = act_acc.size(0);
    std::vector<double> control(act_size);
    for(int i = 0; i < act_size; ++i) {
        control[i] = act_acc[i];
    } 
    d->ctrl = control.data();
    
    for (int i=0; i != frameskip; ++i){
        mj_step(m, d);
    }
}


Cheetah::Cheetah(const mjModel* m, mjData* d): 
model{m}, data{d} {}
    
torch::Tensor Cheetah::obs() {
    auto obs_ = current_state(model, data);
    auto sizes = obs_.sizes()[0]; 
    obs_ = obs_.slice(0, 1, sizes);
    return obs_;
}

std::tuple<torch::Tensor, torch::Tensor, bool> Cheetah::step(const torch::Tensor& action) {
    auto xpos_init = data->qpos[0];
    auto action_cpu = action.clone().to(torch::kCPU);
    simulator(model, data, action_cpu, 5);
    auto xpos_final = data->qpos[0];
    auto reward_ctrl = -0.1*(action*action).sum().item<float>();
    auto reward_run = (xpos_final - xpos_init) / model->opt.timestep;
    auto reward = (reward_ctrl + reward_run)*torch::ones({1}, options)[0];
    auto obs_ = obs();
     
    return {obs_, reward, false};
}

torch::Tensor Cheetah::reset() {
    reset_mujoco(model, data);
    return obs();
}

//Use one dof for validation
int64_t Cheetah::obs_dim() const {
    return model->nq + model->nv - 1;
}

int64_t Cheetah::act_dim() const {
    return model->nu;
}
    
}
