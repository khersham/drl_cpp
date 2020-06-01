#include <iostream>

#include "drl.h"

namespace drl {

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

torch::Tensor VPGBuffer::discount(torch::Tensor xarray, const float& gamma_){

    auto shape = xarray.sizes()[0];
    torch::Tensor yarray = torch::zeros(shape);
    for (int64_t i=0; i != shape; ++i){
        auto ind = torch::arange(shape, options);
        auto disc = torch::pow(gamma_, ind - i);
        auto disc_rtg = xarray.slice(-1, i, shape, 1)*disc.slice(-1, i, shape, 1);
        auto rtg_sum = disc_rtg.sum();
        yarray[i] = rtg_sum; 
    }
        
    return yarray;
}
   
void VPGBuffer::store(const torch::Tensor& obs, 
                      const torch::Tensor& act, 
                      const torch::Tensor& rew,
                      const torch::Tensor& val, 
                      const torch::Tensor& logp) {

    if (ptr < max_size) {
        obs_buf[ptr] = obs;
        act_buf[ptr] = act;
        rew_buf[ptr] = rew;
        val_buf[ptr] = val;
        logp_buf[ptr] = logp;
        ptr++;
    } else {
        std::cout << "Buffer is full" << std::endl;
    }
}

void VPGBuffer::finish_path(const torch::Tensor& last_val){
    //need to unsqueeze because zero-dim tensor cannot be concatenated
    auto rews = torch::cat({rew_buf.slice(-1, ptr_start_idx, ptr, 1), last_val.unsqueeze({0})},{0});        
    auto vals = torch::cat({val_buf.slice(-1, ptr_start_idx, ptr, 1), last_val.unsqueeze({0})},{0});
    auto deltas = rews.slice(-1, 0, -1, 1) +
                  gamma*vals.slice(-1, 1, vals.sizes()[0],1) -
                  vals.slice(-1, 0, -1, 1);
 
    adv_buf.slice(-1, ptr_start_idx, ptr, 1) = discount(deltas, gamma * lam);
    ret_buf.slice(-1, ptr_start_idx, ptr, 1) = discount(rews, gamma).slice(-1, 0, -1, 1);
    ptr_start_idx = ptr;
}

std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> VPGBuffer::get() {

    assert(ptr == max_size);
    ptr = 0;
    ptr_start_idx = 0;
    auto adv_mean = adv_buf.mean();
    auto adv_std = adv_buf.std(false);
    adv_buf = (adv_buf - adv_mean) / adv_std;
    return {obs_buf, act_buf, ret_buf, adv_buf, logp_buf};
}

torch::Tensor compute_loss_v(const torch::Tensor& obs, 
                             const torch::Tensor& ret, 
                             ActorCritic& ac) {

    auto v = (ac->critic_)->forward(obs);
    auto loss_v = ((v - ret).pow(2)).mean();
    //v loss
    return loss_v;
}

torch::Tensor compute_loss_pi(const torch::Tensor& obs, 
                              const torch::Tensor& act, 
                              const torch::Tensor& adv, 
                              ActorCritic& ac){

    auto [pi, logp] = (ac->actor_)->forward(obs, act);
    auto loss_pi = -(logp * adv).mean();
    //Policy loss
    return loss_pi;
}

void update(VPGBuffer& buf, ActorCritic& ac, 
            torch::optim::Adam& pi_optimizer, 
            torch::optim::Adam& v_optimizer, 
            const int& critic_iter, 
            const torch::Device& device) {

    auto [obs, act, ret, adv, logp] = buf.get();
    obs = obs.to(device);
    act = act.to(device);
    
    pi_optimizer.zero_grad();
    auto pi_loss = compute_loss_pi(obs, act, adv, ac);

    pi_loss.backward();
    pi_optimizer.step();
        
    torch::Tensor v_loss; 
    for (int i=0; i != critic_iter; ++i){
        v_optimizer.zero_grad();
        v_loss = compute_loss_v(obs, ret, ac);
        v_loss.backward();
        v_optimizer.step();
    }
}

float Logger::mean(){
    float sum = std::accumulate(dict.begin(), dict.end(), 0.0);
    float mean = sum / dict.size();
    return mean;
}

void Logger::reset(){
    dict.clear();
}

}
