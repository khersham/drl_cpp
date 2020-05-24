#include <cmath>
#include <iostream>
#include <algorithm>
#include "drl.h"

namespace torch{
//PyTorch does not generate this constructor by default
torch::Tensor normal(const torch::Tensor &mean, 
                     const torch::Tensor &std, 
                     torch::Generator *generator = nullptr) {
  return at::normal(mean, std, generator);  
}
}

namespace drl {


ReplayBuffer::ReplayBuffer(const int64_t obs_dim_, 
                           const int64_t act_dim_, 
                           const int64_t max_size_):
    max_size{max_size_} {

    obs_buf = torch::zeros({max_size, obs_dim_}, options);
    obs2_buf = torch::zeros({max_size, obs_dim_}, options);
    act_buf = torch::zeros({max_size, act_dim_}, options); 
    rew_buf = torch::zeros(max_size, options); 
    //done_buf = torch::zeros(max_size, torch::TensorOptions().dtype(torch::kInt32) ); 
}

//skip isdone matrix
void ReplayBuffer::store(const torch::Tensor& obs, 
                         const torch::Tensor& act, 
                         const torch::Tensor& rew,
                         const torch::Tensor& next_obs) {

    obs_buf[ptr] = obs;
    obs2_buf[ptr] = next_obs;
    act_buf[ptr] = act;
    rew_buf[ptr] = rew;
    //done_buf[ptr] = done;
    ptr = (ptr + 1) % max_size;
    size = std::min(size + 1, max_size);
}


std::tuple<torch::Tensor, 
           torch::Tensor, 
           torch::Tensor, 
           torch::Tensor> ReplayBuffer::sample(const int64_t batch_size=32) {

    torch::Tensor idx = torch::randint(0, size, {batch_size}, 
                                       torch::TensorOptions().dtype(torch::kInt64));
    return {obs_buf.index(idx), 
            obs2_buf.index(idx), 
            act_buf.index(idx), 
            rew_buf.index(idx)};
}

torch::Tensor compute_loss_q(const torch::Tensor& obs, 
                             const torch::Tensor& act, 
                             const torch::Tensor& obs2, 
                             const torch::Tensor& rew, 
                             const float gamma,
                             ActorCritic& ac,
                             ActorCritic& ac_targ) {

    auto q = (ac->qnet_)->forward(obs, act);
    torch::Tensor q_pi_targ;    
    torch::Tensor backup;

    {
        torch::NoGradGuard no_grad; 
        auto act_infer = ac_targ->actor_->forward(obs2);
        q_pi_targ = ac_targ->qnet_->forward(obs2, act_infer);
        backup = rew + gamma* q_pi_targ;
    }

    auto loss_q = ((q - backup).pow(2)).mean();
    //Policy loss
    return loss_q;
}

torch::Tensor compute_loss_pi(const torch::Tensor& obs,
                              ActorCritic& ac) { 

    auto act_infer = ac->actor_->forward(obs);
    auto loss_pi = -(ac->qnet_->forward(obs, act_infer)).mean();
    //Policy loss
    return loss_pi;
}

void update(ReplayBuffer& buf, 
            ActorCritic& ac, 
            ActorCritic& ac_targ,
            const int64_t batch_size,
            const float gamma, 
            const float polyak,
            torch::optim::Adam& pi_optimizer, 
            torch::optim::Adam& q_optimizer, 
            const torch::Device& device) {

    auto [obs, obs2, act, rew] = buf.sample(batch_size);
    obs = obs.to(device);
    obs2 = obs2.to(device);
    act = act.to(device);
    rew = rew.to(device);
    //isdone = isdone.to(device);
    
    q_optimizer.zero_grad();
    torch::Tensor q_loss = compute_loss_q(obs, act, obs2, rew, gamma, ac, ac_targ);
    q_loss.backward();
    q_optimizer.step();

    
    for (const auto& pair : ac->qnet_->named_parameters()) {
        ac->qnet_->named_parameters()[pair.key()].set_requires_grad(false);
    }

    pi_optimizer.zero_grad();
    auto pi_loss = compute_loss_pi(obs, ac);
    pi_loss.backward();
    pi_optimizer.step();
        
    for (const auto& pair : ac->qnet_->named_parameters()) {
        ac->qnet_->named_parameters()[pair.key()].set_requires_grad(true);
    }

    {
        torch::NoGradGuard no_grad; 
        for (const auto& pair : ac->named_parameters()) {
            auto phi_old = ac_targ->named_parameters()[pair.key()].clone(); 
            auto phi_new = ac->named_parameters()[pair.key()].clone(); 
            auto phi_targ_new = polyak*phi_old + (1-polyak)*phi_new;            

            ac_targ->named_parameters()[pair.key()].set_data(phi_targ_new);
        }
    }    

}

torch::Tensor get_action(const torch::Tensor& obs, 
                         const float act_noise,
                         const int act_dim,
                         const float act_limit,
                         ActorCritic& ac,
                         const torch::TensorOptions& options) {

    auto act = ac->forward(obs);
    act = act + act_noise*torch::randn({act_dim}, options);
    act = torch::clamp(act, -act_limit, act_limit);

    return act;
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
