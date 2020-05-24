#include <cmath>
#include <iostream>
#include <string>

#include "drl.h"
#include "mpi.h"

namespace torch{
//PyTorch does not generate this constructor by default
torch::Tensor normal(const torch::Tensor &mean, 
                     const torch::Tensor &std, 
                     torch::Generator *generator = nullptr) {
  return at::normal(mean, std, generator);  
}

}

namespace drl {

//log_prob function 
torch::Tensor log_prob(const torch::Tensor& mean, 
                       const torch::Tensor& scale, 
                       const torch::Tensor& value) {

    auto std = torch::exp(scale);
    auto var = std * std;
    auto logscale = std.log();
    auto nom = (mean - value)*(value - mean);
    auto constant = log(sqrt(2 * M_PI) );
    return nom / (2 * var) -logscale - constant;
}

torch::Tensor sample(const torch::Tensor& mean, 
                     const torch::Tensor& scale) {

    torch::NoGradGuard no_grad;
    return torch::normal(mean, torch::exp(scale));
}


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

torch::Tensor VPGBuffer::discount(torch::Tensor xarray, 
                                  const float& gamma_) {

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

void VPGBuffer::finish_path(const torch::Tensor& last_val) {

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

std::tuple<float, float> mpi_stat(const torch::Tensor& adv_buf) {

    int n_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    float global_sum{0.0}, global_sumsq{0.0};
    
    float local_sum = adv_buf.sum().item<float>();
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float adv_mean = global_sum / n_ranks / adv_buf.size(0);

    float local_sumsq = ((adv_buf - adv_mean).pow(2)).sum().item<float>();
    MPI_Allreduce(&local_sumsq, &global_sumsq, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float adv_std = sqrt(global_sumsq / n_ranks / adv_buf.size(0));

    return {adv_mean, adv_std};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VPGBuffer::get() {

    assert(ptr == max_size);
    ptr = 0;
    ptr_start_idx = 0;
    auto [adv_mean, adv_std] = mpi_stat(adv_buf);
    adv_buf = (adv_buf - adv_mean) / adv_std;
    return {obs_buf, act_buf, ret_buf, adv_buf, logp_buf};
}

torch::Tensor compute_loss_v(const torch::Tensor& obs, 
                             const torch::Tensor& ret, 
                             ActorCritic& ac) {

    auto v = (ac->critic_)->forward(obs);
    auto loss_v = ((v - ret).pow(2)).mean();
    //Policy loss
    return loss_v;
}

torch::Tensor compute_loss_pi(const torch::Tensor& obs, 
                              const torch::Tensor& act, 
                              const torch::Tensor& adv, 
                              ActorCritic& ac) {

    auto [pi, logp] = (ac->actor_)->forward(obs, act);
    auto loss_pi = -(logp * adv).mean();
    //Policy loss
    return loss_pi;
}

torch::Tensor mpi_avg(const torch::Tensor& gradtensor) {

    int n_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    auto count = gradtensor.numel();
    auto global_grad = torch::zeros_like(gradtensor);
    
    float* plocal = gradtensor.data_ptr<float>();
    float* pglobal = global_grad.data_ptr<float>();
    MPI_Allreduce(plocal, pglobal, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    auto grad_mean = global_grad / n_ranks;

    return grad_mean;
}


template<typename T>
void mpi_avg_grad(T& ac) {

    for (const auto& pair : ac->named_parameters()) {
        
        auto gradtensor = pair.value().grad().clone();
        pair.value().grad().set_data(mpi_avg(gradtensor));      
    
    }
}


void update(VPGBuffer& buf, 
            ActorCritic& ac, 
            torch::optim::Adam& pi_optimizer, 
            torch::optim::Adam& v_optimizer, 
            const int& critic_iter, 
            const torch::Device& device) 
{
    auto [obs, act, ret, adv, logp] = buf.get();
    obs = obs.to(device);
    act = act.to(device);
    
    pi_optimizer.zero_grad();
    auto pi_loss = compute_loss_pi(obs, act, adv, ac);

    pi_loss.backward();
    mpi_avg_grad(ac->actor_);
    pi_optimizer.step();
        
    torch::Tensor v_loss; 
    for (int i=0; i != critic_iter; ++i){
        v_optimizer.zero_grad();
        v_loss = compute_loss_v(obs, ret, ac);
        v_loss.backward();
        mpi_avg_grad(ac->critic_);
        v_optimizer.step();
    }
}

float Logger::mean(){
    int n_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    float global_sum{0.0};
    float local_sum = std::accumulate(dict.begin(), dict.end(), 0.0);
    
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float mean = global_sum / n_ranks / dict.size();

    return mean;
}

void Logger::reset(){
    dict.clear();
}

}
