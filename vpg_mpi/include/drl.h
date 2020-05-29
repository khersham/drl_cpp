#ifndef DRL_H
#define DRL_H

#include <vector>
#include <string>
#include <tuple>

#include "torch/torch.h"
#include "mujoco.h"

namespace torch{
//PyTorch does not generate this constructor by default
torch::Tensor normal(const torch::Tensor &mean, 
                     const torch::Tensor &std, 
                     torch::Generator *generator = nullptr) {
  return at::normal(mean, std, generator);  
}
}

namespace drl {

class MLPImpl: public torch::nn::Module{
private:
    torch::nn::Linear fc1, fc2, fc3;
public: 
    //Constructor
    MLPImpl(const int64_t& obs_dim, const int64_t& act_dim, const int64_t& hidden):
    fc1{torch::nn::Linear(obs_dim, hidden)},
    fc2{torch::nn::Linear(hidden, hidden)},
    fc3{torch::nn::Linear(hidden, act_dim)}
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(MLP);


class Gaussian {
private:
    torch::Tensor mean_;
    torch::Tensor scale_; 
public:
    //Constructor
    Gaussian(const torch::Tensor& mean, const torch::Tensor& scale): 
    mean_{mean}, scale_{scale} {};
    
    torch::Tensor log_prob(const torch::Tensor& value) {
        auto stdev = torch::exp(scale_);
        auto var = stdev.pow(2);
        auto logscale_ = stdev.log();
        auto nom = -(mean_ - value).pow(2);
        auto constant = log(sqrt(2 * M_PI) );
        return nom / (2 * var) -logscale_ - constant;
    };
    
    torch::Tensor sample() {
        torch::NoGradGuard no_grad;
        return torch::normal(mean_, torch::exp(scale_));
    };           
};


class MLPGaussianActorImpl: public torch::nn::Module {
private:
    torch::Tensor log_std_;
    MLP mu_net_;
    
public:
    MLPGaussianActorImpl(const int64_t& obs_dim, const int64_t& act_dim, const int64_t& hidden):
    log_std_{-0.5*torch::ones({act_dim})}, 
    mu_net_{obs_dim, act_dim, hidden} {
        register_module("actor", mu_net_);
        register_parameter("log_std", log_std_);
    }
    
    Gaussian dist(const torch::Tensor& obs) {
        Gaussian dist_ {mu_net_->forward(obs), log_std_};
        return dist_;
    };
    
    std::tuple<Gaussian, torch::Tensor> forward(const torch::Tensor& obs, const torch::Tensor& act) {
        auto gauss = dist(obs);
        auto logprob = (gauss.log_prob(act)).sum(-1);
        return {gauss, logprob.to(torch::kCPU)};
    };
    
};
TORCH_MODULE(MLPGaussianActor);


class MLPCriticImpl: public torch::nn::Module {
private:
    MLP critic_net_;
        
public:
    MLPCriticImpl(const int64_t& obs_dim, const int64_t& hidden):
    critic_net_{obs_dim, 1, hidden} {
        register_module("critic", critic_net_);
    }
    
    //Argument: obs
    torch::Tensor forward(const torch::Tensor& obs) {
        return (critic_net_->forward(obs)).to(torch::kCPU).squeeze({-1});
    };
    
};
TORCH_MODULE(MLPCritic);


class ActorCriticImpl: public torch::nn::Module{
public:
    MLPGaussianActor actor_;
    MLPCritic critic_;
    ActorCriticImpl(const int64_t& obs_dim, const int64_t& act_dim, const int64_t& hidden):
    actor_{obs_dim, act_dim, hidden},
    critic_{obs_dim, hidden} {
        register_module("ac-actor", actor_);
        register_module("ac-critic", critic_);
    };

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs){    
        torch::NoGradGuard no_grad;
        //No Grad now
        auto policy = actor_->dist(obs);
        auto action = policy.sample();
        auto logp_a = (policy.log_prob(action)).sum();
        auto v_value = critic_->forward(obs);
        //v is zero dim tensor
        return {action.to(torch::kCPU), v_value.to(torch::kCPU), logp_a.to(torch::kCPU)};
    };

};
TORCH_MODULE(ActorCritic);


class VPGBuffer {
private:
    const float gamma;
    const float lam;
    int64_t ptr = 0;
    int64_t ptr_start_idx = 0;
    int64_t max_size;
    torch::Tensor obs_buf;
    torch::Tensor act_buf;
    torch::Tensor adv_buf;
    torch::Tensor rew_buf;
    torch::Tensor ret_buf;
    torch::Tensor val_buf;
    torch::Tensor logp_buf;
    const torch::TensorOptions options = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .layout(torch::kStrided);
public:
    VPGBuffer(const int64_t&, const int64_t&, const int64_t&, const float&, const float&);
    torch::Tensor discount(torch::Tensor, const float&);
    void store(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
               const torch::Tensor&, const torch::Tensor&);
    void finish_path(const torch::Tensor&);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get();
};


torch::Tensor compute_loss_v(const torch::Tensor&, const torch::Tensor&, ActorCritic&);
torch::Tensor compute_loss_pi(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, ActorCritic&);
void update(VPGBuffer&, ActorCritic&, torch::optim::Adam&, 
            torch::optim::Adam&, const int&, const torch::Device&);

class Logger{
public:
    std::vector<float> dict;
    Logger() {};
    
    float mean();
    void reset();
};

}

#endif
