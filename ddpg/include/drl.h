#ifndef DRL_H
#define DRL_H

#include <vector>
#include <string>
#include <tuple>

#include "torch/torch.h"
#include "mujoco.h"

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
    
    torch::Tensor forward(torch::Tensor x, bool isclip) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = isclip ? torch::tanh(fc3->forward(x)) : fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(MLP);


class MLPGaussianActorImpl: public torch::nn::Module {
private:
    float act_limit;
    MLP pi;
    
public:
    MLPGaussianActorImpl(const int64_t& obs_dim, 
                         const int64_t& act_dim, 
                         const int64_t& hidden,
                         const float& act_limit_):
    act_limit{act_limit_},
    pi{obs_dim, act_dim, hidden} {

        register_module("actor", pi);
    }

    torch::Tensor forward(const torch::Tensor& obs) {
        return act_limit*(pi->forward(obs, true));
    };
    
};
TORCH_MODULE(MLPGaussianActor);


class MLPQImpl: public torch::nn::Module {
private:
    MLP qnet;
        
public:
    MLPQImpl(const int64_t& obs_dim, 
             const int64_t& act_dim,
             const int64_t& hidden):
    qnet{obs_dim + act_dim, 1, hidden} {

        register_module("qnet", qnet);
    }

    //Argument: obs
    torch::Tensor forward(const torch::Tensor& obs,
                          const torch::Tensor& act) {
        auto q = qnet->forward(torch::cat({obs, act},{-1}), false);
        return q.to(torch::kCPU).squeeze({-1});
    };
    
};
TORCH_MODULE(MLPQ);


class ActorCriticImpl: public torch::nn::Module {
public:
    float act_limit;
    MLPGaussianActor actor_;
    MLPQ qnet_;

    ActorCriticImpl(const int64_t& obs_dim, 
                    const int64_t& act_dim, 
                    const int64_t& hidden,
                    const float& act_limit_):
    act_limit{act_limit_},
    actor_{obs_dim, act_dim, hidden, act_limit_},
    qnet_{obs_dim, act_dim, hidden} {

        register_module("ac-actor", actor_);
        register_module("ac-qnet", qnet_);
    };

    torch::Tensor forward(const torch::Tensor& obs) {
        torch::NoGradGuard no_grad;
        //No Grad now
        auto policy = actor_->forward(obs);
        //v is zero dim tensor
        return policy.to(torch::kCPU);
 
    };

};
TORCH_MODULE(ActorCritic);


class ReplayBuffer {
private:
    int64_t ptr = 0;
    int64_t size = 0;
    int64_t max_size;
    torch::Tensor obs_buf;
    torch::Tensor obs2_buf;
    torch::Tensor act_buf;
    torch::Tensor rew_buf;
    torch::Tensor done_buf;
    const torch::TensorOptions options = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .layout(torch::kStrided);
public:
    ReplayBuffer(const int64_t, 
                 const int64_t, 
                 const int64_t);

    void store(const torch::Tensor&, 
               const torch::Tensor&, 
               const torch::Tensor&, 
               const torch::Tensor&);

    std::tuple<torch::Tensor, 
               torch::Tensor, 
               torch::Tensor, 
               torch::Tensor> sample(const int64_t);
};


torch::Tensor compute_loss_q(const torch::Tensor&, 
                             const torch::Tensor&, 
                             const torch::Tensor&, 
                             const torch::Tensor&,
                             const float,
                             ActorCritic&, 
                             ActorCritic&);

torch::Tensor compute_loss_pi(const torch::Tensor&, ActorCritic&);

void update(ReplayBuffer&, 
            ActorCritic&, 
            ActorCritic&,
            const int64_t,
            const float, 
            const float,
            torch::optim::Adam&, 
            torch::optim::Adam&, 
            const torch::Device&);

torch::Tensor get_action(const torch::Tensor&, 
                         const float,
                         const int,
                         const float,
                         ActorCritic&,
                         const torch::TensorOptions&);


class Logger{
public:
    std::vector<float> dict;
    Logger() {};
    
    float mean();
    void reset();
};

}

#endif
