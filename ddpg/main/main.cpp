#include <vector>
#include <iostream>
#include <tuple>
#include <chrono>
#include <cmath>

#include "mujoco.h"
#include "torch/torch.h"
#include "drl.h"
#include "drlmodel.h"


int main(int argc, const char *argv[])
{
    if(argc < 3) {
        std::cout << "Usage: ./main <path-to-Mujoco-key> <model-file> gpu" << std::endl;
        std::cout << "Type gpu at the end if you want to use CUDA." << std::endl;
        return 1;
    }    

    bool use_gpu = false;
    if(argc == 4) {
        std::string gpu_arg = argv[3];
        use_gpu = (gpu_arg == "gpu") ? true: false; 
    }

    torch::DeviceType device_type;
    if (torch::cuda::is_available() && use_gpu) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    
    at::set_num_threads(4);
    std::cout << "Number of threads: " << at::get_num_threads() << std::endl;

    mj_activate(argv[1]);
    char error[1000];
    mjModel* m = nullptr;
    mjData* d = nullptr;

    // load model from file and check for errors
    m = mj_loadXML(argv[2], nullptr, error, 1000);
    if( !m )
    {
       std::cout << error << std::endl;
       return 1;
    }

    d = mj_makeData(m);

    drlmodel::Cheetah cheetah{m, d}; 
    //Dimension for Action space and Observation space
    const int act_dim = cheetah.act_dim();
    const int obs_dim = cheetah.obs_dim();
    float act_max = cheetah.act_limit();
    std::cout << "Number of Observation: " << obs_dim << std::endl;
    std::cout << "Number of Action: " << act_dim << std::endl;

    //Hyperparameter
    auto steps_per_epoch = 4000;
    auto epoch = 100;
    float gamma = 0.99;
    auto replay_size = 1000000;
    float polyak = 0.995;
    float actor_lr = 1.0e-3;
    float q_lr = 1.0e-3;
    int64_t batch_size = 100;
    auto start_steps = 10000;
    auto update_after = 1000;
    auto update_every = 50;
    float act_noise = 0.1;
    auto num_test_episodes = 10;
    auto max_ep_len = 1000;
    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .layout(torch::kStrided);

    //define buffer
    drl::ReplayBuffer buf{obs_dim, act_dim, replay_size};

    //define logger
    drl::Logger logger{};
    
    //define actor-critic
    drl::ActorCritic ac{obs_dim, act_dim, 256, act_max};
    ac->to(device);
    drl::ActorCritic ac_targ{obs_dim, act_dim, 256, act_max};    
    ac_targ->to(device);

    //copy parameters from ac to ac_targ
    for (const auto& pair : ac->named_parameters()) {

        torch::NoGradGuard no_grad;
        ac_targ->named_parameters()[pair.key()].set_data(pair.value().clone());
        ac_targ->named_parameters()[pair.key()].set_requires_grad(false);
    }

    torch::optim::Adam pi_optimizer(ac->actor_->parameters(), 
                                    torch::optim::AdamOptions(actor_lr));   
 
    torch::optim::Adam q_optimizer(ac->qnet_->parameters(), 
                                   torch::optim::AdamOptions(q_lr));

    auto total_steps = steps_per_epoch * epoch;
    float ep_ret = 0.0;
    auto ep_len = 0;
    
    auto obs = cheetah.reset();

    //Main training loop
    for (int t=0; t != total_steps; ++t) {
        //auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor action;        

        if (t > start_steps) {

            action = drl::get_action(obs, act_noise, act_dim, act_max, ac, options);
        } else {
 
            action = cheetah.sample();
        }       
            
        auto [next_obs, reward, d] = cheetah.step(action);
        ep_ret += reward.item<float>();
        ++ep_len;

        d = (ep_len == max_ep_len) ? false : d;
        
        //auto dtensor = d ? torch::ones({1}, torch::TensorOptions().dtype(torch::kInt32))[0] :
        //                   torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32))[0];

        buf.store(obs, action, reward, next_obs);

        obs = next_obs;

        if (d || (ep_len == max_ep_len)) {

            logger.dict.push_back(ep_ret); 
            obs = cheetah.reset();
            ep_ret = 0.0;
            ep_len = 0;
        }

        if ((t >= update_after) && ( (t % update_every) == 0)) {
        
            for(int i=0; i != update_every; ++i ) {
    
                drl::update(buf,
                            ac,
                            ac_targ,
                            batch_size,
                            gamma,
                            polyak,
                            pi_optimizer,
                            q_optimizer,
                            device);
            }
        } 
      
        if ( ((t+1) % steps_per_epoch) == 0) {

            int ep = floor((t+1)/ steps_per_epoch);
            std::cout << "EpRet = " << logger.mean() <<std::endl;
            
            auto [obs, obs2, act, rew] = buf.sample(batch_size);
            std::cout << "Reward: " << rew << std::endl;
            std::cout << "Action: " << act << std::endl;

        } 
        logger.reset();
    
        //auto end = std::chrono::high_resolution_clock::now();
	    //std::cout<< "Time(s) " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;     
    } 
    
    torch::save(ac, "model.pt");

    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    return 0;
}
