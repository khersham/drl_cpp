#include <vector>
#include <iostream>
#include <tuple>
#include <chrono>

#include "mpi.h"
#include "mujoco.h"
#include "torch/torch.h"
#include "drl.h"
#include "drlmodel.h"

int main(int argc, char** argv)
{
    if(argc < 3) {
        std::cout << "Usage: ./main <path-to-Mujoco-key> <model-file> gpu" << std::endl;
        std::cout << "Type gpu at the end if you want to use CUDA." << std::endl;
        return 1;
    }    

    //Initialze MPI
    MPI_Init(&argc, &argv);
    int my_rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    std::cout << "Starting MPI process with rank: " << my_rank << std::endl;
    
    //Use CUDA if intended, for DRL CUDA is slower than CPU
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
    
    //Set proper number of threads per MPI process
    auto num_threads = at::get_num_threads() / n_ranks;
    at::set_num_threads(num_threads);
    std::cout << "Number of threads: " << at::get_num_threads() << std::endl;

    //Initialize Mujoco model and data
    mj_activate(argv[1]);
    char error[1000];
    mjModel* m = nullptr;
    mjData* d = nullptr;

    //load model from file and check for errors
    m = mj_loadXML(argv[2], nullptr, error, 1000);
    if( !m )
    {
       std::cout << error << std::endl;
       return 1;
    }

    d = mj_makeData(m);
    
    //initialize Half-Cheetah model
    drlmodel::Cheetah cheetah{m, d}; 

    //Dimension for Action space and Observation space
    const int act_dim = cheetah.act_dim();
    const int obs_dim = cheetah.obs_dim();
    std::cout << "Number of Observation: " << obs_dim << std::endl;
    std::cout << "Number of Action: " << act_dim << std::endl;

    //Hyperparameter
    auto steps_per_epoch = 4000;
    auto epoch = 150;
    float gamma = 0.99;
    float actor_lr = 3.0e-4;
    float critic_lr = 3.0e-3;
    auto critic_iter = 80;
    float lam = 0.97;
    auto max_ep_len = 1000;
    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .layout(torch::kStrided);

    //define buffer
    drl::VPGBuffer buf{obs_dim, act_dim, steps_per_epoch, gamma, lam};

    //define logger
    drl::Logger logger{};
    
    //define actor-critic
    drl::ActorCritic ac{obs_dim, act_dim, 64};
    ac->to(device);   

    //sync parameters, broadcast root NN parameters to workers
    {
        torch::NoGradGuard no_grad;
        MPI_Barrier(MPI_COMM_WORLD);
        for (const auto& pair : ac->named_parameters()) {
            auto count = pair.value().numel();
            float* temp_arr = pair.value().data_ptr<float>();
        
            MPI_Bcast(temp_arr, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
        } 
    }

 
    torch::optim::Adam pi_optimizer(
    ac->actor_->parameters(), torch::optim::AdamOptions(actor_lr));   
 
    torch::optim::Adam v_optimizer(
    ac->critic_->parameters(), torch::optim::AdamOptions(critic_lr));
  
    for (const auto& pair : ac->named_parameters()) {
        std::cout << pair.key() << " : " << pair.value().sizes() << std::endl;
    }

    float ep_ret = 0.0;
    auto ep_len = 0;
    
    auto obs = cheetah.reset();

    //Main training loop
    for (int i=0; i != epoch; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if (my_rank == 0) {
            std::cout << "Epoch: " << i << std::endl;
        }
        for (int t=0; t!= steps_per_epoch; ++t) {
            auto [action, v, logp] = ac->forward(obs.to(device));
            
            auto [next_obs, reward, d] = cheetah.step(action);

            //Add up the reward
            ep_ret += reward.item<float>();
            ++ep_len;
            buf.store(obs, action, reward, v, logp);

            obs = next_obs;

            auto timeout = (ep_len == max_ep_len);
            auto terminal = (timeout || d);
            auto epoch_ended = (t == (steps_per_epoch - 1));
            if (terminal || epoch_ended) {
                if (epoch_ended && !terminal){
                    std::cout << "Trajectory cut off" << std::endl;
                }
                if (timeout || epoch_ended) {
                    auto [dummy1, v, dummy2] = ac->forward(obs.to(device));
                    buf.finish_path(v);
                } else {
                    buf.finish_path(torch::zeros(1, options).squeeze({-1}) );
                }

                if (terminal) {
                    logger.dict.push_back(ep_ret);
                }

                obs = cheetah.reset();
                ep_ret = 0.0;
                ep_len = 0;
            }
        }

        drl::update(buf, ac, pi_optimizer, v_optimizer, critic_iter, device);
        
        auto logmean = logger.mean(); 
        if (my_rank == 0) {
            std::cout << "EpRet = " << logmean <<std::endl;
            auto end = std::chrono::high_resolution_clock::now();
	        std::cout<< "Time(s) " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;     
        }

        logger.reset();
    
    } 
    
    torch::save(ac, "model.pt");
    
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();
    
    MPI_Finalize();
    
    return 0;
}
