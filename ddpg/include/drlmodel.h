#ifndef DRLMODEL_H
#define DRLMODEL_H

#include <tuple>

#include "torch/torch.h"
#include "mujoco.h"

namespace drlmodel {

//See drlmodel.cpp for implementation
void simulator(const mjModel*, mjData*, const torch::Tensor&, const int&);
void reset_mujoco(const mjModel*, mjData*);
torch::Tensor current_state(const mjModel*, const mjData*);

class Cheetah{
private:
    const mjModel* model;
    mjData* data;
    const float act_limit_ = 1.0;
    const torch::TensorOptions options = torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .layout(torch::kStrided);
public:
    Cheetah(const mjModel* m, mjData* d); 
    
    torch::Tensor obs();

    std::tuple<torch::Tensor, torch::Tensor, bool> step(const torch::Tensor& action);

    torch::Tensor reset();

    int64_t obs_dim() const;
    int64_t act_dim() const;
    float act_limit() const {return act_limit_;};
        
    torch::Tensor sample();
        
};

}
#endif
