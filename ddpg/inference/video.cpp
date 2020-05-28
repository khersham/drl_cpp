#include <string>
#include <iostream>
#include <chrono>

#include "mujoco.h"
#include "glfw3.h"
#include "torch/torch.h"
#include "drl.h"
#include "drlmodel.h"


// MuJoCo data structures
mjModel* m = nullptr;                  
mjData* d = nullptr;                  
mjvCamera cam;                     
mjvOption opt;                    
mjvScene scn;                    
mjrContext con;                 


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// main function
int main(int argc, const char** argv)
{
    // check command-line arguments
    if( argc!=4 )
    {
        std::cout << " USAGE: ./video <path-to-Mujoco-key> <mujoco-model-file> <pytorch-trained-model>" 
        << std::endl;
        return 1;
    }

    // activate software
    mj_activate(argv[1]);

    // load and compile model
    char error[1000] = "Could not load binary model";
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[2], 0);
    else
        m = mj_loadXML(argv[2], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 1000);                   
    mjr_makeContext(m, &con, mjFONTSCALE_100);   

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);

    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = 0;

    //load Cheetah model
    drlmodel::Cheetah cheetah{m, d};
    const int act_dim = cheetah.act_dim();
    const int obs_dim = cheetah.obs_dim();
    float act_max = cheetah.act_limit();
    auto obs = cheetah.reset();

    //Load ActorCritic and the trained model
    drl::ActorCritic ac{obs_dim, act_dim, 256, act_max};
    torch::load(ac, argv[3]);

    // run main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        mjtNum simstart = d->time;
        while( (d->time - simstart) < 1.0/ 20.0 ) {
            auto action = ac->forward(obs);
            auto [next_obs, reward, d] = cheetah.step(action);
            std::this_thread::sleep_for (std::chrono::milliseconds(30));
            obs = next_obs;
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    return 0;
}
