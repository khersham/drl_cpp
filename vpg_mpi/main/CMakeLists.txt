set(MUJOCO_ROOT "${CMAKE_SOURCE_DIR}/extern/mujoco200")
message(STATUS "Adding Mujoco header " ${MUJOCO_ROOT})

add_executable(main main.cpp)
message(STATUS "Adding PyTorch Libraries " "${TORCH_LIBRARIES}")

target_include_directories(main 
    PRIVATE "${MUJOCO_ROOT}/include"
    PRIVATE "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(main 
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200nogl.so"
    PRIVATE drl
    PRIVATE drlmodel
    PRIVATE MPI::MPI_CXX
    PRIVATE "${TORCH_LIBRARIES}"
)
