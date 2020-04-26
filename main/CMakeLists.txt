set(MUJOCO_ROOT "${vpg_SOURCE_DIR}/extern/mujoco200")
message(STATUS "Adding Mujoco header " ${MUJOCO_ROOT})

add_executable(main main.cpp)
message(STATUS "Adding PyTorch Libraries " "${TORCH_LIBRARIES}")

target_include_directories(main 
    PRIVATE "${MUJOCO_ROOT}/include"
    PRIVATE "${vpg_SOURCE_DIR}/include"
)
target_link_libraries(main 
    PRIVATE GL
    PRIVATE "${MUJOCO_ROOT}/bin/libglew.so"
    PRIVATE "${MUJOCO_ROOT}/bin/libglfw.so.3"
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200.so"
    PRIVATE drl
    PRIVATE drlmodel
    PRIVATE "${TORCH_LIBRARIES}"
)
