set(MUJOCO_ROOT "${CMAKE_SOURCE_DIR}/extern/mujoco200")
message(STATUS "Adding Mujoco header " ${MUJOCO_ROOT})

add_executable(video video.cpp)
message(STATUS "Adding PyTorch Libraries " "${TORCH_LIBRARIES}")

target_include_directories(video
    PRIVATE "${MUJOCO_ROOT}/include"
    PRIVATE "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(video
    PRIVATE GL
    PRIVATE "${MUJOCO_ROOT}/bin/libglew.so"
    PRIVATE "${MUJOCO_ROOT}/bin/libglfw.so.3"
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200.so"
    PRIVATE drl
    PRIVATE drlmodel
    PRIVATE "${TORCH_LIBRARIES}"
)

