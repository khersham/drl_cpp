set(HEADER_LIST "${CMAKE_SOURCE_DIR}/include")
set(MUJOCO_ROOT "${CMAKE_SOURCE_DIR}/extern/mujoco200")

message(STATUS "Adding utils header " ${HEADER_LIST})

add_library(drl SHARED
    "${HEADER_LIST}/drl.h"
    drl.cpp
)
target_link_libraries(drl 
    PRIVATE "${TORCH_LIBRARIES}"
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200nogl.so"
)
target_include_directories(drl 
    PRIVATE "${HEADER_LIST}"
    PRIVATE "${MUJOCO_ROOT}/include"
)

add_library(drlmodel SHARED
    "${HEADER_LIST}/drlmodel.h"
    drlmodel.cpp
) 
target_link_libraries(drlmodel 
    PRIVATE "${TORCH_LIBRARIES}"
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200nogl.so"
)
target_include_directories(drlmodel 
    PRIVATE "${HEADER_LIST}"
    PRIVATE "${MUJOCO_ROOT}/include"
)           
