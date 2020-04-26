set(HEADER_LIST "${vpg_SOURCE_DIR}/include")
set(MUJOCO_ROOT "${vpg_SOURCE_DIR}/extern/mujoco200")

message(STATUS "Adding utils header " ${HEADER_LIST})

add_library(drl SHARED
    "${HEADER_LIST}/drl.h"
    drl.cpp
)
target_link_libraries(drl 
    PRIVATE "${TORCH_LIBRARIES}"
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200.so"
)
target_include_directories(drl 
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${MUJOCO_ROOT}/include"
)

add_library(drlmodel SHARED
    "${HEADER_LIST}/drlmodel.h"
    drlmodel.cpp
) 
target_link_libraries(drlmodel 
    PRIVATE "${TORCH_LIBRARIES}"
    PRIVATE "${MUJOCO_ROOT}/bin/libmujoco200.so"
)
target_include_directories(drlmodel 
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    PRIVATE "${MUJOCO_ROOT}/include"
)           
