cmake_minimum_required(VERSION 3.13)

project(vpg VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -pedantic -Werror -O3 
                    -Wno-unused-variable -Wno-unused-parameter
                    -mavx -pthread -fopenmp
)

find_package(Torch REQUIRED)

add_subdirectory(src)
add_subdirectory(main)


