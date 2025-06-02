cmake_minimum_required(VERSION 3.10)
project(DRLTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Torch (LibTorch)
find_package(Torch REQUIRED)
# Find MPI
find_package(MPI REQUIRED)

# Add include directory for drl.h, drlmodel.h
include_directories(../include)

# Add the executable for the tests
# We need to compile drl.cpp as it contains VPGBuffer implementation.
# drlmodel.cpp might not be strictly needed for VPGBuffer tests, but drl.h includes drlmodel.h indirectly via other headers.
# For now, let's add drl.cpp. We will stub out MPI calls for unit testing context.
add_executable(RunDRLTests test_vpgbuffer.cpp ../src/drl.cpp ../src/drlmodel.cpp)

# Link against Torch and MPI
# Also need to link against pthread if Torch requires it explicitly on some systems.
target_link_libraries(RunDRLTests PRIVATE Torch::torch MPI::MPI_CXX Threads::Threads)

# Define a preprocessor directive to handle MPI calls during tests
# target_compile_definitions(RunDRLTests PRIVATE UNIT_TESTS)
# The MPI calls in drl.cpp are not easily stubbed without modifying drl.cpp itself or having a mock MPI library.
# For VPGBuffer::store(), MPI is not directly called.
# VPGBuffer::get() calls mpi_stat.
# The logger also uses MPI.
# For now, we will proceed without UNIT_TESTS define and see if it links.
# If mpi_stat or other MPI functions are problematic for store tests, we'll need to adjust.
# VPGBuffer::store does not call MPI, so this should be fine.

enable_testing()
add_test(NAME VPGBufferTests COMMAND RunDRLTests)
