cmake_minimum_required (VERSION 3.11)

project (illum)

option (BUILD_VTK_GUI "Build GUI application using VTK" OFF)
option (USE_TBB "Use TBB to parallelize" OFF)

include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG ("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

find_package (Boost 1.60 REQUIRED COMPONENTS system filesystem)

if (BUILD_VTK_GUI)
  find_package (VTK REQUIRED)
  include (${VTK_USE_FILE})
endif ()

if (USE_MPI)
  find_package (MPI REQUIRED)
endif ()

if (USE_TBB)
  set (TBB_BUILD_TESTS OFF CACHE BOOL "Build TBB tests and enable testing")
  add_subdirectory (tbb)
endif ()

# If we're building with MPI, we don't want Armadillo to find and link
# HDF5 in its wrapper library. The reason for this is that we only use
# HDF5 through Armadillo if we're running on a single node without MPI
# (in which case, we access HDF5 through Armadillo for saving matrices
# for ease of use and simplicity). If we use MPI, we use parallel HDF5
# to read and write HDF5 files, which isn't supported by default when
# using HDF5 through Armadillo. Armadillo also doesn't expose any way
# of using parallel HDF5 through its interface (as far as I know). So,
# we need to do things more manually with the HDF5 C API than we would
# otherwise have to using Armadillo. This dichotomy repeats itself
# elsewhere in this CMakeLists.txt file and in the C++ source code
# itself.
if (USE_MPI)
  set (DETECT_HDF5 OFF CACHE BOOL "Should Armadillo detect and build HDF5")
  set (ARMA_USE_HDF5 OFF)
  find_package (HDF5 REQUIRED)
  if (NOT ${HDF5_IS_PARALLEL})
    message (FATAL_ERROR "Parallel HDF5 required for use with MPI")
  endif ()
endif ()
add_subdirectory (armadillo)

add_subdirectory (cxxopts)
add_subdirectory (fastbvh)
add_subdirectory (tinyobjloader)

configure_file (config.hpp.in config.hpp)

add_library (illum)
target_sources (illum PRIVATE illum.cpp)
target_include_directories (illum PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries (illum PUBLIC armadillo)
target_link_libraries (illum PRIVATE fastbvh)
if (USE_TBB)
  target_link_libraries (illum PRIVATE tbb)
endif ()
target_link_libraries (illum PRIVATE tinyobjloader)
target_compile_features (illum PRIVATE cxx_std_14)

add_executable (illum_cli)
target_sources (illum_cli PUBLIC illum_cli.cpp)
target_include_directories (illum_cli PRIVATE ${Boost_INCLUDE_DIR})
target_include_directories (illum_cli PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if (USE_MPI)
  target_include_directories (illum_cli PRIVATE ${MPI_C_INCLUDE_DIRS})
endif ()
target_link_libraries (illum_cli PRIVATE ${Boost_LIBRARIES})
target_link_libraries (illum_cli PRIVATE cxxopts)
target_link_libraries (illum_cli PRIVATE illum)
if (USE_MPI)
  target_link_libraries (illum_cli PRIVATE ${HDF5_LIBRARIES})
  target_link_libraries (illum_cli PRIVATE ${MPI_LIBRARIES})
endif ()
target_compile_features (illum_cli PRIVATE cxx_std_17)

if (BUILD_VTK_GUI)
  add_executable (illum_gui)
  target_sources (illum_gui PUBLIC illum_gui.cpp)
  target_link_libraries (illum_gui PRIVATE illum)
  target_link_libraries (illum_gui PRIVATE ${VTK_LIBRARIES})
  target_compile_features (illum_gui PRIVATE cxx_std_11)
endif ()
