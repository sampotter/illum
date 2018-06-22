cmake_minimum_required (VERSION 3.11)

project (illum)

option (USE_TBB "Use TBB to parallelize" OFF)

include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG ("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

find_package (VTK)
include (${VTK_USE_FILE})

if (USE_TBB)
  set (TBB_BUILD_TESTS OFF CACHE BOOL "Build TBB tests and enable testing")
  add_subdirectory (tbb)
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
target_link_libraries (illum_cli PRIVATE cxxopts)
target_link_libraries (illum_cli PRIVATE illum)
target_compile_features (illum_cli PRIVATE cxx_std_11)

if (VTK_FOUND)
  add_executable (illum_gui)
  target_sources (illum_gui PUBLIC illum_gui.cpp)
  target_link_libraries (illum_gui PRIVATE illum)
  target_link_libraries (illum_gui PRIVATE ${VTK_LIBRARIES})
  target_compile_features (illum_gui PRIVATE cxx_std_11)
endif ()
