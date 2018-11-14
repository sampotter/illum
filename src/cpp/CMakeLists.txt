cmake_minimum_required (VERSION 3.11)

project (illum)

enable_language (CXX Fortran)

option (BUILD_RENDERER "Build the command-line renderer" ON)
option (USE_MPI "Use MPI to distribute" OFF)
option (USE_TBB "Use TBB to parallelize" OFF)

include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG ("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

find_package (Boost 1.60 REQUIRED COMPONENTS system filesystem)

if (BUILD_RENDERER)
  find_library (pngpp_LIBRARY png++)
  find_package (PNG REQUIRED)
endif ()

if (USE_MPI)
  find_package (MPI REQUIRED)
endif ()

if (USE_TBB)
  set (TBB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tbb")
  include (${TBB_ROOT}/cmake/TBBBuild.cmake)
  tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)
  find_package(TBB REQUIRED tbb_preview)
endif ()

add_subdirectory (armadillo)
add_subdirectory (cxxopts)
add_subdirectory (fastbvh)
add_subdirectory (tinyobjloader)

configure_file (config.hpp.in config.hpp)

add_library (conduction STATIC)
target_sources (conduction PRIVATE conductionT.f conductionQ.f tridag.for
  conduction.cpp)

add_library (illum STATIC)
target_compile_options (illum PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_sources (illum PRIVATE illum.cpp arma_util.cpp conduction.cpp
  obj_util.cpp)
target_include_directories (illum PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries (illum PUBLIC armadillo)
target_link_libraries (illum PRIVATE conduction)
target_link_libraries (illum PRIVATE fastbvh)
if (USE_TBB)
  target_link_libraries (illum PRIVATE ${TBB_IMPORTED_TARGETS})
endif ()
target_link_libraries (illum PRIVATE tinyobjloader)
target_compile_features (illum PRIVATE cxx_std_14)

add_executable (illum_cli)
target_compile_options (illum_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_property (TARGET illum_cli PROPERTY BUILD_RPATH ${Boost_LIBRARY_DIRS})
target_sources (illum_cli PUBLIC illum_cli.cpp arma_util.cpp)
target_include_directories (illum_cli PRIVATE ${Boost_INCLUDE_DIR})
target_include_directories (illum_cli PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if (USE_MPI)
  target_include_directories (illum_cli PRIVATE ${MPI_C_INCLUDE_DIRS})
endif ()
target_link_libraries (illum_cli PRIVATE ${Boost_LIBRARIES})
target_link_libraries (illum_cli PRIVATE cxxopts)
target_link_libraries (illum_cli PRIVATE illum)
if (USE_MPI)
  target_link_libraries (illum_cli PRIVATE ${MPI_LIBRARIES})
endif ()
target_compile_features (illum_cli PRIVATE cxx_std_14)

add_custom_command (
  TARGET illum_cli
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_SOURCE_DIR}/../../data/vesta_xtiny.obj ${CMAKE_CURRENT_BINARY_DIR})

if (BUILD_RENDERER)
  add_executable (illum_renderer_cli)
  target_compile_options (illum_renderer_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_property (TARGET illum_renderer_cli PROPERTY BUILD_RPATH ${Boost_LIBRARY_DIRS})
  target_sources (illum_renderer_cli PUBLIC illum_renderer_cli.cpp obj_util.cpp)
  target_include_directories (illum_renderer_cli PRIVATE ${Boost_INCLUDE_DIR})
  target_include_directories (illum_renderer_cli PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries (illum_renderer_cli PRIVATE armadillo)
  target_link_libraries (illum_renderer_cli PRIVATE ${Boost_LIBRARIES})
  target_link_libraries (illum_renderer_cli PRIVATE cxxopts)
  target_link_libraries (illum_renderer_cli PRIVATE fastbvh)
  target_link_libraries (illum_renderer_cli PRIVATE ${PNG_LIBRARIES})
  target_link_libraries (illum_renderer_cli PRIVATE tinyobjloader)
  target_compile_features (illum_renderer_cli PRIVATE cxx_std_17)
endif ()
