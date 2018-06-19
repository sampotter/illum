cmake_minimum_required (VERSION 3.10)

project (illum)

# list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# include (PatchOpenMPApple)

include (CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG ("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

find_package (Armadillo REQUIRED)
include_directories (${ARMADILLO_INCLUDE_DIRS})

find_package (HDF5 REQUIRED)
include_directories (${HDF5_INCLUDE_DIRS})

find_package (VTK REQUIRED)
include (${VTK_USE_FILE})

# find_package (OpenMP REQUIRED)
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_subdirectory (cxxopts)
add_subdirectory (fastbvh)
add_subdirectory (tinyobjloader)

add_executable (illum_cli)
target_sources (illum_cli PUBLIC illum_cli.cpp)
target_link_libraries (illum_cli PRIVATE ${ARMADILLO_LIBRARIES})
target_link_libraries (illum_cli PRIVATE ${HDF5_LIBRARIES})
target_link_libraries (illum_cli PRIVATE cxxopts)
target_link_libraries (illum_cli PRIVATE fastbvh)
target_link_libraries (illum_cli PRIVATE tinyobjloader)
target_compile_features (illum_cli PUBLIC cxx_range_for)

add_executable (illum_gui)
target_sources (illum_gui PUBLIC illum_gui.cpp)
target_link_libraries (illum_gui PRIVATE ${VTK_LIBRARIES})
