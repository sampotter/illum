cmake_minimum_required (VERSION 3.11)

project (illum)

add_subdirectory (fastbvh)
add_subdirectory (tinyobjloader)

add_executable (phobos_test)
target_sources (phobos_test PUBLIC phobos_test.cpp)
target_link_libraries (phobos_test PRIVATE fastbvh)
target_link_libraries (phobos_test PRIVATE tinyobjloader)
target_compile_features (phobos_test
  PUBLIC
    cxx_range_for)
