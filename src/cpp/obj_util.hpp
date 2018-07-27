#ifndef __OBJ_UTIL_HPP__
#define __OBJ_UTIL_HPP__

#include <fastbvh>
#include <vector>

namespace obj_util {

std::vector<Object *> get_objects(const char * path, int shape_index);

}

#endif // __OBJ_UTIL_HPP__
