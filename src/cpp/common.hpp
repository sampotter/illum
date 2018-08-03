#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#define ARMA_USE_SUPERLU 1
#include <armadillo>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <fastbvh>

template <class T>
using opt_t = boost::optional<T>;

template <class... T>
using var_t = boost::variant<T...>;

#endif // __COMMON_HPP__
