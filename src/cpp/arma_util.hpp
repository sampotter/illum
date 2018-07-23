#ifndef __ARMA_UTIL_HPP__
#define __ARMA_UTIL_HPP__

#include "common.hpp"

namespace arma_util {

void save_mat(
  arma::mat const & mat,
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1);

arma::mat load_mat(
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1);

}
  
#endif // __ARMA_UTIL_HPP__
