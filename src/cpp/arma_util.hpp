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

// NB: since Armadillo stores sparse matrices in CSC format, this
// function takes A.t() instead of A, whic is equivalent to passing A
// in CSR format
//
// NB: this function assumes that all diagonal entries of A are
// nonzero
arma::vec forward_solve(arma::sp_mat const & A_t, arma::vec const & b);

}
  
#endif // __ARMA_UTIL_HPP__
