#ifndef __ARMA_UTIL_HPP__
#define __ARMA_UTIL_HPP__

#include "common.hpp"

namespace arma_util {

template <class mat_t>
void
save_mat(mat_t const & mat, boost::filesystem::path const & path)
{
  mat.save(path.string() + ".bin");
}

template <class mat_t>
mat_t
load_mat(boost::filesystem::path const & path)
{
  mat_t mat;
  mat.load(path.string());
  return mat;
}

// NB: since Armadillo stores sparse matrices in CSC format, this
// function takes A.t() instead of A, whic is equivalent to passing A
// in CSR format
//
// NB: this function assumes that all diagonal entries of A are
// nonzero
arma::vec forward_solve(arma::sp_mat const & A_t, arma::vec const & b);

}
  
#endif // __ARMA_UTIL_HPP__
