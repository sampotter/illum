#include "arma_util.hpp"

arma::vec
arma_util::forward_solve(arma::sp_mat const & A_t, arma::vec const & b)
{
  arma::vec x(b);
  for (arma::uword i = 0; i < A_t.n_cols; ++i) {
    auto it = A_t.begin_col(i), it_end = A_t.end_col(i);
    for (; it != it_end; ++it) {
      arma::uword j = it.row();
      if (j == i) {
        continue;
      }
      x(i) -= A_t(j, i)*x(j);
    }
    x(i) /= A_t(i, i);
  }
  return x;
}
