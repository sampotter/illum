#include "arma_util.hpp"

void
arma_util::save_mat(arma::mat const & mat, boost::filesystem::path const & path)
{
  mat.save(path.string() + ".bin");
}

arma::mat
arma_util::load_mat(boost::filesystem::path const & path)
{
  arma::mat mat;
  mat.load(path.string());
  return mat;
}

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
