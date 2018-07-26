#include "arma_util.hpp"

void
arma_util::save_mat(
  arma::mat const & mat,
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1)
{
#if USE_MPI
  std::string path node_path = path.string() + "_" + std::to_string(*i0) + "_" +
    std::to_string(*i1) + ".bin";
  mat.save(node_path);
#else
  (void) i0;
  (void) i1;
  mat.save(path.string() + ".bin");
#endif
}

arma::mat
arma_util::load_mat(
  boost::filesystem::path const & path,
  opt_t<int> i0,
  opt_t<int> i1)
{
  arma::mat mat;
#if USE_MPI
  // TODO: this very probably doesn't work... will probably need to
  // cook something up using MPI-IO or disallow this
  std::string node_path = path + "_" + std::to_string(*i0) + "_" +
    std::to_string(*i1) + ".bin";
  mat.load(node_path);
#else
  (void) i0;
  (void) i1;
  mat.load(path.string());
#endif
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
