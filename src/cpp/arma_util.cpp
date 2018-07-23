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
