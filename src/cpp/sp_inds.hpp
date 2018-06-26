#ifndef __SP_INDS_HPP__
#define __SP_INDS_HPP__

#include <vector>

template <class uint>
struct sp_inds {


  void append(sp_inds<uint> const & other) {
    if (rowind.empty()) {
      if (other.rowind.empty()) {
        return;
      } else {
        rowind = other.rowind;
        colptr = other.colptr;
        return;
      }      
    }
    auto nelts = rowind.size();
    if (nelts != 0) {
      assert(nelts == *--colptr.end());
    }
    rowind.insert(rowind.end(), other.rowind.begin(), other.rowind.end());
    for (auto it = ++other.colptr.begin();
         it != other.colptr.end();
         ++it) {
      colptr.push_back(nelts + *it);
    }
  }
  
  std::vector<uint> rowind, colptr;
};

#endif // __SP_INDS_HPP__
