#include "lib.h"


EXPORT(ProjectSU3,{
    
    void* _dst,* _src;
    if (!PyArg_ParseTuple(args, "ll", &_dst, &_src)) {
      std::cout << "C code: problems parsing arguments" << std::endl;
      return NULL;
    }

    std::cout << "C code: before the casting" << std::endl;
    
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;

    std::cout << "C code: before the calling cgpt_Lattice_base->ProjectOnGroup()" << std::endl;

    dst->Project(src);

    // dst->copy_from(src);

    std::cout << "C code: cgpt_Lattice_base->ProjectOnGroup() finished" << std::endl;
    
    return PyLong_FromLong(0);
  });
