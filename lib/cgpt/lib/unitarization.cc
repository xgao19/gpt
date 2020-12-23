#include "lib.h"


EXPORT(ProjectSU3,{
    
    void* _dst,* _src;
    if (!PyArg_ParseTuple(args, "l", &_src)) {
      std::cout << "C code: problems parsing arguments" << std::endl;
      return NULL;
    }
    
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;

    src->Project();
    
    return PyLong_FromLong(0);
  });
