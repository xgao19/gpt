#include "lib.h"
#include "slice_trace.h"

EXPORT(slice_trace,{
    
    PyObject* _basis;
    long dim;
    if (!PyArg_ParseTuple(args, "Ol", &_basis, &dim)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis);

    // PVector<Lattice<vSpinColourMatrix>> tmp(basis.size());
    // for(int i=0; i<basis.size(); i++)
    //     tmp[i] = compatible<iColourMatrix<vComplexD>>(basis[i])->l;
    PVector<Lattice<vSpinColourMatrix>> tmp;
    cgpt_basis_fill(tmp, basis);

    // return cgpt_lattice_slice_trace(tmp, (int)dim);
    // PVector<Lattice<T>> basis;
    // cgpt_basis_fill(basis, _basis);
    return cgpt_slice_trace<vSpinColourMatrix>(tmp, dim);


  });