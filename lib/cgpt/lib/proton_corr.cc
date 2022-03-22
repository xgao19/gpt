#include "lib.h"
#include "proton_corr.h"
  
  EXPORT(slice_proton_2pt,{
    
    PyObject* _lhs;
    PyObject* _mom;
    long dim;
    if (!PyArg_ParseTuple(args, "OOl", &_lhs, &_mom, &dim)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    std::vector<cgpt_Lattice_base*> basism;
    cgpt_basis_fill(basis,_lhs);
    cgpt_basis_fill(basism, _mom);

    // PVector<Lattice<vSpinColourMatrix>> tmp(basis.size());
    // for(int i=0; i<basis.size(); i++)
    //     tmp[i] = compatible<iColourMatrix<vComplexD>>(basis[i])->l;
    PVector<Lattice<vSpinColourMatrix>> lhs;
    PVector<LatticeComplex> mom;
    cgpt_basis_fill(lhs, basis);
    cgpt_basis_fill(mom, basism);

    return cgpt_slice_Proton<vSpinColourMatrix>(lhs, mom, dim);


  });

Export(fill_proton_seq_src,{

    PyObject* _propagator;
    PyObject* _src;
    long tf;

    if (!PyArg_ParseTuple(args, "OOl", &_propagator, &_src, &tf)) {
        return NULL;
    }

    std::vector<cgpt_Lattice_base*> tmp1;
    std::vector<cgpt_Lattice_base*> tmp2;
    cgpt_basis_fill(tmp1,_propagator);
    cgpt_basis_fill(tmp2, _src);


    PVector<Lattice<vSpinColourMatrix>> propagator;
    PVector<Lattice<vSpinColourMatrix>> src;
    cgpt_basis_fill(propagator, tmp1);
    cgpt_basis_fill(src, tmp2);

    //fill the seq. src for the proton QPDF calculation, note in the function below we use a std. argument for the 
    //isospin singlet, to change, add another integer in function call.
    fill_seq_src<vSpinColourMatrix>(propagator, src, tf);     

});