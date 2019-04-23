//#include <bifrost/cpu_add.h>
#include <bifrost/array.h>
#include <bifrost/common.h>
#include <pybind11/pybind11.h>
//#include <bifrost/ring.h>
#include "bifrost/utils.hpp"
#include <stdlib.h>
#include "add_stuff.hpp"
//#include <stdio.h>
//#include <iostream>

//#ifdef __cplusplus
//extern "C" {
//#endif

BFstatus AddStuff(BFarray *xdata, BFarray *ydata)
{
    long nelements = num_contiguous_elements(xdata);

    float* x = (float *)xdata->data;
    float* y = (float *)ydata->data;

    for(int i=0; i < nelements; i +=1)
    {
       x[i] = x[i] + y[i];
    }

    return BF_STATUS_SUCCESS;
}

//#ifdef __cplusplus
//} // extern "C"
//#endif

PYBIND11_MODULE(bf_add_stuff, m) {
 m.doc() = "pybind11 example plugin"; // optional module docstring

 m.def("AddStuff", &AddStuff, "A function which adds two numbers");
}
