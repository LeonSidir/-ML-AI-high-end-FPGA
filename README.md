# Implementation of ML-AI in high-end FPGA.

Source code files:

1) **blas.cpp:** Contains basic linear algebra functions for operations between matrices.
2) **convert.cpp:** Contains functions for matrix data type convertions, from float to double and vise versa.
3) **gmres.cpp:** Contains the gmres (Generalized minimal residual) method.
4) **hpl-ai.cpp:** Contains the main() function and it is the main file of the benchmark.
5) **hpl-ai.h:** Contains all function definitions.
6) **matgen.cpp:** Contains functions responsible for creating and initializing matrix A with random values.
7) **print.cpp:** Contains functions for printing the contents of a float or double matrix on the console.
8) **sgetrf_nopiv.cpp:** Contains the LU factorization method.
9) **timer.cpp:** Calculates function runtimes.
10) **wide_vadd.cpp:** Contains the host code for Vitis.
11) **wide_vadd_krnl.cpp:** Contains the kernel code of the customized sgemm function for Vitis HLS and Vitis.
12) **Vitis HLS testbench.cpp:** Vitis HLS testbench code.
13) **Vitis HLS kernel.cpp:** Vitis HLS kernel code (same as **wide_vadd_krnl.cpp**).
14) **event_timer.cpp, event_timer.hpp, xcl2.cpp, xcl2.hpp:** OpenCL libraries.
