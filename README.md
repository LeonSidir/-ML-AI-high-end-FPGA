# Implementation of ML-AI in high-end FPGA.

**Description**

This thesis focuses on optimizing the **HPL-AI benchmark**, which integrates methods from High-Performance Computing (HPC), Machine Learning, and AI. The process begins with a time analysis of the benchmark code running on an Intel Core i7-7700HQ (2.80GHz) using the **Intel VTune Profiler** to identify performance bottlenecks. Based on the analysis, we optimize the most time-consuming functions to improve hardware efficiency, leveraging High-Level Synthesis (HLS) tools such as **Vivado HLS** and **Vitis HLS**. Finally, we implement the optimized code on an **Alveo U200 FPGA**, compare its performance against the CPU implementation, and evaluate the achieved acceleration.

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
