#include <iostream>
#include <memory>
#include <string>
#include "hpl-ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "xcl2.hpp"
#include <vector>
#include "event_timer.hpp"

#define BUFSIZE (512*512)

void sgetrf_nopiv_hw(int m, int n, float *A1, int lda, cl_int err , cl::Context context ,
					cl::Kernel krnl , cl::CommandQueue q);

void compareFiles(char* file1, char* file2) { // A function that compares the contents of 2 files (float input)
	int counter = 0;
    FILE* f1 = fopen(file1, "r");
    FILE* f2 = fopen(file2, "r");

    if (f1 == NULL || f2 == NULL) {
        printf("Error opening files.\n");
        return;
    }

    float element1, element2;
    int isSame = 1;  // Assume files are the same initially

    // Compare each element in the files
    while ((fscanf(f1, "%f", &element1) == 1) && (fscanf(f2, "%f", &element2) == 1)) {
        if (element1 != element2) {
            isSame = 0;  // Files are not the same
            printf("Files do not match!\n\n");
            return;
        }
    }

    // Check if one file reached EOF while the other still has elements
    if ((fscanf(f1, "%f", &element1) == 1) || (fscanf(f2, "%f", &element2) == 1)){
    	printf("Files do not match!\n\n");
        isSame = 0;
        return;
    }
	printf("Files do match!\n\n");
    fclose(f1);
    fclose(f2);
}

void matrix_to_file(int size, float *matrix, char *matrix_filename){ //A function that passes the contents of a float matrix to a file

	FILE *fp;
    fp = fopen(matrix_filename, "w"); // open file for writing

    if (fp == NULL) { // check if file was opened successfully
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < size; i++) {
    	fprintf(fp, "%f ", matrix[i]);
    }

    fclose(fp); // close the file
    printf("Matrix saved to file successfully!\n");
}

void file_to_matrix(float *matrix_final, char *matrix_filename){ //A function that passes the contents of a file to a float matrix

	FILE *fp;
	int size = 0;
    fp = fopen(matrix_filename, "r"); // open file for reading

    if (fp == NULL) { // check if file was opened successfully
        printf("Error opening file.\n");
        return;
    }

    while (fscanf(fp, "%f", &matrix_final[size]) != EOF) { // read from the file
        size++; // increment size of array
    }
    fclose(fp); // close the file

    printf("File contents saved to matrix successfully!\n");
}

void call_kernel(int m, int n, int k, float alpha, float *A1, float *B1, float *C1, int sizeA1, int sizeB1, int sizeC1,
				 cl_int err , cl::Context context , cl::Kernel krnl , cl::CommandQueue q){
	EventTimer et;
	
	std::vector<float, aligned_allocator<float>> vectorA(sizeA1);
	std::vector<float, aligned_allocator<float>> vectorB(sizeB1);
	std::vector<float, aligned_allocator<float>> vectorC(sizeC1);
	
	for(int i = 0; i < sizeA1; i++){
		vectorA[i] = A1[i];
	}
	for(int i = 0; i < sizeB1; i++){
		vectorB[i] = B1[i];
	}
	for(int i = 0; i < sizeC1; i++){
		vectorC[i] = C1[i];
	}
	
	et.add("Allocate Buffer in Global Memory");
    OCL_CHECK(err, cl::Buffer a_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeA1 * sizeof(float),
    								vectorA.data(), &err));
    OCL_CHECK(err, cl::Buffer b_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeB1 * sizeof(float),
    								vectorB.data(), &err));
    OCL_CHECK(err, cl::Buffer c_buf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeC1 * sizeof(float),
    								vectorC.data(), &err));
    et.finish();

    // Set the Kernel Arguments
	et.add("Set the Kernel Arguments");
	int nargs = 0;
    OCL_CHECK(err, err = krnl.setArg(nargs++, a_buf));
    OCL_CHECK(err, err = krnl.setArg(nargs++, b_buf));
    OCL_CHECK(err, err = krnl.setArg(nargs++, c_buf));
	OCL_CHECK(err, err = krnl.setArg(nargs++, alpha));
    OCL_CHECK(err, err = krnl.setArg(nargs++, m));
    OCL_CHECK(err, err = krnl.setArg(nargs++, k));
    OCL_CHECK(err, err = krnl.setArg(nargs++, n));
    et.finish();

	et.add("Copy input data to device from global memory");
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a_buf, b_buf, c_buf}, 0 /* 0 means from host*/));
    et.finish();

    et.add("Launch the Kernel");
    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl));
	et.finish();

    // Copy Result from Device Global Memory to Host Local Memory
	et.add("Copy Result from Device Global Memory to Host Local Memory");
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({c_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
	et.finish();
	et.add("Queue finish");
    OCL_CHECK(err, err = q.finish());
    et.finish();
    et.print();

    for(int i = 0; i < sizeC1; i++){
    	C1[i] = vectorC[i];
	}
}


int main(int argc, char* argv[]) {

    //Matrices dimensions
    int n = 512;

	int lda = (n + 16 - 1) / 16 * 16;  // round up to multiple of 16

    unsigned long long iseed = 1;      // RNG seed

    double* A = (double*)malloc(BUFSIZE * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    float* sA = (float*)malloc(BUFSIZE * sizeof(float));
    float* sb = (float*)malloc(n * sizeof(float));

    matgen(A, lda, n, iseed);
    vecgen(b, n, iseed+1);

    // Convert A and b to single.
    convert_double_to_float(A, lda, sA, lda, n, n);
    convert_double_to_float(b, n, sb, n, n, 1);

	//FPGA initialization
	EventTimer et;
    std::string binaryFile = argv[1];

	//Allocate Memory in Host Memory
    et.add("Allocate Memory in Host Memory");
    std::vector<float, aligned_allocator<float>> A_SW(BUFSIZE);
    std::vector<float, aligned_allocator<float>> A_HW(BUFSIZE);
    et.finish();

	et.add("Fill the buffers");
	for(int i = 0; i < BUFSIZE; i++){
		A_SW[i] = sA[i];
		A_HW[i] = sA[i];
	}

	et.finish();

	et.add("OpenCL host code");
    // OPENCL HOST CODE AREA START
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl;
    auto devices = xcl::get_xil_devices();
    et.finish();

    et.add("Read_binary_file");
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl = cl::Kernel(program, "wide_vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    et.finish();


    //LU factorization without pivoting.
	sgetrf_nopiv(n, n, A_SW.data(), lda);
	sgetrf_nopiv_hw(n, n, A_HW.data(), lda, err, context, krnl, q);

	matrix_to_file(BUFSIZE, A_SW.data(), "/home/student10/Desktop/exported/matrixA_SW.txt");
	matrix_to_file(BUFSIZE, A_HW.data(), "/home/student10/Desktop/exported/matrixA_HW.txt");
	compareFiles("/home/student10/Desktop/exported/matrixA_SW.txt", "/home/student10/Desktop/exported/matrixA_HW.txt");

	std::cout << "\nFINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl
	              << std::endl;
    return 0;
}
