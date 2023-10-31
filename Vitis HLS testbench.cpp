#include <stdio.h>
#include <memory>
#include <string>
#include <iostream>
//Including to use ap_uint<> datatype
#include <ap_int.h>
#define DATAWIDTH 512
typedef ap_uint<DATAWIDTH> uint512_dt;

#define DATATYPE_SIZE 32
#define VECTOR_SIZE (DATAWIDTH / DATATYPE_SIZE) // vector size is 16 (512/32 = 16)
#define lda 512
#define ldb 512
#define ldc 512
#define BUFSIZE (512 * 512)

typedef ap_uint<DATATYPE_SIZE> din_type;

float uint_to_float(din_type x);
unsigned int float_to_uint(float x);

extern "C"{
	void wide_vadd(
		const  uint512_dt *in1, // Read-Only Vector 1
		const uint512_dt *in2,  // Read-Only Vector 2
		uint512_dt *out,        // Output Result
		float alpha_const,
		int sizeM,               
		int sizeK,
		int sizeN
	);
}

void test_kernel(int M, int N, int K, float ALPHA,
        float *A,
        float *B,
        float *C,
		float *D)
{
	 bool verified = true;

	 for(int j = 0; j < N; j++){
        for(int k = 0; k < K; k++){
            for(int m = 0; m < M; m++)
			{
				C[j*ldc+m] += ALPHA*B[j*ldb+k]*A[k*lda+m];
			}
        }
    }
	 for(int j = 0; j < BUFSIZE; j++){
		if (D[j] != C[j])
		{
			verified = false;
			std::cout << "ERROR: software and hardware vadd do not match: "
					<< D[j] << "!=" << C[j] << " at position " << j << std::endl;

			if (verified) {
				std::cout
					<< std::endl
					<< "OCL-mapped contiguous buffer example complete!"
					<< std::endl
					<< std::endl;
			}
			else {
				std::cout
					<< std::endl
					<< "OCL-mapped contiguous buffer example complete! (with errors)"
					<< std::endl
					<< std::endl;
			}
			return;
		}
    }

	if (verified) {
	    std::cout
	        << std::endl
	        << "OCL-mapped contiguous buffer example complete!"
	        << std::endl
	        << std::endl;
	}
	else {
	    std::cout
	        << std::endl
	        << "OCL-mapped contiguous buffer example complete! (with errors)"
	        << std::endl
	        << std::endl;
	}
}

int main(){
	//define Matrix dimensions
    int krnl_M = 512;
	int krnl_K = 512;
	int krnl_N = 512;

	float ALPHA = -1.0;

	//define the matrices
	float *a =(float *)malloc(BUFSIZE*sizeof(float));
	float *b =(float *)malloc(BUFSIZE*sizeof(float));
	float *c =(float *)malloc(BUFSIZE*sizeof(float));
	float *d =(float *)malloc(BUFSIZE*sizeof(float));

	//define the uint512_dt matrices
	uint512_dt *a_512 =(uint512_dt *)malloc(BUFSIZE/VECTOR_SIZE*sizeof(uint512_dt));
	uint512_dt *b_512 =(uint512_dt *)malloc(BUFSIZE/VECTOR_SIZE*sizeof(uint512_dt));
	uint512_dt *d_512 =(uint512_dt *)malloc(BUFSIZE/VECTOR_SIZE*sizeof(uint512_dt));

	//Set the values of the matrices
    for (int i = 0; i < BUFSIZE; i++) {
    	if(i%3 == 0)
    	{
    		a[i] = 0.157873;
    	}
    	if(i%3 == 1)
    	{
    		a[i] = 0.258985;
    	}
    	if(i%3 == 2)
    	{
    		a[i] = 0.346597;
    	}
    }
    for (int i = 0; i < BUFSIZE; i++) {
    	if(i%3 == 0)
    	{
    		b[i] = 1.125451;
    	}
    	if(i%3 == 1)
    	{
    		b[i] = 0.591342;
    	}
    	if(i%3 == 2)
    	{
    		b[i] = 0.502133;
    	}
    }

    for (int i = 0; i < BUFSIZE; i++) {
    	if(i%3 == 0)
    	{
    		c[i] = 0.578956;
    	}
    	if(i%3 == 1)
    	{
    		c[i] = 2.298757;
    	}
    	if(i%3 == 2)
    	{
    		c[i] = 1.196358;
    	}
    }

    for (int i = 0; i < BUFSIZE; i++) {
		if(i%3 == 0)
		{
			d[i] = 0.578956;
		}
		if(i%3 == 1)
		{
			d[i] = 2.298757;
		}
		if(i%3 == 2)
		{
			d[i] = 1.196358;
		}
	}

    //Convert A array from 32 bit to 512 bit
    for (int i = 0; i < BUFSIZE/VECTOR_SIZE; i++){
        uint512_dt tmp_a;
        for (int vector = 0; vector < VECTOR_SIZE; vector++){
        	tmp_a.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(a[i*VECTOR_SIZE + vector]);
        }
        a_512[i] = tmp_a;
    }

    //Convert B array from 32 bit to 512 bit
	for (int i = 0; i < BUFSIZE/VECTOR_SIZE; i++){
		uint512_dt tmp_b;
		for (int vector = 0; vector < VECTOR_SIZE; vector++){
			tmp_b.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(b[i*VECTOR_SIZE + vector]);
		}
		b_512[i] = tmp_b;
	}

    //Convert D array from 32 bit to 512 bit
    for (int i = 0; i < BUFSIZE/VECTOR_SIZE; i++){
        uint512_dt tmp_d;
        for (int vector = 0; vector < VECTOR_SIZE; vector++){
        	tmp_d.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = float_to_uint(d[i*VECTOR_SIZE + vector]);
        }
        d_512[i] = tmp_d;
    }

    //Call Kernel
    wide_vadd(a_512, b_512, d_512, ALPHA, krnl_M, krnl_K, krnl_N);


    //Convert D array from 512 bit to 32 bit
    for (int i = 0; i < BUFSIZE/VECTOR_SIZE; i++) {
		uint512_dt tmp_d = d_512[i];
		for (int vector = 0; vector < VECTOR_SIZE; vector++) {
			d[i*VECTOR_SIZE + vector] = uint_to_float(tmp_d.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE));
		}
	}

    //Verify Kernel
    test_kernel(krnl_M, krnl_N, krnl_K, ALPHA, a, b, c, d);

    return 0;
}

