//Including to use ap_uint<> datatype
#include <ap_int.h>
#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 512
#define DATAWIDTH 512
#define DATATYPE_SIZE 32
#define VECTOR_SIZE (DATAWIDTH / DATATYPE_SIZE) // vector size is 16 (512/32 = 16)
typedef ap_uint<DATAWIDTH> uint512_dt;
typedef ap_uint<DATATYPE_SIZE> din_type;
typedef ap_uint<DATATYPE_SIZE + 1> dout_type;

#define lda 512
#define ldb 512
#define ldc 512

#define BUFFER_M 64
#define BUFFER_K 32

#define UNROLL_M 4
#define UNROLL_K 4



float uint_to_float(din_type x){
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.i=(unsigned int)x;
    return conv.f;
}

unsigned int float_to_uint(float x){
#pragma HLS INLINE
    union {
        unsigned int i;
        float f;
    } conv;
    conv.f=x;
    return conv.i;
}


/*
    Matrix Multiplication  Kernel Implementation using uint512_dt datatype
    Arguments:
        in1   (input)     --> Input Vector1 (uint512_dt)
        in2   (input)     --> Input Vector2 (uint512_dt)
        out   (output)    --> Output Vector (uint512_dt)
   */
extern "C"{
    void wide_vadd(
        const  uint512_dt *in1, // Read-Only Vector 1
        const uint512_dt *in2,  // Read-Only Vector 2
		uint512_dt *out,		// Output Result
		float alpha_const,
        int sizeM,
		int sizeK,
		int sizeN
    )
    {
		#pragma HLS INTERFACE m_axi port = in1 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem
		#pragma HLS INTERFACE m_axi port = in2 max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem1
		#pragma HLS INTERFACE m_axi port = out max_write_burst_length = 32 max_read_burst_length = 32 offset = slave bundle = gmem2
		#pragma HLS INTERFACE s_axilite port = in1 bundle = control
		#pragma HLS INTERFACE s_axilite port = in2 bundle = control
		#pragma HLS INTERFACE s_axilite port = out bundle = control
		#pragma HLS INTERFACE s_axilite port = sizeM bundle = control
		#pragma HLS INTERFACE s_axilite port = sizeK bundle = control
		#pragma HLS INTERFACE s_axilite port = sizeN bundle = control
		#pragma HLS INTERFACE s_axilite port = alpha_const bundle = control
		#pragma HLS INTERFACE s_axilite port = return bundle = control

    	uint512_dt vC_local[BUFFER_M]; 			 // Local memory to store Output Vector C
    	uint512_dt vA_local[BUFFER_K][BUFFER_M]; // Local memory to store Input Vector A
    	uint512_dt vB_local[BUFFER_K];           // Local memory to store Input Vector B

    	float test = alpha_const;

		#pragma HLS array_partition variable=vA_local cyclic factor=UNROLL_M dim=2
		#pragma HLS array_partition variable=vA_local cyclic factor=UNROLL_K/2 dim=1
		#pragma HLS array_partition variable=vC_local cyclic factor=UNROLL_M
		#pragma HLS array_partition variable=vB_local cyclic factor=UNROLL_K/2

		// Input vector size for integer vectors. However kernel is directly
		// accessing 512bit data (total 16 elements). So total number of read
		// from global memory is calculated here:
		int sizeM_in16 = sizeM / VECTOR_SIZE;
		int sizeM_in16_mod = sizeM % VECTOR_SIZE; //calculate the amount of elements that are in the last uint512_dt element

		//In case sizeM / VECTOR_SIZE is not perfect we need to add one more uint512_dt element, 
		//which contains the final float elements that are required for our calculations. 
		if(sizeM_in16_mod != 0){
			sizeM_in16++;		 
		}

		for(int kk = 0; kk < sizeK; kk+=BUFFER_K){
			#pragma HLS LOOP_FLATTEN off
			int curKsz=BUFFER_K;
			if((kk+BUFFER_K)>sizeK){
				curKsz=sizeK-kk;
			}

			int curKsz_mod = 0, curKsz_in16 = curKsz/16;
			if(curKsz % 16 != 0){ //If true it means this is the last uint512_dt element
				curKsz_mod = curKsz % 16;
				curKsz_in16++;
			}

			for(int ex_m = 0; ex_m < sizeM_in16; ex_m+=BUFFER_M){
				int curMsz=BUFFER_M;
				if((ex_m+BUFFER_M)>sizeM_in16){
					curMsz=sizeM_in16-ex_m;
				}

				vA_rd:
				for(int k = 0; k < curKsz; k++){
					for (int m = 0; m < curMsz; m++) {
						#pragma HLS pipeline
						#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
						vA_local[k][m] = in1[(int(((kk+k) * lda)/16))+ex_m+m]; //Warning!!!The last uint512_dt element may not be full!
					}
				}

				for(int j = 0; j < sizeN; j++){
					#pragma HLS LOOP_FLATTEN off
					vC_vB_rd:
					for (int i = 0; i < curMsz || i < curKsz_in16; i++){
						#pragma HLS pipeline
						#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
						if(i < curMsz){
							vC_local[i] = out[int((j * ldc)/16) + ex_m + i];
						}
						if(i < curKsz_in16){
							vB_local[i] = in2[(j*ldb)/16 + kk/16 + i];
						}
					}

					for(int k = 0; k < curKsz; k+=UNROLL_K){
						#pragma HLS LOOP_FLATTEN off
						#pragma HLS LOOP_TRIPCOUNT min=1 max=64
						for(int mm = 0; mm < curMsz; mm+=UNROLL_M){
							#pragma HLS pipeline
							//perform vector addition
							v2_rd_add:
							for (int kkk = 0; kkk < UNROLL_K; kkk++){
								for (int m = 0; m < UNROLL_M; m++){
									uint512_dt tmpV1 = vA_local[k+kkk][mm+m];
									uint512_dt tmpV2 = vC_local[mm+m];
									uint512_dt tmpV3 = vB_local[(k+kkk)/16];
									uint512_dt tmpOut = 0;
									din_type val1, val2;
									din_type val3 = tmpV3.range(DATATYPE_SIZE * ((k+kkk)%16 + 1) - 1, ((k+kkk)%16) * DATATYPE_SIZE);
									dout_type res;

									v2_parallel_add:
									for (int vector = 0; vector < VECTOR_SIZE; vector++){
									    #pragma HLS UNROLL
										#pragma HLS LOOP_TRIPCOUNT min=1 max=16
										if(ex_m+m+mm == sizeM_in16 - 1){
											if(vector >= sizeM_in16_mod && sizeM_in16_mod != 0){
												val2 = tmpV2.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
												res = float_to_uint((uint_to_float(val2)));
												tmpOut.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = res;
												continue;
											}
										}
										val1 = tmpV1.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
										val2 = tmpV2.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE);
										res = float_to_uint((uint_to_float(val2))  + (test*(uint_to_float(val3))*(uint_to_float(val1))));
										tmpOut.range(DATATYPE_SIZE * (vector + 1) - 1, vector * DATATYPE_SIZE) = res;
									}
									vC_local[mm+m] = tmpOut;
									if(k+kkk >= curKsz || m+mm >= curMsz){
										vC_local[mm+m] = tmpV2;
									}
								}
							}
						}
					}
					vC_write:
					for (int m = 0; m < curMsz; m++) {
						#pragma HLS pipeline
						#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
						out[int((j * ldc)/16)+ex_m+m] = vC_local[m];
					}
				}
			}
		}
    }
}
