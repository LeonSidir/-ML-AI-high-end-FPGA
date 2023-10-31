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

#define A(i, j) *HPLAI_INDEX2D(A, (i), (j), lda)
#define BUFSIZE (512*512)

int kernel_calls = 0;

void call_kernel(int m, int n, int k, float alpha, float *A1, float *B1, float *C1, int sizeA1, int sizeB1, int sizeC1,
				 cl_int err , cl::Context context , cl::Kernel krnl , cl::CommandQueue q);

void sgetrf_nopiv(int m, int n, float *A, int lda) {

    int j;
    int nb = 32;
    int jb = nb;

    // Use unblock code.
    if (nb > m || nb > n) {
        sgetrf2_nopiv(m, n, A, lda);
        return;
    }

    int min_mn = m < n ? m : n;

    for (j = 0; j < min_mn; j += nb) {
        if (min_mn - j < nb) {
            jb = min_mn - j;
        }

        // Factor panel
        sgetrf2_nopiv(m - j, jb, &A(j, j), lda);

        if (j + jb < n) {
            strsm('L', 'L', 'N', 'U', jb, n - j - jb, 1.0, &A(j, j), lda,
                  &A(j, j + jb), lda);

            if (j + jb < m) {

                sgemm('N', 'N', m - j - jb, n - j - jb, jb, -1.0, &A(j + jb, j),
                      lda, &A(j, j + jb), lda, 1.0, &A(j + jb, j + jb), lda);
			}
        }
    }
}

void sgetrf2_nopiv(int m, int n, float *A, int lda) {

    int i;

    if (m <= 1 || n == 0) {
        return;
    }

    if (n == 1) {
        for (i = 1; i < m; i++) {
            A(i, 0) /= A(0, 0);
        }
    } else {  // Use recursive code

        int n1 = (m > n ? n : m) / 2;
        int n2 = n - n1;

        sgetrf2_nopiv(m, n1, A, lda);

        strsm('L', 'L', 'N', 'U', n1, n2, 1.0, A, lda, &A(0, n1), lda);

        sgemm('N', 'N', m - n1, n2, n1, -1.0, &A(n1, 0), lda, &A(0, n1), lda,
              1.0, &A(n1, n1), lda);
		sgetrf2_nopiv(m - n1, n2, &A(n1, n1), lda);
    }
    return;
}

void sgetrf2_nopiv_hw(int m, int n, float *A1, int lda, cl_int err, cl::Context context,
					cl::Kernel krnl, cl::CommandQueue q) {

    int i;

    if (m <= 1 || n == 0) {
        return;
    }

    if (n == 1) {
        for (i = 1; i < m; i++) {
            A1[0*lda+i] /= A1[0*lda+0];
        }
    } else {  // Use recursive code

        int n1 = (m > n ? n : m) / 2;
        int n2 = n - n1;

        sgetrf2_nopiv_hw(m, n1, A1, lda, err, context, krnl, q);

        strsm_hw('L', 'L', 'N', 'U', n1, n2, 1.0, A1, lda, &A1[n1*lda+0], lda);
        printf("\nkernel call NO:%d\n", kernel_calls++);
        
		call_kernel(m - n1, n2, n1, -1.0, &A1[0*lda+n1], &A1[n1*lda+0], &A1[n1*lda+n1], BUFSIZE - (0*lda+n1), BUFSIZE - (n1*lda+0), BUFSIZE - (n1*lda+n1),
				 err , context , krnl , q);

		sgetrf2_nopiv_hw(m - n1, n2, &A1[n1*lda+n1], lda, err, context, krnl, q);
    }
    return;
}
void sgetrf_nopiv_hw(int m, int n, float *A1, int lda, cl_int err , cl::Context context ,
					cl::Kernel krnl , cl::CommandQueue q) {
    int j;
    int nb = 32;
    int jb = nb;

    // Use unblock code.
    if (nb > m || nb > n) {
        sgetrf2_nopiv_hw(m, n, A1, lda, err, context, krnl, q);
        return;
    }

    int min_mn = m < n ? m : n;

    for (j = 0; j < min_mn; j += nb) {
        if (min_mn - j < nb) {
            jb = min_mn - j;
        }

        // Factor panel
        sgetrf2_nopiv_hw(m - j, jb, &A1[j*lda+j], lda, err, context, krnl, q);

        if (j + jb < n) {
            strsm_hw('L', 'L', 'N', 'U', jb, n - j - jb, 1.0, &A1[j*lda+j], lda,
                  &A1[(j+jb)*lda+j], lda);

            if (j + jb < m) {
            	printf("\nkernel call NO:%d\n", kernel_calls++);
                
				call_kernel(m - j - jb, n - j - jb, jb, -1.0, &A1[j*lda + j+jb], &A1[(j+jb)*lda+j], &A1[(j+jb)*lda+j+jb], BUFSIZE - (j*lda + j+jb), BUFSIZE - ((j+jb)*lda+j), BUFSIZE - ((j+jb)*lda+j+jb),
							err , context , krnl , q);

			}
        }
    }
}

