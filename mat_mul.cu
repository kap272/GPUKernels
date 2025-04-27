//nvcc mat_mul.cu -o main && ./main
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

 void add_vec(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }

}

float* generate_random_vector(int n) {
    float* vec = (float*) malloc(sizeof(float) * n); 
    for (int i = 0; i < n; i++) {
        vec[i] = rand() % 10;
    }
    return vec;
}

float* allocate_matrix(int rows, int cols) {
    float* vec = (float*) malloc(sizeof(float) * rows * cols); 
    return vec;
}

float* generate_random_matrix(int rows, int cols) {
    float* M = (float*) malloc(sizeof(float) * rows * cols); 
    for (int i = 0; i < rows * cols; i++) {
        M[i] = rand() % 10;
    }
    return M;

}

void print_vector(float* v, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f\n", v[i]);
    }
    printf("\n");
}    

void print_matrix(float* M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", M[cols * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool matrices_are_equal(float* A, float *B, int A_rows, int A_cols, int B_rows, int B_cols) {
    if (A_rows != B_rows || A_cols != B_cols) {
        return false;
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            if (A[i * A_cols + j] != B[i * B_cols + j]) {
                return false;
            }
        }
    }

    return true;
}

void mat_mul(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float C_val = 0.0;
            for (int l = 0; l < A_cols; l++) {
                C_val += A[i * A_cols + l] * B[l * B_cols + j];
            }
            C[i * B_cols + j] = C_val;
        }
    }
}

__global__ void mat_mul_knl_naive(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if (row < A_rows && col < B_cols) {
        float C_val = 0.0;
        for (int inner_idx = 0; inner_idx < A_cols; inner_idx++) {
            C_val += A[row * A_cols + inner_idx] * B[inner_idx * B_cols + col];
        }
        C[row * B_cols + col] += C_val;

    }
}


int main() {
    float A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; 
    float* C = (float*) malloc(9 * sizeof(float)); 

//    print_matrix(A, 3, 3); 
//    print_matrix(B, 3, 3); 
    mat_mul(A, B, C, 3, 3, 3);
//    print_matrix(C, 3, 3); 



    // test kernel
    float* C_h = (float*) malloc(9 * sizeof(float)); 
    // set up on-device variables
    float* A_d;
    float* B_d; 
    float* C_d;
    // allocate memory on device
    cudaMalloc((void**) &A_d, 9 * sizeof(float));
    cudaMalloc((void**) &B_d, 9 * sizeof(float));
    cudaMalloc((void**) &C_d, 9 * sizeof(float));
    // copy data from host to device
    //      target, source, size, direction
    cudaMemcpy(A_d, A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, 9 * sizeof(float), cudaMemcpyHostToDevice);
    // invoke kernel
    mat_mul_knl_naive<<<dim3(3, 3), dim3(1, 1)>>>(A_d, B_d, C_d, 3, 3, 3);
    // waith for kernel to finish
    cudaDeviceSynchronize();
    // move data back to host}
    cudaMemcpy(C_h, C_d, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    // free data on device

    print_matrix(C, 3, 3);
    print_matrix(C_h, 3, 3);

    assert(matrices_are_equal(C, C_h, 3, 3, 3, 3));
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
