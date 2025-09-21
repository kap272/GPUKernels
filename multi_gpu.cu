#include <stdio.h>
#include <cuda_runtime.h>

float* generate_random_matrix(int rows, int cols) {
    float* M = (float*) malloc(sizeof(float) * rows * cols); 
    for (int i = 0; i < rows * cols; i++) {
        M[i] = rand() % 10;
    }
    return M;

}

float* allocate_matrix(int rows, int cols) {
    float* vec = (float*) malloc(sizeof(float) * rows * cols); 
    return vec;
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

void print_matrix(float* M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", M[cols * i + j]);
        }
        printf("\n");
    }
    printf("\n");
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
    int gpu_count;
    bool print_results = true ;
    cudaError_t err = cudaGetDeviceCount(&gpu_count);
    if (err != cudaSuccess) {
        printf("failed to count gpus; "); printf(cudaGetErrorString(err));
    }
    printf("found %d GPUs\n", gpu_count);
    printf("hello world\n");

    int A_rows = 3;
    int A_cols = 3;
    int B_cols = 3;

    int A_size = sizeof(float) * A_rows * A_cols;
    int B_size = sizeof(float) * A_cols * B_cols;
    int C_size = sizeof(float) * A_rows * B_cols;

    float* A = generate_random_matrix(A_rows, A_cols); 
    float* B = generate_random_matrix(A_cols, B_cols); 

    float* C_ref = allocate_matrix(A_rows, B_cols);

    // establish ground truth
//    print_matrix(A, A_rows, A_cols);
//    print_matrix(B, A_cols, B_cols);
    mat_mul(A, B, C_ref, A_rows, A_cols, B_cols);
    //print_matrix(C_ref, A_rows, B_cols);


    // run kernel f
    float* C_h_f = allocate_matrix(A_rows, B_cols);

    float* A_d;
    float* B_d;
    float* C_d_f = allocate_matrix(A_rows, B_cols);

    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size); 
    cudaMalloc(&C_d_f, C_size); 

    cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice); 

    float f_time;
    cudaEvent_t f_start, f_stop;    
    cudaEventCreate(&f_start);
    cudaEventCreate(&f_stop);
    cudaEventRecord(f_start, 0);

    dim3 f_grid_dims(3, 3);
    dim3 f_block_dims(1, 1);

    mat_mul_knl_naive<<<f_grid_dims, f_block_dims>>>(A_d, B_d, C_d_f, A_rows, A_cols, B_cols);
    
    cudaEventRecord(f_stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&f_time, f_start, f_stop);

    cudaMemcpy(C_h_f, C_d_f, C_size, cudaMemcpyDeviceToHost); 
    cudaFree(C_d_f);

    if (!matrices_are_equal(C_ref, C_h_f, A_rows, B_cols, A_rows, B_cols)) {
        printf("kernel f is wrong!\n");
    } else {
        printf("kernel f is not wrong \n");
    }
    if (print_results) {
        printf("f result: \n");
        print_matrix(C_h_f, A_rows, B_cols);
    }

    free(C_h_f);
}
