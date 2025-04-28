//nvcc mat_mul.cu -o main && ./main
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

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

#define TILE_WIDTH 3 
__global__ void mat_mul_knl_tiled(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    // like in the naive kernel, each thread will be responsible for a single element of C
    float C_val = 0.0;
    // each tile corresponds 1-1 to a block
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; 
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; 

    if (row >= A_rows || col >= B_cols) {
        return; 
    }

    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    // loop across inner dimension in strides of the TILE_WIDTH
    // we're concerned with the row'th row of A and the col'th column of B so when 
    // we loop over the inner dimension, we vary column/row
    int A_tile_col = 0;
    int B_tile_row = 0;
    for (int k = 0; k < ((A_cols + TILE_WIDTH- 1)/TILE_WIDTH); k++) {
        // fill up the current tile with elements of A and B 
        A_tile_col =  k * TILE_WIDTH + threadIdx.x;
        B_tile_row =  TILE_WIDTH * k + threadIdx.y;
        // TODO: this should happen in a separate loop to cut down on ifs
        if (A_tile_col < A_cols) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * A_cols + A_tile_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0; 
        }

        if (B_tile_row < A_cols) {
            B_tile[threadIdx.y][threadIdx.x] = B[B_tile_row * B_cols + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0; 
        }

        __syncthreads();
        // at this point all threads in the block have filled up a square tile
        for (int i = 0; i < TILE_WIDTH; i++) {
            C_val += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    } 

    C[row * B_cols + col] = C_val;
}


typedef void (*mat_mul_func)(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols); 

void compare_mat_mul_kernels(mat_mul_func f, mat_mul_func g, int A_rows, int A_cols, int B_cols) {
    int A_size = sizeof(float) * A_rows * A_cols;
    int B_size = sizeof(float) * A_cols * B_cols;
    int C_size = sizeof(float) * A_rows * B_cols;

    float* A = generate_random_matrix(A_rows, A_cols); 
    float* B = generate_random_matrix(A_cols, B_cols); 

    float* C_ref = allocate_matrix(A_rows, B_cols);

    // establish ground truth
  //  print_matrix(A, A_rows, A_cols);
   // print_matrix(B, A_cols, B_cols);
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

    time_t f_start = time(NULL);
    f<<<dim3(A_rows, B_cols), dim3(TILE_WIDTH, TILE_WIDTH)>>>(A_d, B_d, C_d_f, A_rows, A_cols, B_cols);
    cudaDeviceSynchronize(); 
    time_t f_time = time(NULL) - f_start;

    cudaMemcpy(C_h_f, C_d_f, C_size, cudaMemcpyDeviceToHost); 
    cudaFree(C_d_f);

    if (!matrices_are_equal(C_ref, C_h_f, A_rows, B_cols, A_rows, B_cols)) {
        printf("kernel f is wrong!\n");
    }
    printf("f result: \n");
//    print_matrix(C_h_f, A_rows, B_cols);

    free(C_h_f);

    // then knl g
    float* C_h_g = allocate_matrix(A_rows, B_cols);
    float* C_d_g= allocate_matrix(A_rows, B_cols);
    cudaMalloc(&C_d_g, C_size);

    time_t g_start = time(NULL);
    g<<<dim3(1+ (A_rows/TILE_WIDTH), 1 + (B_cols/TILE_WIDTH)), dim3(TILE_WIDTH, TILE_WIDTH)>>>(A_d, B_d, C_d_g, A_rows, A_cols, B_cols);
    cudaDeviceSynchronize(); 
    time_t g_time = time(NULL) - g_start;

    cudaMemcpy(C_h_g, C_d_g, C_size, cudaMemcpyDeviceToHost); 
    if (!matrices_are_equal(C_ref, C_h_g, A_rows, B_cols, A_rows, B_cols)) {
        printf("kernel g is wrong!\n");
    }
    cudaFree(C_d_g);
    printf("g result: \n");
//    print_matrix(C_h_g, A_rows, B_cols);
    free(C_h_g);

    printf("kernel f run time: %d\n", f_time);
    printf("kernel g run time: %d\n", g_time);
    cudaFree(A_d);
    cudaFree(B_d);
    free(A);
    free(B);
    free(C_ref);
}

int main() {
    compare_mat_mul_kernels(mat_mul_knl_naive, mat_mul_knl_tiled, 102, 300, 102);
}

