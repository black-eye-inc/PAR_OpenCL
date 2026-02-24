// ============================================================
// kernels/my_kernels.cl
// ============================================================

// Basic addition kernel
kernel void add(global const int* A, global const int* B, global int* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

// TODO 1: Modify this for multiplication
// kernel void add(global const int* A, global const int* B, global int* C) {
//     int id = get_global_id(0);
//     C[id] = A[id] * B[id];
// }

// TODO 2: Create a separate multiplication kernel
kernel void mult(global const int* A, global const int* B, global int* C) {
    int id = get_global_id(0);
    C[id] = A[id] * B[id];
}

// TODO 3: Create combined kernel performing C = A * B + B
kernel void multadd(global const int* A, global const int* B, global int* C) {
    int id = get_global_id(0);
    C[id] = A[id] * B[id] + B[id];
}

// TODO 4: Create float addition kernel
kernel void addf(global const float* A, global const float* B, global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

// TODO 5: Create double addition kernel (if supported)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void addd(global const double* A, global const double* B, global double* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}
