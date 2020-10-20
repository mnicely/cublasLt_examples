/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This example demonstrates how to use the cuBLASLt library
 * to perform SGEMM. It is nearly a drop-in replacement for
 * cublasSgemm. It performs multiplications on input/output/
 * compute types CUDA_R_32F.
 *
 * It cycles through square matrices up to maxN.
 * Computes the reference matrix, then performs matrix multiplication
 * on the device, downloads the output matrix, and compares the answer.
 *
 * Note: This example computes a reference answer on the host
 * side and can take awhile to process in N is large.
 *
 * This example has been tested with compute capability 6.0 and greater.
 */

/* Includes, system */
#include <cstdio>

/* Includes, cuda & thrust */
#include <cublasLt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

#define ROW_MAJOR 1

auto constexpr maxN = 2048;

/* Host implementation of a simple version of sgemm */
static void simple_sgemm( int const &  m,
                          int const &  n,
                          int const &  k,
                          float const &alpha,
                          float const *A,
                          float const *B,
                          float const &beta,
                          float *      C ) {

    for ( int i = 0; i < m; ++i ) {
        for ( int j = 0; j < n; ++j ) {
            float prod = 0;

            for ( int l = 0; l < k; ++l ) {
#if ROW_MAJOR
                prod += A[l + i * k] * B[j + l * n];
#else
                prod += A[l * m + i] * B[j * k + l];
#endif
            }
#if ROW_MAJOR
            C[j + i * m] = alpha * prod + beta * C[j + i * m];
#else
            C[j * m + i] = alpha * prod + beta * C[j * m + i];
#endif
        }
    }
}

void LtSgemm( cublasLtHandle_t  ltHandle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int const &       m,
              int const &       n,
              int const &       k,
              float const *     alpha,
              float const *     A,
              int const &       lda,
              float const *     B,
              int const &       ldb,
              float const *     beta,
              float *           C,
              int const &       ldc,
              void *            workspace,
              size_t            workspaceSize ) {

    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    int                             returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

#if ROW_MAJOR
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
#endif

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t
    // for details about defaults; here we just set the transforms for
    // A and B.
    CUDA_RT_CALL( cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F ) );
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof( transa ) ) );
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof( transa ) ) );

    // Create matrix descriptors. Not setting any extra attributes.
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &Cdesc, CUDA_R_32F, m, n, ldc ) );

#if ROW_MAJOR
    CUDA_RT_CALL(
        cublasLtMatrixLayoutSetAttribute( Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
    CUDA_RT_CALL(
        cublasLtMatrixLayoutSetAttribute( Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
    CUDA_RT_CALL(
        cublasLtMatrixLayoutSetAttribute( Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
#endif

    // Create preference handle; In general, extra attributes can be
    // used here to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C. However, for simplicity
    // here we assume A,B,C are always well aligned (e.g., directly
    // come from cudaMalloc)
    CUDA_RT_CALL( cublasLtMatmulPreferenceCreate( &preference ) );
    CUDA_RT_CALL( cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof( workspaceSize ) ) );

    // We just need the best available heuristic to try and run matmul.
    // There is no guarantee that this will work. For example, if A is
    // badly aligned, you can request more (e.g. 32) algos and try to
    // run them one by one until something works.
    CUDA_RT_CALL( cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults ) );

    if ( returnedResults == 0 )
        throw std::runtime_error( "!!!! Unable to find any suitable algorithms" );

    CUDA_RT_CALL( cublasLtMatmul( ltHandle,
                                  operationDesc,
                                  alpha,
                                  A,
                                  Adesc,
                                  B,
                                  Bdesc,
                                  beta,
                                  C,
                                  Cdesc,
                                  C,
                                  Cdesc,
                                  &heuristicResult.algo,
                                  workspace,
                                  workspaceSize,
                                  0 ) );

    // Descriptors are no longer needed as all GPU work was already
    // enqueued.
    CUDA_RT_CALL( cublasLtMatmulPreferenceDestroy( preference ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Cdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Bdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Adesc ) );
    CUDA_RT_CALL( cublasLtMatmulDescDestroy( operationDesc ) );
}

void calculate( int const &m, int const &n, int const &k ) {

    float alpha = 1.0f, beta = 0.0f;
    int   lda = k, ldb = n, ldc = n;
    void *d_workspace = nullptr;

    size_t sizeA         = m * k;
    size_t sizeB         = k * n;
    size_t sizeC         = m * n;
    size_t workspaceSize = 4096;

    float error_norm = 0.0f;
    float ref_norm   = 0.0f;
    float diff       = 0.0f;

    cublasLtHandle_t handle;

    /* Initialize cuBLASLt */
    CUDA_RT_CALL( cublasLtCreate( &handle ) );

    /* Allocate device memory for workspace */
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_workspace, workspaceSize ) );

    /* Initialize CUBLAS */
    printf( "cublasLt %dx%dx%d test running..\n", m, n, k );

    /* Allocate host memory for the matrices */
    thrust::host_vector<float> h_A( sizeA, 0.0f );
    thrust::host_vector<float> h_B( sizeB, 0.0f );
    thrust::host_vector<float> h_C( sizeC, 0.0f );
    thrust::host_vector<float> h_C_ref( sizeC, 0.0f );

    /* Fill the matrices with test data */
    /* Assume square matrices */
    for ( int i = 0; i < m * m; i++ ) {
        h_A[i] = rand( ) / static_cast<float>( RAND_MAX );
        h_B[i] = rand( ) / static_cast<float>( RAND_MAX );
    }

    /* Allocate device memory for the matrices */
    thrust::device_vector<float> d_A( h_A );
    thrust::device_vector<float> d_B( h_B );
    thrust::device_vector<float> d_C( sizeC, 0.0f );

    /* Retrieve raw pointer for device data */
    float *d_A_ptr = thrust::raw_pointer_cast( &d_A[0] );
    float *d_B_ptr = thrust::raw_pointer_cast( &d_B[0] );
    float *d_C_ptr = thrust::raw_pointer_cast( &d_C[0] );

    /* Performs operation using plain C code */
    simple_sgemm( m, n, k, alpha, h_A.data( ), h_B.data( ), beta, h_C_ref.data( ) );

    LtSgemm( handle,
             CUBLAS_OP_N,
             CUBLAS_OP_N,
             m,
             n,
             k,
             &alpha,
             d_A_ptr,
             lda,
             d_B_ptr,
             ldb,
             &beta,
             d_C_ptr,
             ldc,
             d_workspace,
             workspaceSize );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    /* Allocate host memory for reading back the result from device memory */
    h_C = d_C;

    /* Check result against reference */
    for ( int i = 0; i < m * m; i++ ) {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = static_cast<float>( sqrt( static_cast<double>( error_norm ) ) );
    ref_norm   = static_cast<float>( sqrt( static_cast<double>( ref_norm ) ) );

    if ( fabs( ref_norm ) < 1e-7 )
        throw std::runtime_error( "!!!! reference norm is 0\n" );

    /* Shutdown */
    CUDA_RT_CALL( cublasLtDestroy( handle ) );

    if ( error_norm / ref_norm < 1e-4f )
        printf( "cuBLASLt SGEMM test passed.\n" );
    else
        throw std::runtime_error( "!!!! cuBLASLt SGEMM test failed.\n" );
}

/* Main */
int main( int argc, char **argv ) {

    int dev {};
    CUDA_RT_CALL( cudaGetDevice( &dev ) );

    // Compute square matrices
    for ( int i = 16; i <= maxN; i *= 2 )
        calculate( i, i, i );

    return ( EXIT_SUCCESS );
}
