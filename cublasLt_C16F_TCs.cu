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
 * There are two scenarios that use tensor operation with complex
 * half precision.
 *
 * CUDA_C_16F, CUDA_C_16F, CUDA_C_16F, CUDA_C_32F, CUDA_C_32F
 * CUDA_C_16F, CUDA_C_16F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F
 *
 * First, set IDENTITY 1 and PRINT 1. This will create 2 input identity
 * matrices, in matrix A and B. The result should print a 16x16 identity
 * matrix. You'll notice they are pairs, to show real and imaginary parts.
 *
 * Next, set PRINT 0 and you can test multiple square matrices
 * with the identity test.
 *
 * Lastly, set IDENTITY 0 and PRINT 0 and you will test multiple
 * square and non-square matrices with randomly generated data.
 *
 * You should notice another define variable TIME_TRANSFORM.
 * When matrix A and B are generated with random numbers, they
 * were generated with an interleaved layout. Meaning data is
 * stored [real, imaginary, real, imaginary, ...]. In order to
 * utilize Tensor Cores the data must be in planar layout. Meaning
 * data is stored [real, real, real, .... (half way), imaginary, imaginary, imaginary].
 *
 * When TIME_TRANSFORM 1 then the time taken to transform A and B
 * to planar layout, perform matrix multiplication, and the time
 * taken to transform C from planar to interleaved layout is calculated.
 * When TIME_TRANSFORM 0 only matrix multiplication is profiled.
 *
 * This example requires compute capability 7.0 or greater.
 */

/* Includes, system */
#include <cstdio>

/* Includes, cuda */
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

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

#define PRINT 0
#define IDENTITY 0
#define SCENARIO 0
#define TIME_TRANSFORM 1

auto constexpr kernelRepeats   = 50;
auto constexpr threadsPerBlock = 1024;

// Key
// *I = Input
// *O = Output
// *s = Scale
// *C = Compute

#if SCENARIO == 0  // CUDA_C_16F, CUDA_C_16F, CUDA_C_16F, CUDA_C_32F, CUDA_C_32F
auto constexpr cudaDataTypeI = CUDA_C_16F;
typedef half2 dataTypeI;
auto constexpr cudaDataTypeO = CUDA_C_16F;
typedef half2                  dataTypeO;
typedef thrust::complex<float> dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_C_32F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F;

#elif SCENARIO == 1  // CUDA_C_16F, CUDA_C_16F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F
auto constexpr cudaDataTypeI = CUDA_C_16F;
typedef half2 dataTypeI;
auto constexpr cudaDataTypeO = CUDA_C_32F;
typedef thrust::complex<float> dataTypeO;
typedef thrust::complex<float> dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_C_32F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F;
#endif

struct GenRand {
    __device__ dataTypeI operator( )( int const &idx ) {
        dataTypeI                                result;
        thrust::default_random_engine            randEng;
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard( idx );
        result.x = __float2half( uniDist( randEng ) );
        result.y = __float2half( uniDist( randEng ) );
        return ( result );
    }
};

struct setIdentity {
    int const m;
    setIdentity( int const &_m ) : m( _m ) {}
    __device__ dataTypeI operator( )( int const &idx ) {
        dataTypeI result;
        result.x          = __float2half( 0.0f );
        result.y          = __float2half( 0.0f );
        int const diagIdx = ( m + 1 );  // Since we are using complex half.
        if ( idx % ( diagIdx ) == 0 )
            result.x = __float2half( 1.0f );
        return ( result );
    }
};

template<typename Pointer>
__global__ void __launch_bounds__( threadsPerBlock )
    checkIdentity( int const n, int const m, dataTypeO const *d_C, Pointer d_p ) {
    for ( int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x ) {
        int const diagIdx = m + 1;
#if SCENARIO == 0
        if ( tid % ( diagIdx ) == 0 ) {  // If thread index is on the diagonal
            if ( __hge( fabsf( d_C[tid].x - __float2half( 1.0f ) ), __float2half( 1e-7f ) ) )
                *d_p = false;  // abs( d_C - 1.0f ) > 1e-7
        } else if ( __hge( d_C[tid].x, __float2half( 1e-7f ) ) )
            *d_p = false;
#elif SCENARIO == 1
        if ( tid % ( diagIdx ) == 0 ) {  // If thread index is on the diagonal
            if ( fabsf( d_C[tid].real( ) - 1.0f ) > 1e-7f )
                *d_p = false;
        } else if ( d_C[tid].real( ) > 1e-7f )
            *d_p = false;
#endif
    }
};

void LtSgemm( cublasLtHandle_t  ltHandle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int const &       m,
              int const &       n,
              int const &       k,
              dataTypeS const * alpha,
              int const &       sizeA,
              dataTypeI const * A,
              int const &       lda,
              int const &       sizeB,
              dataTypeI const * B,
              int const &       ldb,
              dataTypeS const * beta,
              int const &       sizeC,
              dataTypeO *       C,
              int const &       ldc,
              void *            workSpace,
              size_t            workSpaceSize ) {

    // The offset should start right after real data
    size_t planarOffsetA = ( sizeA * sizeof( dataTypeI ) ) / 2;
    size_t planarOffsetB = ( sizeB * sizeof( dataTypeI ) ) / 2;
    size_t planarOffsetC = ( sizeC * sizeof( dataTypeO ) ) / 2;

    cublasLtMatmulDesc_t   operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    cublasLtMatmulPreference_t preference = nullptr;

    dataTypeI *                   Atransform, *Btransform;
    dataTypeO *                   Ctransform;
    cublasLtMatrixTransformDesc_t transformDescI = nullptr, transformDescO = nullptr;
    cublasLtMatrixLayout_t        AtransformDesc = nullptr, BtransformDesc = nullptr, CtransformDesc = nullptr;

    // Allocate memory for transformed matrix
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &Atransform ), sizeA * sizeof( dataTypeI ) ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &Btransform ), sizeB * sizeof( dataTypeI ) ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &Ctransform ), sizeC * sizeof( dataTypeO ) ) );

    // Create preference handle; In general, extra attributes can be
    // used here to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C. However, for simplicity
    // here we assume A,B,C are always well aligned (e.g., directly
    // come from cudaMalloc)
    CUDA_RT_CALL( cublasLtMatmulPreferenceCreate( &preference ) );
    CUDA_RT_CALL( cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof( workSpaceSize ) ) );

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t
    // for details about defaults; here we just set the transforms for
    // A and B.
    CUDA_RT_CALL( cublasLtMatmulDescCreate( &operationDesc, cudaComputeType, cudaDataTypeC ) );
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof( transa ) ) );
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof( transa ) ) );

    // Create matrix descriptors for interleaved data. Not setting any extra attributes.
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &Adesc, cudaDataTypeI, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &Bdesc, cudaDataTypeI, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &Cdesc, cudaDataTypeO, m, n, ldc ) );

    // Create transform descriptor to convert interleaved to planar
    CUDA_RT_CALL( cublasLtMatrixTransformDescCreate( &transformDescI, cudaDataTypeC ) );
    CUDA_RT_CALL( cublasLtMatrixTransformDescCreate( &transformDescO, cudaDataTypeC ) );

    // Create matrix descriptors for planar data. Not setting any extra attributes.
    // Need to double check 3rd parameter
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &AtransformDesc, cudaDataTypeI, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &BtransformDesc, cudaDataTypeI, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &CtransformDesc, cudaDataTypeO, m, n, ldc ) );

    // Configure inputs and outputs to as planar layout
    CUDA_RT_CALL( cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &planarOffsetA, sizeof( planarOffsetA ) ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &planarOffsetB, sizeof( planarOffsetB ) ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &planarOffsetC, sizeof( planarOffsetC ) ) );

    // Create CUDA event to time the execution time of each algo
    cudaEvent_t  startEvent = nullptr, stopEvent = nullptr;
    cudaStream_t stream = nullptr;

#if TIME_TRANSFORM == 0
    CUDA_RT_CALL( cublasLtMatrixTransform(
        ltHandle, transformDescI, alpha, A, Adesc, beta, nullptr, nullptr, Atransform, AtransformDesc, stream ) );

    CUDA_RT_CALL( cublasLtMatrixTransform(
        ltHandle, transformDescI, alpha, B, Bdesc, beta, nullptr, nullptr, Btransform, BtransformDesc, stream ) );
#endif

    CUDA_RT_CALL( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventRecord( startEvent, stream ) );

    for ( int loop = 0; loop < kernelRepeats; loop++ ) {

#if TIME_TRANSFORM == 1
        // Transform interleaved data to planar
        CUDA_RT_CALL( cublasLtMatrixTransform(
            ltHandle, transformDescI, alpha, A, Adesc, beta, nullptr, nullptr, Atransform, AtransformDesc, stream ) );

        CUDA_RT_CALL( cublasLtMatrixTransform(
            ltHandle, transformDescI, alpha, B, Bdesc, beta, nullptr, nullptr, Btransform, BtransformDesc, stream ) );
#endif

        CUDA_RT_CALL( cublasLtMatmul( ltHandle,
                                      operationDesc,
                                      alpha,
                                      Atransform,
                                      AtransformDesc,
                                      Btransform,
                                      BtransformDesc,
                                      beta,
                                      Ctransform,
                                      CtransformDesc,
                                      Ctransform,
                                      CtransformDesc,
                                      nullptr,
                                      workSpace,
                                      workSpaceSize,
                                      stream ) );

#if TIME_TRANSFORM == 1
        // Transform planar to interleaved data in output matrix
        CUDA_RT_CALL( cublasLtMatrixTransform(
            ltHandle, transformDescO, alpha, Ctransform, CtransformDesc, beta, nullptr, nullptr, C, Cdesc, stream ) );
#endif
    }

    CUDA_RT_CALL( cudaEventRecord( stopEvent, stream ) );
    CUDA_RT_CALL( cudaEventSynchronize( stopEvent ) );
    float time;
    CUDA_RT_CALL( cudaEventElapsedTime( &time, startEvent, stopEvent ) );

#if TIME_TRANSFORM == 0
    // Transform planar to interleaved data in output matrix
    CUDA_RT_CALL( cublasLtMatrixTransform(
        ltHandle, transformDescO, alpha, Ctransform, CtransformDesc, beta, nullptr, nullptr, C, Cdesc, stream ) );
#endif

    printf(
#if IDENTITY
        "%d %d %d %d %d %d %d %0.2f ",
#else
        "%d %d %d %d %d %d %d %0.2f \n",
#endif
        m,
        n,
        k,
        cudaDataTypeI,
        cudaDataTypeI,
        cudaDataTypeO,
        cudaDataTypeC,
        time / kernelRepeats );

    // Descriptors are no longer needed as all GPU work was already enqueued.
    CUDA_RT_CALL( cublasLtMatmulPreferenceDestroy( preference ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Cdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Bdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Adesc ) );
    CUDA_RT_CALL( cublasLtMatmulDescDestroy( operationDesc ) );
    CUDA_RT_CALL( cudaFree( Atransform ) );
    CUDA_RT_CALL( cudaFree( Btransform ) );
    CUDA_RT_CALL( cudaFree( Ctransform ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( AtransformDesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( BtransformDesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( CtransformDesc ) );
    CUDA_RT_CALL( cublasLtMatrixTransformDescDestroy( transformDescI ) );
    CUDA_RT_CALL( cublasLtMatrixTransformDescDestroy( transformDescO ) );
    CUDA_RT_CALL( cudaEventDestroy( startEvent ) );
    CUDA_RT_CALL( cudaEventDestroy( stopEvent ) );
}

void calculate( int const &m, int const &n, int const &k, int &count, int const &square ) {

    dataTypeS alpha = 1.0f;
    dataTypeS beta  = 0.0f;
    int       lda = m, ldb = k, ldc = m;
    void *    d_workspace = nullptr;

    size_t sizeA     = m * k;
    size_t sizeB     = k * n;
    size_t sizeC     = m * n;
    size_t workspace = 4096;

    cublasLtHandle_t handle;

    /* Initialize cuBLASLt */
    CUDA_RT_CALL( cublasLtCreate( &handle ) );

    /* Allocate device memory for workspace */
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_workspace, workspace ) );

    /* Allocate device memory for the matrices */
    thrust::device_vector<dataTypeI> d_A( sizeA, __float2half2_rn( 0.0f ) );
    thrust::device_vector<dataTypeI> d_B( sizeB, __float2half2_rn( 0.0f ) );
#if SCENARIO == 0
    thrust::device_vector<dataTypeO> d_C( sizeC, __float2half2_rn( 0.0f ) );
#elif SCENARIO == 1
    thrust::device_vector<dataTypeO> d_C( sizeC, 0.0f );
#endif

    /* Retrieve raw pointer for device data */
    dataTypeI *d_A_ptr = thrust::raw_pointer_cast( &d_A[0] );
    dataTypeI *d_B_ptr = thrust::raw_pointer_cast( &d_B[0] );
    dataTypeO *d_C_ptr = thrust::raw_pointer_cast( &d_C[0] );

#if IDENTITY
    /* Generate identity matrix on device */
    thrust::transform( thrust::make_counting_iterator( 0 ),
                       thrust::make_counting_iterator( static_cast<int>( sizeA ) ),
                       d_A.begin( ),
                       setIdentity( m ) );
    thrust::transform( thrust::make_counting_iterator( 0 ),
                       thrust::make_counting_iterator( static_cast<int>( sizeB ) ),
                       d_B.begin( ),
                       setIdentity( m ) );
#else
    /* Generate random data on device */
    thrust::transform( thrust::make_counting_iterator( 0 ),
                       thrust::make_counting_iterator( static_cast<int>( sizeA ) ),
                       d_A.begin( ),
                       GenRand( ) );
    thrust::transform( thrust::make_counting_iterator( 0 ),
                       thrust::make_counting_iterator( static_cast<int>( sizeB ) ),
                       d_B.begin( ),
                       GenRand( ) );
#endif

    printf( "%d %d ", count, square );
    count++;

    LtSgemm( handle,
             CUBLAS_OP_N,
             CUBLAS_OP_N,
             m,
             n,
             k,
             &alpha,
             sizeA,
             d_A_ptr,
             lda,
             sizeB,
             d_B_ptr,
             ldb,
             &beta,
             sizeC,
             d_C_ptr,
             ldc,
             d_workspace,
             workspace );

#if IDENTITY
    /* Generate device vector to hold flag */
    thrust::device_vector<bool> d_p( 1, true );

    checkIdentity<<<sizeC / threadsPerBlock + 1, threadsPerBlock>>>( sizeC, m, d_C_ptr, d_p.data( ) );

    /* Copy device flag to host */
    thrust::host_vector<bool> h_p = d_p;

#if PRINT
    thrust::host_vector<dataTypeI> h_A = d_A;
    thrust::host_vector<dataTypeI> h_B = d_B;
    thrust::host_vector<dataTypeO> h_C = d_C;

    printf( "\n" );  // Formatting stdout

    for ( int a = 0; a < k; a++ ) {
        for ( int b = 0; b < n; b++ )
            printf( "{%0.1f %0.1f} ", __half2float( h_A[a * k + b].x ), __half2float( h_A[a * k + b].y ) );
        printf( "\n" );
    }
    printf( "\n" );

    for ( int a = 0; a < m; a++ ) {
        for ( int b = 0; b < k; b++ )
            printf( "{%0.1f %0.1f} ", __half2float( h_B[a * m + b].x ), __half2float( h_A[a * m + b].y ) );
        printf( "\n" );
    }
    printf( "\n" );

    for ( int a = 0; a < m; a++ ) {
        for ( int b = 0; b < n; b++ )
#if SCENARIO == 0
            printf( "{%0.1f %0.1f} ", __half2float( h_C[a * m + b].x ), __half2float( h_C[a * m + b].y ) );
#elif SCENARIO == 1
            printf( "{%0.1f, %0.1f} ", h_C[a * m + b].real( ), h_C[a * m + b].imag( ) );
#endif
        printf( "\n" );
    }
    printf( "\n" );
#endif

    if ( h_p[0] )
        printf( "Passed Identity Test\n" );
    else
        printf( "\n" );
#endif

    /* Destroy workspace */
    CUDA_RT_CALL( cudaFree( d_workspace ) );

    /* Shutdown */
    CUDA_RT_CALL( cublasLtDestroy( handle ) );
}

/* Main */
int main( int argc, char **argv ) {

    int dev {};
    CUDA_RT_CALL( cudaGetDevice( &dev ) );

    // Ensure GPU found is compute capability 7.0 or greater
    cudaDeviceProp deviceProp;
    CUDA_RT_CALL( cudaGetDeviceProperties( &deviceProp, dev ) );

    if ( deviceProp.major < 7 ) {
        throw std::runtime_error( "ERROR: This sample utilizes compute capability 7.0 or greater!" );
    }

    printf( "Computing matrix multiplication for the following types:\n" );
    printf( "Input Type (A):\t\t%s\n", "CUDA_C_16F" );
    printf( "Input Type (B):\t\t%s\n", "CUDA_C_16F" );
    printf( "Output Type (C):\t%s\n", ( cudaDataTypeO == 6 ) ? "CUDA_C_16F" : "CUDA_C_32F" );
    printf( "Scale Type:\t\t%s\n", "CUDA_C_32F" );
    printf( "Compute Type:\t\t%s\n\n", "CUDA_C_32F" );

    printf( "Run Square M N K A_Type B_Type C_Type Compute_Type Time(ms)\n" );

    int count  = 0;
    int square = 1;

#if IDENTITY
    // Identity for square matrices
#if PRINT
    for ( int m = 16; m <= 16; m *= 2 )
#else
    for ( int m = 16; m <= 8192; m *= 2 )
#endif
        calculate( m, m, m, count, square );

    printf( "\n" );  // For better readability stdout

#else

    // Compute matrices
    for ( int m = 512; m <= 8192; m *= 2 )
        for ( int k = 1024; k <= 4096; k *= 2 )
            calculate( m, m, k, count, square );

    printf( "\n" );  // For better readability stdout

    count  = 0;
    square = 0;

    // Compute non-square matrices
    for ( int m = 4096; m <= 32768; m *= 2 )
        for ( int n = 512; n <= 8192; n *= 2 )
            for ( int k = 8; k <= 128; k *= 2 )
                calculate( m, n, k, count, square );

    printf( "\n" );  // For better readability stdout

#endif

    return ( EXIT_SUCCESS );
}
