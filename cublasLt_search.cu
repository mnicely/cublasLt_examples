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
 * to do an exhaustive search to obtain valid algorithms for GEMM.
 *
 * There are four scenarios performing half and single precision GEMMS.
 *
 * CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
 * CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F
 * CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
 * CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
 *
 * There is one scenario performing a complex, single precision GEMM.
 *
 * CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F
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
 * This example has been tested with compute capability 6.0 and greater.
 */

/* Includes, system */
#include <algorithm>
#include <cstdio>

/* Includes, cuda & thrust */
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

auto constexpr algoCombinations = 600;
auto constexpr algoIds          = 20;
auto constexpr printAlgos       = 1;
auto constexpr kernelRepeats    = 10;
auto constexpr threadsPerBlock  = 1024;

// Key
// *I = Input
// *O = Output
// *s = Scale
// *C = Compute

// Set data types
#if SCENARIO == 0  // CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
auto constexpr cudaDataTypeI = CUDA_R_16F;
typedef half dataTypeI;
auto constexpr cudaDataTypeO = CUDA_R_16F;
typedef half dataTypeO;
auto constexpr cudaDataTypeS = CUDA_R_16F;
typedef half dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_R_16F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_16F;

#elif SCENARIO == 1  // CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F
auto constexpr cudaDataTypeI = CUDA_R_16F;
typedef half dataTypeI;
auto constexpr cudaDataTypeO = CUDA_R_16F;
typedef half dataTypeO;
auto constexpr cudaDataTypeS = CUDA_R_32F;
typedef float dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_R_32F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F;

#elif SCENARIO == 2  // CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
auto constexpr cudaDataTypeI = CUDA_R_16F;
typedef half dataTypeI;
auto constexpr cudaDataTypeO = CUDA_R_32F;
typedef float dataTypeO;
auto constexpr cudaDataTypeS = CUDA_R_32F;
typedef float dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_R_32F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F;

#elif SCENARIO == 3  // CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
auto constexpr cudaDataTypeI = CUDA_R_32F;
typedef float dataTypeI;
auto constexpr cudaDataTypeO = CUDA_R_32F;
typedef float dataTypeO;
auto constexpr cudaDataTypeS = CUDA_R_32F;
typedef float dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_R_32F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F;

#elif SCENARIO == 4  // CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F
auto constexpr cudaDataTypeI = CUDA_C_32F;
typedef thrust::complex<float> dataTypeI;
auto constexpr cudaDataTypeO = CUDA_C_32F;
typedef thrust::complex<float> dataTypeO;
auto constexpr cudaDataTypeS = CUDA_C_32F;
typedef thrust::complex<float> dataTypeS;
auto constexpr cudaDataTypeC   = CUDA_C_32F;
auto constexpr cudaComputeType = CUBLAS_COMPUTE_32F;

#endif

/* Structure to store information about different run trials */
typedef struct {
    cublasLtMatmulAlgo_t      algo;
    cublasStatus_t            status;
    float                     time;
    size_t                    workspaceSize;  // actual memory workspace needed
    cublasMath_t              mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int                       customOption;
    float                     wavesCount;
} customMatmulPerf_t;

/* CAUTION : must match cublasLtMatmulTile_t */
const char *const matmulTileName[] = {
    "UNDEF",  "8x8",    "8x16",    "16x8",   "8x32",   "16x16",   "32x8",    "8x64",   "16x32",
    "32x16",  "64x8",   "32x32",   "32x64",  "64x32",  "32x128",  "64x64",   "128x32", "64x128",
    "128x64", "64x256", "128x128", "256x64", "64x512", "128x256", "256x128", "512x64",
};

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure( const customMatmulPerf_t &perf, int const &m, int const &n, int const &k ) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, mathMode;

    const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof( algoId ), nullptr );
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof( tile ), nullptr );
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof( numSplitsK ), nullptr );
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof( reductionScheme ), nullptr );
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof( swizzle ), nullptr );
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof( customOption ), nullptr );
    cublasLtMatmulAlgoCapGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CAP_MATHMODE_IMPL, &mathMode, sizeof( mathMode ), nullptr );

    /* Calculate GFLOPS */
    double timeAvg = ( perf.time * 1e-3 ) / kernelRepeats;  // Convert to seconds, then divide by loops
#if SCENARIO < 4
    double gflop = ( 2 * static_cast<unsigned long long int>( m * n ) * k ) * 1e-9;  // Real
#else
    double gflop = ( 8 * static_cast<unsigned long long int>( m * n ) * k ) * 1e-9;  // Complex
#endif

    printf(
#if IDENTITY
        "%d %d (%s) %d %d %d %d %d %f %d %d %f %0.0f ",
#else
        "%d %d (%s) %d %d %d %d %d %f %d %d %f %0.0f \n",
#endif
        algoId,
        tile,
        matmulTileName[tile],
        numSplitsK,
        reductionScheme,
        swizzle,
        customOption,
        perf.status,
        perf.time,
        static_cast<int>( perf.workspaceSize ),
        mathMode,
        perf.wavesCount,
        gflop / timeAvg );
}

static inline bool time_compare( const customMatmulPerf_t &perf_a, const customMatmulPerf_t &perf_b ) {
    return ( ( perf_a.status == CUBLAS_STATUS_SUCCESS ) && ( perf_a.time < perf_b.time ) );
}

struct GenRand {
    __device__ dataTypeI operator( )( int const &idx ) {
        thrust::default_random_engine            randEng;
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard( idx );
        return ( static_cast<dataTypeI>( uniDist( randEng ) ) );
    }
};

struct setIdentity {
    int const m;
    setIdentity( int const &_m ) : m( _m ) {}
    __device__ dataTypeI operator( )( int const &idx ) {
        dataTypeI out     = 0.0f;
        int const diagIdx = m + 1;
        if ( idx % ( diagIdx ) == 0 )
            out = 1.0f;
        return ( out );
    }
};

template<typename Pointer>
__global__ void __launch_bounds__( threadsPerBlock )
    checkIdentity( int const n, int const m, dataTypeO const *d_C, Pointer d_p ) {
    for ( int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x ) {
        int const diagIdx = m + 1;
#if SCENARIO < 4
        if ( tid % ( diagIdx ) == 0 ) {  // If thread index is on the diagonal
            if ( fabsf( __half2float( d_C[tid] ) - 1.0f ) > 1e-7f )
                *d_p = false;
        } else if ( __half2float( d_C[tid] ) > 1e-7f )
            *d_p = false;
#else
        if ( tid % ( diagIdx ) == 0 ) {  // If thread index is on the diagonal
            if ( fabsf( d_C[tid].real( ) - 1.0f ) > 1e-7f )
                *d_p = false;
        } else if ( d_C[tid].real( ) > 1e-7f )
            *d_p = false;
#endif
    }
};

static cublasStatus_t customMatmulRun( cublasLtHandle_t       ltHandle,  // to get the capabilities (required a GPU)
                                       cublasLtMatmulDesc_t   operationDesc,
                                       void const *           alpha, /* host or device pointer */
                                       void const *           A,
                                       cublasLtMatrixLayout_t Adesc,
                                       void const *           B,
                                       cublasLtMatrixLayout_t Bdesc,
                                       void const *           beta, /* host or device pointer */
                                       void const *           C,
                                       cublasLtMatrixLayout_t Cdesc,
                                       void *                 D,
                                       cublasLtMatrixLayout_t Ddesc,
                                       cublasLtMatmulAlgo_t const &algo,
                                       void *                      workSpace,
                                       size_t                      workSpaceSizeInBytes,
                                       customMatmulPerf_t &        perfResults,
                                       cudaStream_t                stream,
                                       cudaEvent_t &               startEvent,
                                       cudaEvent_t &               stopEvent ) {

    cublasLtMatmulHeuristicResult_t heurResult;

    /* Looping over the Algo */
    cublasStatus_t algoStatus =
        cublasLtMatmulAlgoCheck( ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult );

    if ( algoStatus == CUBLAS_STATUS_SUCCESS ) {
        if ( heurResult.workspaceSize <= workSpaceSizeInBytes ) {
            cudaError_t err, err1, err2, err3;
            err = cudaEventRecord( startEvent, stream );
            for ( int loop = 0; loop < kernelRepeats; loop++ ) {
                cublasStatus_t oneRunStatus = cublasLtMatmul( ltHandle,
                                                              operationDesc,
                                                              alpha, /* host or device pointer */
                                                              A,
                                                              Adesc,
                                                              B,
                                                              Bdesc,
                                                              beta, /* host or device pointer */
                                                              C,
                                                              Cdesc,
                                                              D,
                                                              Ddesc,
                                                              &algo,
                                                              workSpace,
                                                              workSpaceSizeInBytes,
                                                              stream );
                if ( oneRunStatus != CUBLAS_STATUS_SUCCESS ) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord( stopEvent, stream );
            err2 = cudaEventSynchronize( stopEvent );
            float time;
            err3 = cudaEventElapsedTime( &time, startEvent, stopEvent );
            if ( ( err != cudaSuccess ) || ( err1 != cudaSuccess ) || ( err2 != cudaSuccess ) ||
                 ( err3 != cudaSuccess ) ) {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            if ( algoStatus == CUBLAS_STATUS_SUCCESS ) {
                perfResults.algo          = algo;
                perfResults.time          = time / kernelRepeats;  // Average time
                perfResults.workspaceSize = heurResult.workspaceSize;
                perfResults.wavesCount    = heurResult.wavesCount;
            }
        } else {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
        }
    }
    return algoStatus;
}

// Sample wrapper running through multiple algo and config attributes combination for single precision gemm using
// cublasLt low-level API
void LtGemmSearch( cublasLtHandle_t  ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int const &       m,
                   int const &       n,
                   int const &       k,
                   void const *      alpha, /* host pointer */
                   void const *      A,
                   int const &       lda,
                   void const *      B,
                   int const &       ldb,
                   void const *      beta, /* host pointer */
                   void *            C,
                   int const &       ldc,
                   void *            workSpace,
                   size_t            workSpaceSize ) {

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cudaEvent_t  startEvent = nullptr, stopEvent = nullptr;
    cudaStream_t stream = nullptr;

    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = { 2, 3, 4, 5, 6, 8, 12, 16, 32 };

    // Let try a fixed number of combinations
    int                algoCount = 0;
    int                nbAlgoIds = 0;
    int                algoIdA[algoIds];
    customMatmulPerf_t perfResults[algoCombinations];

    cudaDataType_t computeType = cudaDataTypeC;
    cudaDataType_t scaleType   = cudaDataTypeS;
    cudaDataType_t Atype       = cudaDataTypeI;
    cudaDataType_t Btype       = cudaDataTypeI;
    cudaDataType_t Ctype       = cudaDataTypeO;

    CUDA_RT_CALL( cublasLtMatmulPreferenceCreate( &preference ) );
    CUDA_RT_CALL( cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof( workSpaceSize ) ) );

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CUDA_RT_CALL( cublasLtMatmulDescCreate( &operationDesc, cudaComputeType, computeType ) );
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof( transa ) ) );
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof( transa ) ) );

    // Create matrix descriptors. We are good with the details here so no need to set any extra attributes
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate(
        &Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutCreate( &Cdesc, Ctype, m, n, ldc ) );

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType = Atype = Btype = Ctype = Dtype =
    // CUDA_R_32F)
    CUDA_RT_CALL( cublasLtMatmulAlgoGetIds(
        ltHandle, cudaComputeType, scaleType, Atype, Btype, Ctype, Ctype, algoIds, algoIdA, &nbAlgoIds ) );

    // Create CUDA event to time the execution time of each algo
    CUDA_RT_CALL( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );

    // Loop over the Algo IDs
    for ( int idx = 0; ( idx < nbAlgoIds ) && ( algoCount < algoCombinations ); idx++ ) {
        cublasLtMatmulAlgo_t algo;
        size_t               sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status = cublasLtMatmulAlgoInit(
            ltHandle, cudaComputeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo );
        if ( status != CUBLAS_STATUS_SUCCESS ) {
            continue;
        }
        // Query the tiles enums supported by that algo
        CUDA_RT_CALL(
            cublasLtMatmulAlgoCapGetAttribute( &algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten ) );
        int  nbTiles = int( sizeWritten / sizeof( int ) );
        int *tileA   = new int[nbTiles == 0 ? 1 : nbTiles];
        if ( nbTiles == 0 ) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles  = 1;
        }

        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        CUDA_RT_CALL( cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof( int ) * nbTiles, &sizeWritten ) );
        CUDA_RT_CALL( cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof( splitkSupport ), &sizeWritten ) );
        CUDA_RT_CALL( cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof( redMask ), &sizeWritten ) );
        CUDA_RT_CALL( cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof( swizzlingMax ), &sizeWritten ) );
        CUDA_RT_CALL( cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof( customOptionMax ), &sizeWritten ) );

        /* Loop over the different tiles */
        for ( int tileIdx = 0; tileIdx < nbTiles; tileIdx++ ) {
            /* Loop over the different custom option if any */
            for ( int customOption = 0; customOption <= customOptionMax; customOption++ ) {
                CUDA_RT_CALL( cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof( customOption ) ) );
                /* Loop over the CTAs swizzling support */
                for ( int k = 0; k <= swizzlingMax; k++ ) {
                    int splitK_trial = 0;
                    if ( splitkSupport ) {
                        splitK_trial += sizeof( splitKSequenceA ) / sizeof( splitKSequenceA[0] );
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in addition to the case where
                    // splitK is not enabled
                    for ( int l = 0; ( l < ( 1 + splitK_trial ) ) && ( algoCount < algoCombinations ); l++ ) {
                        /* Setup attribute of the algo to run */
                        CUDA_RT_CALL( cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof( tileA[tileIdx] ) ) );
                        int splitK_val = 0;
                        int redScheme  = CUBLASLT_REDUCTION_SCHEME_NONE;
                        CUDA_RT_CALL( cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof( splitK_val ) ) );
                        CUDA_RT_CALL( cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof( k ) ) );
                        CUDA_RT_CALL( cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof( int ) ) );

                        if ( l > 0 ) {  // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            CUDA_RT_CALL( cublasLtMatmulAlgoConfigSetAttribute( &algo,
                                                                                CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                                &splitKSequenceA[l - 1],
                                                                                sizeof( splitKSequenceA[l - 1] ) ) );
                            /* Going over all the reduction scheme  */
                            for ( redScheme = 1; redScheme < static_cast<int>( CUBLASLT_REDUCTION_SCHEME_MASK ) &&
                                                 ( algoCount < algoCombinations );
                                  redScheme = redScheme << 1 ) {
                                if ( redScheme & redMask ) {
                                    CUDA_RT_CALL(
                                        cublasLtMatmulAlgoConfigSetAttribute( &algo,
                                                                              CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                              &redScheme,
                                                                              sizeof( redScheme ) ) );

                                    status                        = customMatmulRun( ltHandle,
                                                              operationDesc,
                                                              alpha, /* host or device pointer */
                                                              A,
                                                              Adesc,
                                                              B,
                                                              Bdesc,
                                                              beta, /* host or device pointer */
                                                              C,
                                                              Cdesc,
                                                              C,
                                                              Cdesc,
                                                              algo,
                                                              workSpace,
                                                              workSpaceSize,
                                                              perfResults[algoCount],
                                                              stream,
                                                              startEvent,
                                                              stopEvent );
                                    perfResults[algoCount].status = status;
                                    if ( status == CUBLAS_STATUS_SUCCESS ) {
                                        algoCount++;
                                    }
                                }  // end if
                            }      // end for
                        } else {   // Non-splitK case
                            /* if user preference is ok with workspace */
                            if ( algoCount < algoCombinations ) {
                                status                        = customMatmulRun( ltHandle,
                                                          operationDesc,
                                                          alpha, /* host or device pointer */
                                                          A,
                                                          Adesc,
                                                          B,
                                                          Bdesc,
                                                          beta, /* host or device pointer */
                                                          C,
                                                          Cdesc,
                                                          C,
                                                          Cdesc,
                                                          algo,
                                                          workSpace,
                                                          workSpaceSize,
                                                          perfResults[algoCount],
                                                          stream,
                                                          startEvent,
                                                          stopEvent );
                                perfResults[algoCount].status = status;
                                if ( status == CUBLAS_STATUS_SUCCESS )
                                    algoCount++;
                            }
                        }
                    }  // end l
                }      // end k
            }          // end customOption
        }              // end tileIdx
        delete[] tileA;
    }  // end idx

    // Sort the results per run duration
    std::sort( perfResults, perfResults + algoCount, time_compare );
    // Print timing and perf details of the fastest combinations
    for ( int i = 0; i < printAlgos; i++ )
        printPerfStructure( perfResults[i], m, n, k );

    // Descriptors are no longer needed as all GPU work was already enqueued
    CUDA_RT_CALL( cublasLtMatmulPreferenceDestroy( preference ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Cdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Bdesc ) );
    CUDA_RT_CALL( cublasLtMatrixLayoutDestroy( Adesc ) );
    CUDA_RT_CALL( cublasLtMatmulDescDestroy( operationDesc ) );
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
    thrust::device_vector<dataTypeI> d_A( sizeA, 0.0f );
    thrust::device_vector<dataTypeI> d_B( sizeB, 0.0f );
    thrust::device_vector<dataTypeO> d_C( sizeC, 0.0f );

    /* Retrieve raw pointer for device data */
    dataTypeI *d_A_ptr = thrust::raw_pointer_cast( &d_A[0] );
    dataTypeI *d_B_ptr = thrust::raw_pointer_cast( &d_B[0] );
    dataTypeO *d_C_ptr = thrust::raw_pointer_cast( &d_C[0] );

#if IDENTITY
    /* Generate identity matrix on device */
    thrust::counting_iterator<int> idx( 0 );
    thrust::transform( idx, idx + sizeA, d_A.begin( ), setIdentity( m ) );
    thrust::transform( idx, idx + sizeB, d_B.begin( ), setIdentity( m ) );
#else
    /* Generate random data on device */
    thrust::counting_iterator<int> idx( 0 );
    thrust::transform( idx, idx + sizeA, d_A.begin( ), GenRand( ) );
    thrust::transform( idx, idx + sizeB, d_B.begin( ), GenRand( ) );
#endif

    printf( "%d %d %d %d %d %d %d %d %d ",
            count,
            square,
            m,
            n,
            k,
            cudaDataTypeI,
            cudaDataTypeI,
            cudaDataTypeO,
            cudaDataTypeC );
    count++;

    LtGemmSearch( handle,
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
                  workspace );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

#if IDENTITY
    /* Generate device vector to hold flag */
    thrust::device_vector<bool> d_p( 1, true );

    checkIdentity<<<sizeC / threadsPerBlock + 1, threadsPerBlock>>>( sizeC, m, d_C_ptr, d_p.data( ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    /* Copy device flag to host */
    thrust::host_vector<bool> h_p = d_p;

    if ( h_p[0] )
        printf( "Passed Identity Test\n" );
    else
        printf( "\n" );

#if PRINT
    thrust::host_vector<dataTypeI> h_A = d_A;
    thrust::host_vector<dataTypeI> h_B = d_B;
    thrust::host_vector<dataTypeO> h_C = d_C;

    for ( int a = 0; a < k; a++ ) {
        for ( int b = 0; b < n; b++ )
#if SCENARIO < 4
            printf( "%0.1f ", static_cast<float>( h_A[a * m + b] ) );
#else
            printf( "{%0.1f %0.1f} ", h_A[a * m + b].real( ), h_A[a * m + b].imag( ) );
#endif
        printf( "\n" );
    }
    printf( "\n" );

    for ( int a = 0; a < m; a++ ) {
        for ( int b = 0; b < k; b++ )
#if SCENARIO < 4
            printf( "%0.1f ", static_cast<float>( h_B[a * k + b] ) );
#else
            printf( "{%0.1f %0.1f} ", h_B[a * k + b].real( ), h_B[a * m + b].imag( ) );
#endif
        printf( "\n" );
    }
    printf( "\n" );

    for ( int a = 0; a < m; a++ ) {
        for ( int b = 0; b < n; b++ )
#if SCENARIO < 4
            printf( "%0.1f ", static_cast<float>( h_C[a * m + b] ) );
#else
            printf( "{%0.1f %0.1f} ", h_C[a * m + b].real( ), h_C[a * m + b].imag( ) );
#endif
        printf( "\n" );
    }
    printf( "\n" );
#endif

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

    printf( "Run Type M N K A_Type B_Type C_Type Compute_Type Algo_ID Tile_Idx Tile_Size Split_K Reduce Swizzle Custom "
            "Status Time(ms) Workspace Math_Mode Waves GFLOPS\n" );

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
    for ( int m = 512; m <= 4096; m *= 2 )
        for ( int k = 1024; k <= 4096; k *= 2 )
            calculate( m, m, k, count, square );

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
