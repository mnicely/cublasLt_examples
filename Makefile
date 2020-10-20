NVCC	:=nvcc --cudart=static -ccbin g++ 
CFLAGS	:=-O3 -std=c++14

INC_DIR	:=
LIB_DIR	:=
LIBS	:=-lcublasLt

INT8_ARCH :=-gencode arch=compute_72,code=\"compute_72,sm_72\" \
	-gencode arch=compute_75,code=\"compute_75,sm_75\"

C16F_ARCH :=$(INT8_ARCH) \
	-gencode arch=compute_70,code=\"compute_70,sm_70\"

ARCHES	:=$(C16F_ARCH) \
	-gencode arch=compute_60,code=\"compute_60,sm_60\" \
	-gencode arch=compute_61,code=\"compute_61,sm_61\" \
	-gencode arch=compute_62,code=\"compute_62,sm_62\" 

SOURCES := cublasLt_C16F_TCs \
	cublasLt_INT8_TCs \
	cublasLt_search \
	cublasLt_sgemm

all: $(SOURCES)
.PHONY: all

cublasLt_INT8_TCs: cublasLt_INT8_TCs.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${INT8_ARCH} $^ -o $@ $(LIBS)

cublasLt_C16F_TCs: cublasLt_C16F_TCs.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${C16F_ARCH} $^ -o $@ $(LIBS)
	
cublasLt_search: cublasLt_search.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
cublasLt_sgemm: cublasLt_sgemm.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
clean:
	rm -f $(SOURCES)
