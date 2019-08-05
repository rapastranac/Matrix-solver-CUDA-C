NVCC=nvcc

CUDAFLAGS= --gpu-architecture=sm_61
OPT= --device-c
RM= rm -f
EXN= -o a.out	#name of the executable
all:clean main

main: MatrixSolver.o
	${NVCC} ${EXN} ${CUDAFLAGS} MatrixSolver.o

MatrixSolver.o: MatrixSolver.cu
	${NVCC} ${CUDAFLAGS} ${OPT}  MatrixSolver.cu

clean:
	${RM} *.o *.out
