# Matrix solver CUDA C/C++
This solves the system of equations Ax=b using CUDA C/C++.

It reads an augmented matrix from a file in the same directory of the code.
Augmented matrix [N+1][N]=[A|b].

It prints in another file the matrix after elimination and the solution vector X.
{
  A
  X
}

This code implements Gaussian Elimination and backwards substitution. Since this a a triple loop solution, outer loop is solved with a single thread on the GPU to avoid copying data from host-to-device and vicerversa. Thus, dynamic parallelism strategy is applied.

It performs very well with matrices large anough to be benefited from GPU parallelisation. 

Elements under diagonal must be ignored, they are remainders of the procedure, they are actually Zeros.
