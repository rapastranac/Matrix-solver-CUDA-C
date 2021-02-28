# Matrix solver CUDA C/C++
This solves the system of equations Ax=b using CUDA C/C++

It reads an augmented matrix from a file in the same directory of the code.
Augmented matrix [N+1][N]=[A|b].

It prints in another file the matrix after elimination and the solution vector X.
{
  A
  X
}

Elements under diagonal must be ignored, they are remainders of the procedure, they are actually Zeros.
