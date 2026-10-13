#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


void fill_matrix(double *Matrix, size_t rows, size_t cols) {
    for (int i = 0; i < rows * cols; i++)
        Matrix[i] = (double)rand() / RAND_MAX * 10.0;

    return;
}


void init_distribution_parameters(int **rows_per_proc,
                                  int **elements_amount_per_proc,
                                  int **displs, int rows, int cols, int NumProc)
{
    *rows_per_proc = (int*)malloc(NumProc * sizeof(int));
    *elements_amount_per_proc = (int*)malloc(NumProc * sizeof(int));
    *displs = (int*)malloc(NumProc * sizeof(int));

    int base_rows_amount = rows / NumProc;
    int remained_rows_amount = rows % NumProc;
    int sum_rows = 0;

    for (int i = 0; i < NumProc; i++) {
        (*rows_per_proc)[i] = base_rows_amount + (i < remained_rows_amount ? 1 : 0);
        (*elements_amount_per_proc)[i] = (*rows_per_proc)[i] * cols;
        (*displs)[i] = sum_rows * cols;
        sum_rows += (*rows_per_proc)[i];
    }

    return;
}


void matrix_multiplication(double *A, double *B, double *result,
                           size_t rows, size_t cols, double *norm)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;

            for (int k = 0; k < cols; k++)
                sum += A[i * cols + k] * B[k * cols + j];

            result[i * cols + j] = sum;
            *norm += sum * sum;
        }
    }

    return;
}


int main(int argc, char *argv[]) {
    int MyID, NumProc;
    int N, L;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    // double *local_A = NULL;
    // double *local_C = NULL;

    int *rows_per_proc = NULL;
    int *elements_amount_per_proc = NULL;
    int *displs = NULL;

    double start_time, end_time, elapsed_time;
    double local_norm = 0.0, total_norm = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

    // get matrix size
    if (MyID == 0) {
        L = atoi(argv[1]);
        N = L * 10;
        printf(": N = %d, L = %d, rank = %d\n", N, L, NumProc);

        init_distribution_parameters(rows_per_proc, elements_amount_per_proc,
                                     displs, N, L, NumProc);
    }

    // send N L
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for matrices
    if (MyID == 0) {
        A = (double*)malloc(N * L * sizeof(double));
        B = (double*)malloc(L * L * sizeof(double));
        C = (double*)malloc(N * L * sizeof(double));

        fill_matrix(A, N, L);
        fill_matrix(B, L, L);

        MPI_Bcast(B, L * L, MPI_DOUBLE, 0, MPI_COMM_WORLD); // send B for all
    }

    // alloc memory for local parts for A и C
    double *local_A = (double*)malloc(proc_rows_amount * L * sizeof(double));
    double *local_C = (double*)malloc(proc_rows_amount * L * sizeof(double));


    // START COMPUTATION
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Send A from root to others proc
    MPI_Scatterv(A, elements_amount_per_proc, displs, MPI_DOUBLE,
                 local_A, rows_per_proc[MyID] * L, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // local multiplication local_A (my_rows x L) \times B (L x L)
    matrix_multiplication(local_A, B, local_C, rows_per_proc[MyID], L, &local_norm);

    // Collect result of C matrix on root process
    MPI_Gatherv(local_C, rows_per_proc[MyID] * L, MPI_DOUBLE,
                C, elements_amount_per_proc, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Collect norm via MPI_Reduce
    MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    if (MyID == 0) {
        printf("Elapsed time: %.6f sec\n", elapsed_time);
        printf("|C|^2 = %.2f\n", total_norm_sq);

        free(A);
        free(B);
        free(C);
    }

    free(local_A);
    free(local_C);
    free(rows_per_proc);
    free(sendcounts);
    free(displs);

    MPI_Finalize();

    return 0;
}
