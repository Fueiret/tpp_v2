#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void fill_matrix(double *Matrix, size_t rows, size_t cols) {
    for (int i = 0; i < rows * cols; i++)
        Matrix[i] = (double)rand() / RAND_MAX * 10.0;
}

void init_distribution_parameters(int **rows_per_proc,
                                  int **elements_amount_per_proc,
                                  int **displs, int rows, int cols, int size)
{
    *rows_per_proc = (int*)malloc(size * sizeof(int));
    *elements_amount_per_proc = (int*)malloc(size * sizeof(int));
    *displs = (int*)malloc(size * sizeof(int));

    int base_rows_amount = rows / size;
    int remained_rows_amount = rows % size;
    int sum_rows = 0;

    for (int i = 0; i < size; i++) {
        (*rows_per_proc)[i] = base_rows_amount + (i < remained_rows_amount ? 1 : 0);
        (*elements_amount_per_proc)[i] = (*rows_per_proc)[i] * cols;
        (*displs)[i] = sum_rows * cols;
        sum_rows += (*rows_per_proc)[i];
    }
}

void matrix_multiplication(double *A, double *B, double *result,
                           size_t rows, size_t cols, double *norm)
{
    *norm = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols; k++) {
                sum += A[i * cols + k] * B[k * cols + j];
            }
            result[i * cols + j] = sum;
            *norm += sum * sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N, L;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    int *rows_per_proc = NULL;
    int *elements_amount_per_proc = NULL;
    int *displs = NULL;

    double start_time, end_time, elapsed_time;
    double local_norm = 0.0, total_norm = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (argc < 2) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        L = atoi(argv[1]);
        N = L * 10;
        printf("N = %d, L = %d, Size = %d\n", N, L, size);
    }

    // send N L to all process via MPI_Bcast[cite: 31]
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //  init distribution param
    init_distribution_parameters(&rows_per_proc, &elements_amount_per_proc, &displs, N, L, size);

    // alloc mem
    B = (double*)malloc(L * L * sizeof(double));

    if (rank == 0) {
        A = (double*)malloc(N * L * sizeof(double));
        C = (double*)malloc(N * L * sizeof(double));
        fill_matrix(A, N, L);
        fill_matrix(B, L, L);
    }

    // send B matrix
    MPI_Bcast(B, L * L, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // local matrix blocks for each proc
    int my_rows = rows_per_proc[rank];
    double *local_A = (double*)malloc(my_rows * L * sizeof(double));
    double *local_C = (double*)malloc(my_rows * L * sizeof(double));


    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Send A to each proc via MPI_Scatterv
    MPI_Scatterv(A, elements_amount_per_proc, displs, MPI_DOUBLE,
                 local_A, my_rows * L, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // local multiplication
    matrix_multiplication(local_A, B, local_C, my_rows, L, &local_norm);

    // collect result via MPI_Gatherv
    MPI_Gatherv(local_C, my_rows * L, MPI_DOUBLE,
                C, elements_amount_per_proc, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // collect norm via Reduce
    MPI_Reduce(&local_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // RESULTS
    if (rank == 0) {
        double total_euclidean_norm = sqrt(total_norm);
        printf("----------------------------------------\n");
        printf("Elapszed time %.6f sec\n", elapsed_time);
        printf("|C|^2       = %.6f\n", total_norm);
        printf("----------------------------------------\n");

        free(A);
        free(C);
    }

    // free mem
    free(B);
    free(local_A);
    free(local_C);
    free(rows_per_proc);
    free(elements_amount_per_proc);
    free(displs);

    MPI_Finalize();

    return 0;
}
