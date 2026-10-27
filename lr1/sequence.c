#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    long long N = 0;
    double partial_sum = 0.0;
    double total_sum = 0.0;
    double start_time = 0.0, end_time = 0.0;

    N = 1.06e10;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;


    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // if (rank == 0) {
    //     for (int p = 1; p < size; p++) {
    //         MPI_Send(&N, 1, MPI_LONG_LONG, 0, p, MPI_COMM_WORLD);
    //     }
    // }

    // 3. Вычисление локального диапазона индексов для каждого процесса (пункты б, в)
    long long elements_per_proc = N / size;        // базовое количество элементов на процесс
    long long remain = N % size;        // остаток от деления

    long long start_idx, end_idx;
    if (rank < remain) {
        start_idx = 2 + rank * (elements_per_proc + 1);
        end_idx = start_idx + (elements_per_proc + 1) - 1;
    }
    else {
        start_idx = 2 + rank * elements_per_proc + remain;
        end_idx = start_idx + elements_per_proc - 1;
    }


    // Partition sum computation
    for (long long i = start_idx; i <= end_idx; i++) {
        if (i % 2 == 0) {
            partial_sum += 1.0 / (double)(i * i - i);
        } else {
            partial_sum -= 1.0 / (double)(i * i - i);
        }
    }

    // 5. Пересылка частичных сумм нулевому процессу и сбор результата (пункты г, д)
    if (rank != 0) {
        // Отправка частичной суммы на процесс 0
        MPI_Send(&partial_sum, 1, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
    } else {
        total_sum = partial_sum;
        double recv_sum;

        for (int p = 1; p < size; p++) {
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, p, 999, MPI_COMM_WORLD, &status);
            total_sum += recv_sum;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        printf("----------------------------------------\n");
        printf("N = %lld\n", N);
        printf("size = %d\n", size);
        printf("S = %.12f\n", total_sum);
        printf("time elapsed = %f сек.\n", elapsed_time);
        printf("----------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}
