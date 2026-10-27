#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define L_VAL 0.01


double get_c(double u) {
    return 1.0 / (2.25e-3 - 6.08e-10 * u * u);
}

double get_rho(double u) {
    return 7860.0 + 41500.0 / u;
}

double get_lambda(double u) {
    return 1.45 + 2.3e-2 * u - 2.0e-6 * u * u;
}


double initial_condition(double xt, double yt) {
    return 100.0 * (3.0 + yt) * (2.0 - xt);
}

double boundary_mu1(double yt, double tau_t) {
    return 200.0 * (3.0 + yt) + tau_t * (400.0 - 100.0 * cos(M_PI * yt));
}

double boundary_mu2(double yt, double tau_t) {
    return 100.0 * (3.0 + yt) + tau_t * (300.0 * sin((1.0 + yt) * M_PI / 2.0) + 100.0);
}

double boundary_mu3(double xt, double tau_t) {
    return 300.0 * (2.0 - xt) + 100.0 * tau_t * (3.0 + xt);
}

double boundary_mu4(double xt, double tau_t) {
    return 400.0 * (2.0 - xt) + 100.0 * tau_t * (5.0 - 4.0 * xt);
}


// ==========================================
// 3. КОММУНИКАЦИИ: ОБМЕН ПРИЗРАЧНЫМИ СТРОКАМИ
// ==========================================
void exchange_ghost_cells(double *u, int local_rows, int cols, int rank, int size) {
    MPI_Status status;

    // 1. Обмен с верхним соседом (rank - 1)
    if (rank > 0) {
        // Отправляем верхнюю рабочую строку вверх, принимаем верхнюю призрачную строку (индекс 0) от верхнего соседа
        MPI_Sendrecv(&u[1 * cols], cols, MPI_DOUBLE, rank - 1, 1,
                     &u[0 * cols], cols, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
    }

    // 2. Обмен с нижним соседом (rank + 1)
    if (rank < size - 1) {
        // Отправляем нижнюю рабочую строку вниз, принимаем нижнюю призрачную строку (индекс local_rows + 1) от нижнего соседа
        MPI_Sendrecv(&u[local_rows * cols], cols, MPI_DOUBLE, rank + 1, 0,
                     &u[(local_rows + 1) * cols], cols, MPI_DOUBLE, rank + 1, 1,
                     MPI_COMM_WORLD, &status);
    }
}


void update_grid(double *u, double *u_next, int local_rows, int I, int cols, 
                 double dx, double dy, double dtau, int global_start_row, double current_tau) 
{
    for (int r = 1; r <= local_rows; r++) {
        int g_r = global_start_row + (r - 1);
        double xt = (double)g_r * dx / L_VAL;

        for (int c = 1; c < I; c++) {
            double yt = (double)c * dy / L_VAL;

            // Применение краевых условий по пространству
            if (g_r == 0) { u_next[r * cols + c] = boundary_mu3(xt, current_tau); continue; }
            if (g_r == I) { u_next[r * cols + c] = boundary_mu4(xt, current_tau); continue; }
            if (c == 0)   { u_next[r * cols + c] = boundary_mu1(yt, current_tau); continue; }
            if (c == I)   { u_next[r * cols + c] = boundary_mu2(yt, current_tau); continue; }

            double u_ij = u[r * cols + c];
            double c_val = get_c(u_ij);
            double rho_val = get_rho(u_ij);

            // Коэффициенты теплопроводности на полуцелых слоях
            double lam_ip = get_lambda(0.5 * (u[(r + 1) * cols + c] + u_ij));
            double lam_im = get_lambda(0.5 * (u[(r - 1) * cols + c] + u_ij));
            double lam_jp = get_lambda(0.5 * (u[r * cols + (c + 1)] + u_ij));
            double lam_jm = get_lambda(0.5 * (u[r * cols + (c - 1)] + u_ij));

            // Аппроксимация производных[cite: 54]
            double d2x = (lam_ip * (u[(r + 1) * cols + c] - u_ij) - lam_im * (u_ij - u[(r - 1) * cols + c])) / (dx * dx);
            double d2y = (lam_jp * (u[r * cols + (c + 1)] - u_ij) - lam_jm * (u_ij - u[r * cols + (c - 1)])) / (dy * dy);

            // Пересчет значения сетки на следующий слой[cite: 54]
            u_next[r * cols + c] = u_ij + (dtau / (c_val * rho_val)) * (d2x + d2y);
        }
    }
}


void gather_and_save(double *local_u, int local_rows, int I, int cols, 
                     int rank, int size, int *sendcounts, int *displs, 
                     double dx, double dy, const char *filename) 
{
    double *global_u = NULL;
    if (rank == 0) {
        global_u = (double*)malloc((I + 1) * cols * sizeof(double));
    }

    MPI_Gatherv(&local_u[1 * cols], local_rows * cols, MPI_DOUBLE,
                global_u, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        save_field_to_file(filename, global_u, I, dx, dy);
        free(global_u);
    }
}


void save_field_to_file(const char *filename, double *global_u, int I, double dx, double dy) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Can't open file %s\n", filename);
        return;
    }

    for (int i = 0; i <= I; i++) {
        double x = i * dx;
        for (int j = 0; j <= I; j++) {
            double y = j * dy;
            fprintf(f, "%.6f %.6f %.6f\n", x, y, global_u[i * (I + 1) + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Saved to file: %s\n", filename);
}


int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            printf("mpirun -np <p> %s <I (число узлов)> <T (время счета)>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int I = atoi(argv[1]);
    double T = atof(argv[2]);

    double dx = L_VAL / I;
    double dy = L_VAL / I;

    // Шаг по времени из условия устойчивости
    double k_coeff = 2.5 / 1000.0; 
    double dtau = 0.9 / (2.0 * k_coeff * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
    int N_steps = (int)(T / dtau) + 1;
    dtau = T / N_steps;

    // Геометрическая декомпозиция строк по процессам
    int base_rows_amount = (I + 1) / size;
    int remained_rows_amount = (I + 1) % size;
    int local_rows_amount = base_rows_amount + (rank < remained_rows_amount ? 1 : 0);

    int total_local_rows = local_rows_amount + 2;
    int cols = I + 1;

    double *u = (double*)calloc(total_local_rows * cols, sizeof(double));
    double *u_next = (double*)calloc(total_local_rows * cols, sizeof(double));

    // Расчет глобального индекса начала для процесса
    int global_start_row = 0;
    for (int p = 0; p < rank; p++) {
        global_start_row += base_rows_amount + (p < remained_rows_amount ? 1 : 0);
    }

    // Инициализация начального слоя
    for (int r = 0; r < local_rows; r++) {
        int g_r = global_start_row + r;
        double xt = (double)g_r * dx / L_VAL;
        for (int c = 0; c <= I; c++) {
            double yt = (double)c * dy / L_VAL;
            u[(r + 1) * cols + c] = initial_condition(xt, yt);
            u_next[(r + 1) * cols + c] = u[(r + 1) * cols + c];
        }
    }

    gather_and_save(u, local_rows, I, cols, rank, size, sendcounts, displs, dx, dy, "temperature_t0.txt");

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Основной цикл по времени
    for (int n = 0; n < N_steps; n++) {
        double current_tau = n * dtau;

        // 1. Обмениваемся границами
        exchange_ghost_cells(u, local_rows, cols, rank, size);

        // 2. Считаем новый слой сетки
        update_grid(u, u_next, local_rows, I, cols, dx, dy, dtau, global_start_row, current_tau);

        // 3. Меняем указатели местами (без копирования памяти)
        double *temp = u;
        u = u_next;
        u_next = temp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    gather_and_save(u, local_rows, I, cols, rank, size, sendcounts, displs, dx, dy, "temperature_t0.txt");

    if (rank == 0) {
        printf("----------------------------------------\n");
        printf("I = %d, T = %.2f, Size = %d\n", I, T, size);
        printf("Time elapsed = %.4f sec\n", end_time - start_time);
        printf("----------------------------------------\n");
    }

    free(u);
    free(u_next);
    MPI_Finalize();
    return 0;
}
