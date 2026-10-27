/*   Uso: ./calor <N> <T> <K> */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CHECK(llamada) do { \
    cudaError_t e = (llamada); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "Error CUDA en linea %d: %s\n", __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while (0)

__global__ void paso_jacobi(const double *u, double *v, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;   /* columna: hilos seguidos leen celdas seguidas */
    int i = blockIdx.y * blockDim.y + threadIdx.y;   /* fila */
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1)    /* guarda: solo celdas interiores */
        v[i * N + j] = 0.25 * (u[(i - 1) * N + j] + u[(i + 1) * N + j]
                             + u[i * N + j - 1] + u[i * N + j + 1]);
}

double segundos(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    double T = (argc > 2) ? atof(argv[2]) : 100.0;
    int K = (argc > 3) ? atoi(argv[3]) : 2000;
    size_t bytes = (size_t)N * N * sizeof(double);

    double *u = (double *)malloc(bytes);
    double *v = (double *)malloc(bytes);
    double *res = (double *)malloc(bytes);   /* resultado de la GPU */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            u[i * N + j] = (i == 0 || j == 0 || i == N - 1 || j == N - 1) ? T : 0.0;
    for (int i = 0; i < N * N; i++) v[i] = u[i];

    CHECK(cudaFree(0));   /* inicia el contexto aqui para no medir ese costo */
    double t0 = segundos();
    double *du, *dv;
    CHECK(cudaMalloc(&du, bytes));
    CHECK(cudaMalloc(&dv, bytes));
    CHECK(cudaMemcpy(du, u, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dv, v, bytes, cudaMemcpyHostToDevice));

    dim3 bloque(16, 16);
    dim3 malla((N + 15) / 16, (N + 15) / 16);
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int k = 0; k < K; k++) {
        paso_jacobi<<<malla, bloque>>>(du, dv, N);
        CHECK(cudaGetLastError());
        double *tmp = du; du = dv; dv = tmp;   /* solo se cambian los punteros */
    }
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));

    CHECK(cudaMemcpy(res, du, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(du));
    CHECK(cudaFree(dv));
    double t_gpu = segundos() - t0;

    t0 = segundos();
    for (int k = 0; k < K; k++) {
        for (int i = 1; i < N - 1; i++)
            for (int j = 1; j < N - 1; j++)
                v[i * N + j] = 0.25 * (u[(i - 1) * N + j] + u[(i + 1) * N + j]
                                     + u[i * N + j - 1] + u[i * N + j + 1]);
        double *tmp = u; u = v; v = tmp;
    }
    double t_cpu = segundos() - t0;

    double dif = 0.0;
    for (int i = 0; i < N * N; i++)
        dif = fmax(dif, fabs(res[i] - u[i]));

    printf("N=%d T=%g K=%d\n", N, T, K);
    printf("centro: CPU %.6e  GPU %.6e\n", u[(N / 2) * N + N / 2], res[(N / 2) * N + N / 2]);
    printf("max|gpu-cpu|: %.3e  (%s)\n", dif, dif <= 1e-12 * T ? "correcto" : "REVISAR");
    printf("T_cpu: %.4f s  T_gpu: %.4f s  T_kernels: %.4f s\n", t_cpu, t_gpu, ms / 1000.0);
    printf("S_total: %.2f  S_kernels: %.2f  f: %.3f\n",
           t_cpu / t_gpu, t_cpu / (ms / 1000.0), 1.0 - (ms / 1000.0) / t_gpu);

    free(u); free(v); free(res);
    return 0;
}
