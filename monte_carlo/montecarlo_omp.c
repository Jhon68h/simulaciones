#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
static unsigned int saltar(unsigned int estado, unsigned long k) {
    unsigned int a = 1103515245u, c = 12345u; 
    unsigned int A = 1u, C = 0u;              
    for (unsigned long n = 3 * k; n; n >>= 1) {
        if (n & 1) { A *= a; C = C * a + c; }
        c = c * a + c;
        a *= a;
    }
    return A * estado + C;
}

int main(int argc, char *argv[]) {
    long N = (argc > 1) ? atol(argv[1]) : 100000000L;
    if (argc > 2) omp_set_num_threads(atoi(argv[2]));
    const unsigned int semilla = 2201492u;     long dentro = 0;

    unsigned int estado;
    long siguiente = -1;  
    double t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:dentro) private(estado) firstprivate(siguiente) schedule(static)
    for (long i = 0; i < N; i++) {
        if (i != siguiente) estado = saltar(semilla, 2 * (unsigned long)i);
        siguiente = i + 1;
        double x = rand_r(&estado) / (double)RAND_MAX;
        double y = rand_r(&estado) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
            dentro++;
    }
      double t = omp_get_wtime() - t0;

    printf("Estimacion de pi: %.6f\n", 4.0 * dentro / (double)N);
    printf("Hilos: %d  Tiempo: %.4f s\n", omp_get_max_threads(), t);
    return 0;
}
