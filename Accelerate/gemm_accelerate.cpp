// VECLIB_MAXIMUM_THREADS=1 ./gemm_accelerate
// g++ gemm_accelerate.cpp -o gemm_accelerate -O3 -framework Accelerate
#include <iostream>
#include <vector>
#include <chrono> // Para medir el tiempo
#include <cmath>  // Para pow (aunque no estrictamente necesario para 20000*20000)

// Incluir el framework Accelerate
// Esto es específico de macOS. En Xcode, asegúrate de que el proyecto
// esté configurado para enlazar con el framework Accelerate.
// Si compilas desde la línea de comandos, necesitarás -framework Accelerate
#include <Accelerate/Accelerate.h>

// Función para inicializar una matriz con valores aleatorios
void initialize_matrix(std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Valores entre -1.0 y 1.0
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        // Imprimir un mensaje de uso en el canal de error estándar
        std::cerr << "Uso: " << argv[0] << " <tamaño_de_la_matriz>" << std::endl;
        return 1; // Salir del programa con un código de error
    }

    int SIZE = std::stoi(argv[1]);
    //const int SIZE = 10000; // Dimensiones de las matrices (20000x20000)

    // Declarar las matrices
    // Las matrices se almacenan en formato de "row-major" por defecto en C++.
    // BLAS (y Accelerate) prefieren "column-major".
    // Para simplificar, podemos pensar que A y B se transponen conceptualmente al llamarlos,
    // o almacenar nuestros datos en column-major desde el principio.
    // Para cblas_sgemm, si usas CblasRowMajor, los parámetros se ajustan automáticamente.
    // Usaremos CblasRowMajor para que coincida con la forma en que C++ maneja std::vector<float>

    std::vector<float> A(SIZE * SIZE);
    std::vector<float> B(SIZE * SIZE);
    std::vector<float> C(SIZE * SIZE); // Matriz resultado

    // Inicializar las matrices A y B con valores aleatorios
    std::cout << "Inicializando matrices A y B de " << SIZE << "x" << SIZE << "..." << std::endl;
    initialize_matrix(A, SIZE, SIZE);
    initialize_matrix(B, SIZE, SIZE);
    std::cout << "Matrices inicializadas." << std::endl;

    // Medir el tiempo
    auto start_time = std::chrono::high_resolution_clock::now();

    // Realizar la multiplicación de matrices usando cblas_sgemm
    // C = alpha * A * B + beta * C
    // Aquí, alpha = 1.0, beta = 0.0, lo que significa C = A * B

    // Parámetros de cblas_sgemm:
    // CblasRowMajor: Indica que las matrices están en formato row-major (predeterminado de C++)
    // CblasNoTrans: No transponer A
    // CblasNoTrans: No transponer B
    // M: Número de filas de A y C (SIZE)
    // N: Número de columnas de B y C (SIZE)
    // K: Número de columnas de A y filas de B (SIZE)
    // alpha: Escalar para A * B (1.0f)
    // A: Puntero a la matriz A
    // LDA: Leading dimension de A (número de columnas si es CblasRowMajor, o número de filas si es CblasColMajor).
    //      Aquí, es el número de columnas de A, que es SIZE.
    // B: Puntero a la matriz B
    // LDB: Leading dimension de B (número de columnas si es CblasRowMajor). Aquí, es SIZE.
    // beta: Escalar para C (0.0f)
    // C: Puntero a la matriz C (resultado)
    // LDC: Leading dimension de C (número de columnas si es CblasRowMajor). Aquí, es SIZE.

    std::cout << "Realizando la multiplicación de matrices (GEMM) con Accelerate..." << std::endl;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                SIZE, SIZE, SIZE,
                1.0f, A.data(), SIZE,
                B.data(), SIZE,
                0.0f, C.data(), SIZE);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Multiplicación de matrices completada en " << elapsed_seconds.count() << " segundos." << std::endl;

    // Calcular GFLOPs
    // FLOPs para una multiplicación de matrices N x N por N x N es 2 * N^3
    double flops = 2.0 * pow(SIZE, 3);
    double gflops = flops / 1e9; // Convertir a GigaFLOPs

    if (elapsed_seconds.count() > 0) {
        double gflops_per_second = gflops / elapsed_seconds.count();
        std::cout << "Total de FLOPs: " << std::scientific << flops << std::endl;
        std::cout << "Total de GFLOPs: " << std::fixed << gflops << std::endl;
        std::cout << "Rendimiento: " << std::fixed << gflops_per_second << " GFLOPs/segundo" << std::endl;
    } else {
        std::cout << "El tiempo de ejecución fue demasiado corto para calcular GFLOPs/s de manera significativa." << std::endl;
    }
    std::cout << "----------------------------------------------------------------------------------------------------------------------------" << std::endl;

    // Opcional: Verificar un valor en la matriz resultante (para asegurarse de que no haya errores obvios)
    // Tenga en cuenta que para matrices tan grandes, verificar elementos específicos
    // no es práctico y puede ser engañoso si no se hace correctamente.
    // Aquí solo mostramos el primer elemento.
    // std::cout << "Elemento C[0][0]: " << C[0] << std::endl;

    return 0;
}
