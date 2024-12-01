#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>
#include <immintrin.h> 
#include <stdbool.h>
#include <xmmintrin.h>
#include <omp.h>

using namespace std;

void initializeMatrix(float** M, int n) {
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            M[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

bool checkSym_Sequential(float** M, int n){
    bool isSymm = true;
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if(M[i][j] != M[j][i]){
                isSymm = false;
            }
        }
    }
    return isSymm;
}

bool checkSym_Imp(float** M, int n){
    bool isSymm = true;
    for (int i = 0; i < n; i++) {  
        for (int j = i + 1; j < n; j += 4) {  
            int remaining = n - j;
            if (remaining < 4) {
                for (int k = 0; k < remaining; k++) {
                    if (M[i][j + k] != M[j + k][i]) {
                        isSymm = false;
                        break;
                    }
                }
                break;
            }

            __m128 row = _mm_loadu_ps(&M[i][j]);
            __m128 col = _mm_set_ps(M[j][i], M[j + 1][i], M[j + 2][i], M[j + 3][i]);
            __m128 comparison = _mm_cmpneq_ps(row, col);

            if (_mm_movemask_ps(comparison) != 0) {
                isSymm = false;
            }
        }
    }
    return isSymm;
}

bool checkSymOMP_Expl(float** M, int n){
    bool isSymmetric = true;

    #pragma omp parallel for collapse(2) shared(isSymmetric)
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if(M[i][j] != M[j][i]){
                #pragma omp atomic write
                isSymmetric = false;
            }
        }
    }
    return isSymmetric;
}



void matTranspose_Sequential(float** M, float** T, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            T[i][j] = M[j][i];
        }
    }
}

void matTranspose_Imp(float** M, float** T, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j += 8){
            int remaining = n-j;
            if(remaining >= 8){
                __m256 row = _mm256_loadu_ps(&M[j][i]);
                _mm256_storeu_ps(&T[i][j], row);
            }else{
                for(int k = 0; k < remaining; k++){
                    T[i][j + k] = M[j + k][i];
                }
            }
        }
    }
}

void matTransposeOMP_Explicit(float** M, float** T, int n){
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            T[i][j] = M[j][i];
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        cerr << "Usage: ./<runnable_name> <matrix_size>, <threads_number>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if(n <= 0) {
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1; 
    }

    if(num_threads <= 0) {
        cerr << "Error: num_threads must be a positive integer. " << endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    float** M_Seq = new float*[n];
    float** T_Seq = new float*[n];
    for(int i = 0; i < n; i++) {
        M_Seq[i] = new float[n];
        T_Seq[i] = new float[n];
    }

    float** M_Impl = new float*[n];
    float** T_Impl = new float*[n];
    for(int i = 0; i < n; i++) {
        M_Impl[i] = (float*)aligned_alloc(32, n * sizeof(float));  
        T_Impl[i] = (float*)aligned_alloc(32, n * sizeof(float));
    }

    float** M_Expl = new float*[n];
    float** T_Expl = new float*[n];
    for(int i = 0; i < n; i++) {
        M_Expl[i] = new float[n];
        T_Expl[i] = new float[n];
    }

    initializeMatrix(M_Seq, n);
    initializeMatrix(M_Impl, n);
    initializeMatrix(M_Expl, n);

    auto start = chrono::high_resolution_clock::now();
    bool isSymmetric_Sequential = checkSym_Sequential(M_Seq, n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> checkSymTime_Sequential = end - start;

    start = chrono::high_resolution_clock::now();
    bool isSymmetric_Implicit = checkSym_Imp(M_Impl, n);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> checkSymTime_Implicit = end - start;

    start = chrono::high_resolution_clock::now();
    bool isSymmetric_Explicit = checkSymOMP_Expl(M_Expl, n);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> checkSymTime_Explicit = end - start;

    start = std::chrono::high_resolution_clock::now();
    matTranspose_Sequential(M_Seq, T_Seq, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transposeTime_Sequential = end - start;

    start = std::chrono::high_resolution_clock::now();
    matTranspose_Imp(M_Impl, T_Impl, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transposeTime_Implicit = end - start;

    start = std::chrono::high_resolution_clock::now();
    matTransposeOMP_Explicit(M_Expl, T_Expl, n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transposeTime_Explicit = end - start;

    auto calculateImprovement = [](double seqTime, double parTime) -> double {
        return ((seqTime - parTime) / seqTime) * 100.0;
    };

    cout << "CheckSym Elaboration Time (Sequential): " << checkSymTime_Sequential.count() << " seconds\n\n";
    cout << "CheckSym Elaboration Time (Implicit Par): " << checkSymTime_Implicit.count() << " seconds\n";
    cout << "CheckSym Elaboration Time (Explicit Par): " << checkSymTime_Explicit.count() << " seconds\n";

    cout << "Matrix Transpose Elaboration Time (Sequential): " << transposeTime_Sequential.count() << " seconds\n\n";
    cout << "Matrix Transpose Elaboration Time (Implicit Par): " << transposeTime_Implicit.count() << " seconds\n";
    cout << "Implicit Par Improvement: " << calculateImprovement(transposeTime_Sequential.count(), transposeTime_Implicit.count()) << " %\n\n";
    cout << "Matrix Transpose Elaboration Time (Explicit Par): " << transposeTime_Explicit.count() << " seconds\n";
    cout << "Explicit Par Improvement: " << calculateImprovement(transposeTime_Sequential.count(), transposeTime_Explicit.count()) << " %\n";


    for(int i = 0; i < n; i++){
        delete[] M_Seq[i];
        delete[] T_Seq[i];
    }
    delete[] M_Seq;
    delete[] T_Seq;
    
    for(int i = 0; i < n; i++){
        delete[] M_Impl[i];
        delete[] T_Impl[i];
    }
    delete[] M_Impl;
    delete[] T_Impl;

    for(int i = 0; i < n; i++){
        delete[] M_Expl[i];
        delete[] T_Expl[i];
    }
    delete[] M_Expl;
    delete[] T_Expl;

    return 0;
    
}

