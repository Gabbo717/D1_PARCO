#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>
#include <immintrin.h> 
#include <stdbool.h>
#include <xmmintrin.h>
#include <atomic>

using namespace std;

void initializeMatrix(float** M, int n) {
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            M[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

bool checkSymVectorized(float** M, int n) {
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


void matTransposeImpVectorized(float** M, float** T, int n){
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

int main(int argc, char* argv[]){
    if(argc != 2){
        cerr << "Usage: ./Name_Runnable <matrix_size>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);

    if(n <= 0){
        cerr << "Error: Matrix size must be a positive integer." << endl;
        return 1;
    }

    float** M = new float*[n];
    float** T = new float*[n];
    for(int i = 0; i < n; i++){
        M[i] = (float*)aligned_alloc(32, n * sizeof(float));  
        T[i] = (float*)aligned_alloc(32, n * sizeof(float));
    }

    srand(time(0));

    initializeMatrix(M, n);

    auto start = chrono::high_resolution_clock::now();
    bool isSymmetric = checkSymVectorized(M,n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> checkSymTimeVectorized = end - start;

    start = chrono::high_resolution_clock::now();
    matTransposeImpVectorized(M, T, n);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> transposeTimeVectorized = end - start;


    cout << checkSymTimeVectorized.count() << endl;
    cout << transposeTimeVectorized.count() << endl;

    for(int i = 0; i < n; i++){
        free(M[i]);
        free(T[i]);
    }
    delete[] M;
    delete[] T;

    return 0;
}
