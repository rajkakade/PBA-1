#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <limits.h>

// Knuth-Morris-Pratt (KMP) algorithm
void computeLPSArray(char* pattern, int m, int* lps) {
    int len = 0; // Length of the previous longest prefix suffix
    lps[0] = 0;  // LPS[0] is always 0
    int i = 1;

    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    int lps[m];

    // Preprocess the pattern (calculate lps array)
    computeLPSArray(pattern, m, lps);

    int i = 0; // Index for text
    int j = 0; // Index for pattern

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        if (pattern[j] == text[i]) {
            j++;
            if (j == m) {
                printf("KMP: Pattern found at index %d\n", i - j + 1);
                j = lps[j - 1];
            }
        } else if (j != 0) {
            j = lps[j - 1];
            i--; // Need to recheck the same character
        }
    }
}

// Boyer-Moore algorithm
void badCharHeuristic(char* pattern, int m, int badChar[CHAR_MAX]) {
    for (int i = 0; i < CHAR_MAX; i++)
        badChar[i] = -1;

    for (int i = 0; i < m; i++)
        badChar[(int)pattern[i]] = i;
}

void BoyerMooreSearch(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    int badChar[CHAR_MAX];

    // Preprocess the pattern (calculate bad character table)
    badCharHeuristic(pattern, m, badChar);

    int s = 0; // Shift of the pattern relative to text
    while (s <= (n - m)) {
        int j = m - 1;

        while (j >= 0 && pattern[j] == text[s + j])
            j--;

        if (j < 0) {
            printf("Boyer-Moore: Pattern found at index %d\n", s);
            s += (s + m < n) ? m - badChar[(int)text[s + m]] : 1;
        } else {
            s += ((j - badChar[(int)text[s + j]]) > 1) ? (j - badChar[(int)text[s + j]]) : 1;
        }
    }
}

// Main function for comparative study of KMP and Boyer-Moore with OpenMP and MPI
int main(int argc, char *argv[]) {
    char text[] = "ababcabcabababd";
    char pattern[] = "ababd";

    int rank, size;
    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the size of the communicator

    // Divide the work in the case of MPI. Only rank 0 can print the result.
    if (rank == 0) {
        printf("Text: %s\n", text);
        printf("Pattern: %s\n\n", pattern);
    }

    clock_t start, end;
    double cpu_time_used;

    // Parallelize the KMP Search with OpenMP
    start = clock();
    if (rank == 0) {
        printf("Running Knuth-Morris-Pratt Algorithm...\n");
    }
    KMPSearch(text, pattern);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    if (rank == 0) {
        printf("Time taken by KMP: %f seconds\n\n", cpu_time_used);
    }

    // Parallelize the Boyer-Moore Search with OpenMP
    start = clock();
    if (rank == 0) {
        printf("Running Boyer-Moore Algorithm...\n");
    }
    BoyerMooreSearch(text, pattern);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    if (rank == 0) {
        printf("Time taken by Boyer-Moore: %f seconds\n", cpu_time_used);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
