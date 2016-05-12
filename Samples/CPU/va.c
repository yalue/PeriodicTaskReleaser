#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "../gpusync.h"

float *vector_a, *vector_b, *result_vector;
int vector_bytes;

// Sets vector c = a + b
static void Add(float *a, float *b, float *c, int length) {
  int i;
  for (i = 0; i < length; i++) {
    c[i] = a[i] + b[i];
  }
}

// Fills the given vector with random floats between 0 and 1.
static void RandomFill(float *v, int length) {
  int i;
  for (i = 0; i < length; i++) {
    v[i] = ((float) rand()) / ((float) RAND_MAX);
  }
}

static void PrintVector(float *v, int length) {
  int i;
  for (i = 0; i < length; i++) {
    printf("%.04f ", v[i]);
  }
  printf("\n");
}

void init(int sync_level) {
}

void mallocCPU(int numElements) {
  vector_bytes = numElements * sizeof(float);
  vector_a = (float *) malloc(vector_bytes);
  if (!vector_a) {
    printf("Failed allocating vector A.\n");
    exit(1);
  }
  vector_b = (float *) malloc(vector_bytes);
  if (!vector_b) {
    printf("Failed allocating vector B.\n");
    exit(1);
  }
  result_vector = (float *) malloc(vector_bytes);
  if (!result_vector) {
    printf("Failed allocating vector C.\n");
    exit(1);
  }
  if (!mlockall(MCL_CURRENT)) {
    error("Failed to lock code pages");
    exit(1);
  }
}

void mallocGPU(int numElements) {
}

void copyin(int numElements) {
  RandomFill(vector_a, numElements);
  RandomFill(vector_b, numElements);
}

void exec(int numElements) {
  Add(vector_a, vector_b, result_vector, numElements);
}

void copyout() {
}

void freeGPU() {
}

void freeCPU() {
  free(vector_a);
  vector_a = NULL;
  free(vector_b);
  vector_b = NULL;
  free(result_vector);
  result_vector = NULL;
}

void finish() {
}
