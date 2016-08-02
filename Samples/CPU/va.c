#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "../gpusync.h"

float *vector_a, *vector_b, *result_vector;
int vector_bytes;

typedef struct {
  float *vector_a, *vector_b, *result_vector;
  int vector_bytes;
} ThreadData;

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

void* Initialize(int sync_level) {
  ThreadData *g = malloc(sizeof(ThreadData));
  if (!g) {
    printf("Error allocating thread data.\n");
    exit(1);
  }
  return g;
}

void MallocCPU(int numElements, void *thread_data) {
  ThreadData *g = (ThreadData *) thread_data;
  g->vector_bytes = numElements * sizeof(float);
  g->vector_a = (float *) malloc(g->vector_bytes);
  if (!g->vector_a) {
    printf("Failed allocating vector A.\n");
    exit(1);
  }
  g->vector_b = (float *) malloc(g->vector_bytes);
  if (!g->vector_b) {
    printf("Failed allocating vector B.\n");
    exit(1);
  }
  g->result_vector = (float *) malloc(g->vector_bytes);
  if (!result_vector) {
    printf("Failed allocating vector C.\n");
    exit(1);
  }
}

void MallocGPU(int numElements, void *thread_data) {
}

void CopyIn(int numElements, void *thread_data) {
  ThreadData *g = (ThreadData *) thread_data;
  RandomFill(g->vector_a, numElements);
  RandomFill(g->vector_b, numElements);
}

void Exec(int numElements, void *thread_data) {
  ThreadData *g = (ThreadData *) thread_data;
  Add(g->vector_a, g->vector_b, g->result_vector, numElements);
}

void CopyOut(void *thread_data) {
}

void FreeGPU(void *thread_data) {
}

void FreeCPU(void *thread_data) {
  ThreadData *g = (ThreadData *) thread_data;
  free(g->vector_a);
  g->vector_a = NULL;
  free(g->vector_b);
  g->vector_b = NULL;
  free(g->result_vector);
  g->result_vector = NULL;
}

void Finish(void *thread_data) {
  free(thread_data);
}
