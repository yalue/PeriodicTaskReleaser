#define MAX_ITERATIONS 10000
#define VECTOR_ADD 0
#define MATRIX_MUL 1

struct Runner_Args {
  int isActive;
  int ms;
  int datasize;
  int function;
  pthread_mutex_t *mutex;
  pthread_barrier_t *barrier;
  FILE *ostream;
};

void * runner(void *runner_args);