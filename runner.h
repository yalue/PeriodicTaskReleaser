#define MAX_ITERATIONS 10000
#define MS_PER_SEC 1000
#define NS_PER_MS 1000000

struct Runner_Args {
  int isActive;
  int ms;
  int datasize;
  pthread_mutex_t *mutex;
  FILE *ostream;
};

void * runner(void *runner_args);