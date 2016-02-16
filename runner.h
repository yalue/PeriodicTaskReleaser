#define MAX_ITERATIONS 10000
#define MS_PER_SEC 1000
#define NS_PER_MS 1000000
#define MAX_WAIT_TIME 5 // 5 seconds

struct Runner_Args {
  int isActive;
  int thread_id;
  int ms;
  int datasize;
  pthread_cond_t *cond;
  pthread_mutex_t *mutex;
  FILE *ostream;
};

void * runner(void *runner_args);