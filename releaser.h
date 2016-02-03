#define MS_PER_SEC 1000
#define NS_PER_MS 1000000
#define MAX_SIGNALS 11

struct Releaser_Args {
  int thread_id;
  int ms;
  pthread_cond_t *cond;
  FILE *ostream;
};

/**
 * Releases one job on a periodic schedule. 
 * Release period is specified in args->ms.
 * Job id is specified in args->cond.
 */
void * releaser(void *releaser_args);