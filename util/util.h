#define ERROR_LEN 30
#define NS_PER_MS  1000000
#define NS_PER_SEC 1000000000L

#include <time.h>

#define CURRENT_TIME(ref)\
if (clock_gettime(CLOCK_REALTIME, ref)) {\
  error("Error getting time.");\
  exit(EXIT_FAILURE);\
}

void error(char*);

long elapsed_ns(struct timespec *t1, struct timespec *t2);

long elapsed_sec(struct timespec *t1, struct timespec *t2);

char* format_time(struct timespec *t);

void timespec_offset(struct timespec *result, struct timespec *start, long long ms);

int timespec_compare(struct timespec *t1, struct timespec *t2);

