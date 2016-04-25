#define ERROR_LEN 30
#define NS_PER_MS  1000000
#define NS_PER_SEC 1000000000

#include <time.h>

void error(char*);

long elapsed_ns(struct timespec *t1, struct timespec *t2);

char* format_time(struct timespec *t);

void timespec_offset(struct timespec *result, struct timespec *start, int ms);
