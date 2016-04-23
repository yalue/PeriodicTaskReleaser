#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include "util.h"

void error(char* error) {
  char buf[ERROR_LEN];
  strerror_r(errno, buf, ERROR_LEN);
  fprintf(stderr, "%s: %s\n", error, buf);
}

long elapsed_ns(struct timespec *t1, struct timespec *t2) {
  return (long) ((t2->tv_sec - t1->tv_sec) * NS_PER_SEC +
      (t2->tv_nsec - t1->tv_nsec));
}

char* format_time(struct timespec *t) {
  static char time_buf[20];
  snprintf(time_buf, 20, "%d.%.9d", (int) t->tv_sec, (int) t->tv_nsec);
  return time_buf;
}

void timespec_offset(struct timespec *result, struct timespec *s, int ms) {
  long sec = (long) s->tv_sec + ms / MS_PER_SEC;
  long ns = (long) s->tv_nsec + (ms % MS_PER_SEC) * NS_PER_MS;
  int overflow = ns / NS_PER_SEC;
  sec += overflow;
  ns %= NS_PER_SEC;
  result->tv_sec = sec;
  result->tv_nsec = ns;
}
