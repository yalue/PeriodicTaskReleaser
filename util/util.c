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

// Ignores the nsec portion of the timespec, so is accurate to within one second.
long elapsed_sec(struct timespec *t1, struct timespec *t2) {
  if (t2->tv_nsec < t1->tv_nsec) {
    return (long) ((t2->tv_sec - t1->tv_sec) - 1);
  } else {
    return (long) (t2->tv_sec - t1->tv_sec);
  } 
}

char* format_time(struct timespec *t) {
  static char time_buf[20];
  snprintf(time_buf, 20, "%d.%.9d", (int) t->tv_sec, (int) t->tv_nsec);
  return time_buf;
}

// calculates offset of offset_ns from time s and stores in result.
void timespec_offset(struct timespec *result, struct timespec *start, int offset_ns) {
  long sec = (long) start->tv_sec + offset_ns / NS_PER_SEC;
  long ns = (long) start->tv_nsec + (offset_ns % NS_PER_SEC);
  int overflow = ns / NS_PER_SEC;
  sec += overflow;
  ns %= NS_PER_SEC;
  result->tv_sec = sec;
  result->tv_nsec = ns;
}
