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
  return ((long) (t2->tv_sec - t1->tv_sec) * NS_PER_SEC +
      (long) (t2->tv_nsec - t1->tv_nsec));
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
  static char time_buf[23];
  snprintf(time_buf, 22, "%ld.%09ld", (long) t->tv_sec, (long) t->tv_nsec);
  return time_buf;
}

// calculates offset of offset_ns from time s and stores in result.
void timespec_offset(struct timespec *result, struct timespec *start, long long offset_ns) {
  time_t sec = (time_t) start->tv_sec + (time_t) (offset_ns / NS_PER_SEC);
  long long ns = (long long) start->tv_nsec + (long long) (offset_ns % NS_PER_SEC);
  long overflow = ns / NS_PER_SEC;
  sec += overflow;
  ns %= NS_PER_SEC;
  result->tv_sec = sec;
  result->tv_nsec = (long) ns;
}

/**
 * Returns a negative value if t1 is less than t2, zero if they are the same,
 * and a positive value otherwise.
 */
int timespec_compare(struct timespec *t1, struct timespec *t2) {
  int sec = t1->tv_sec - t2->tv_sec;
  int nsec = t1->tv_nsec - t2->tv_nsec;
  return (sec == 0) ? nsec : sec;
}

