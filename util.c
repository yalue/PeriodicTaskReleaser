#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "util.h"

void error(char* error) {
  char buf[ERROR_LEN];
  strerror_r(errno, buf, ERROR_LEN);
  fprintf(stderr, "%s: %s\n", error, buf);
}