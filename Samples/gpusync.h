#ifndef SAMPLES_GPUSYNC_H
#define SAMPLES_GPUSYNC_H

void init(int sync_level);
void mallocCPU(int numElements);
void mallocGPU(int numElements);
void copyin(int numElements);
void exec(int numElements);
void copyout();
void freeGPU();
void freeCPU();
void finish();

#endif  // SAMPLES_GPUSYNC_H
