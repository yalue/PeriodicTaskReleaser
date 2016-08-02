#ifndef SAMPLES_GPUSYNC_H
#define SAMPLES_GPUSYNC_H
// Each benchmark is roughtly divided into phases described by the following
// functions. Each function will exit the program on error. The thread_data
// value may be ignored by benchmarks which still use global state and don't
// support threading (in which case, Initialize will return NULL).

// Initializes the benchmark. Returns a pointer to a thread-local data
// structure. If the benchmark doesn't support threading, this returns NULL
// (and the NULL value should be passed to the remaining functions, too).
void* Initialize(int sync_level);

// Allocates CPU memory in the thread data..
void MallocCPU(int numElements, void *thread_data);

// Allocates GPU memory in the thread data.
void MallocGPU(int numElements, void *thread_data);

// Copies data to the GPU.
void CopyIn(int numElements, void *thread_data);

// Executes GPU kernel computations.
void Exec(int numElements, void *thread_data);

// Copies output data from the GPU.
void CopyOut(void *thread_data);

// Frees GPU memory in the thread data.
void FreeGPU(void *thread_data);

// Frees CPU memory in the thread data.
void FreeCPU(void *thread_data);

// Cleans up and frees the thread data structure itself, and performs any other
// necessary cleanup before the program can exit.
void Finish(void *thread_data);

#endif  // SAMPLES_GPUSYNC_H
