void mm_init(int sync_level);
void mm_mallocHost(int numElements);
void mm_cudaMalloc(int numElements);
void mm_copyin(int numElements);
void mm_exec(int numElements);
void mm_copyout();
void mm_cudaFree();
void mm_freeHost();
void mm_finish();
