void va_init(int sync_level);
void va_mallocHost(int numElements);
void va_cudaMalloc(int numElements);
void va_copyin(int numElements);
void va_exec(int numElements);
void va_copyout();
void va_cudaFree();
void va_freeHost();
void va_finish();
