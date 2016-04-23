void init(int sync_level);
void mallocCPU(int numElements);
void mallocGPU(int numElements);
void copyin(int numElements);
void exec(int numElements);
void copyout();
void freeGPU();
void freeCPU();
void finish();
