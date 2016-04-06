void memtest_init(int sync_level, int numElements);
void memtest_alloc(int numElements);
void memtest_copyin(int numElements);
void memtest_copyout(int numElements);
void memtest_cudafree();
void memtest_cleanup();
