#ifndef PHYSICAL_ALLOCATION_H
#define PHYSICAL_ALLOCATION_H
// This file contains functions that can be used in order to allocate either
// CPU or GPU memory which is constrained to certain physical addresses.
extern "C" {

// Similar to AllocateInPhysicalRegion, but makes the opposite guarantee:
// Allocates memory, backed by cudaMallocHost (therefore returned addresses
// must be freed using cudaFree). Returns NULL on error. If an address is
// returned, this function guarantees it's physical address is *not* between
// base_address and max_address.
void* AllocateOutsidePhysicalRegion(uint64_t size, uint64_t base_address,
    uint64_t max_address);

// Returns a physical address for the given virtual address. This is not very
// efficient (requires opening and mapping /proc/self/pagemap), so don't rely
// on it in tight loops. Returns 0 on error.
uint64_t GetPhysicalAddress(void *virtual_address);

}
#endif  // PHYSICAL_ALLOCATION_H
