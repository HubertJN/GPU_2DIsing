#include "comparison.h"

// Comparison function for qsort
// Takes two values, each a pointer to a pointer. Extracts integer value pointer is pointing to, compares the values.
// Returns 1 if first value is greater than the second, returns -1 if first value is smaller than second. If values are equal returns 0.
int compare(const void *p_to_p_0, const void *p_to_p_1) {
    int i = **(int **)p_to_p_0;
    int j = **(int **)p_to_p_1;
    if(i > j)return 1;
    if(i < j)return -1;
    return 0;
}