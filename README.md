# Odd Even Sort using CUDA

## Concept

We generate 'N' threads where 'N' is number of elements of unsorted array. This 'N' cannot exceed the maximum number of
threads in a block. The code is implemented to work on only first block as stated in CUDA syntax <<<1,n>>>.

**NOTE:** For higher end Nvidia Graphics card the maximum number of threads in a block is 1024. For lower end, it is 512.

Out of these 'N' threads only N/2 threads are used in a phase due to the condition placed (line 39 and 68). First is even phase and next is odd phase. The names of phases are assigned so due to the index number.

The exiting codition for while(1) loop is that if no swappings have been done in even or odd phase then we conclude that array is sorted.
