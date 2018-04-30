## Reduce choice
Your best bet is to start with the reduction example in the CUDA Samples. The scan example is also good for learning the principles of parallel computing on a throughput architecture.

That said, if you actually just want to use a reduction operator in your code then you should look at Thrust (calling from host, cross-platform) and CUB (CUDA GPU specific).

The performance of thrust is really bad, nearly no lifting performance added in this part.

There's no reason you can't use global memory for a reduction, the example code in the toolkit walks through various levels of optimisation but in each case the data starts in global memory.
Your code is inefficient (see the example in the toolkit for more details on the work efficiency!).
Your code is attempting to communicate between threads in different blocks without proper synchronisation; __syncthreads() only synchronises threads within a specific block and not across the different blocks (that would be impossible, at least generically, since you tend to oversubscribe the GPU meaning not all blocks will be running at a given time).

The last point is the most important. If a thread in block X wants to read data that is written by block Y then you need to break this across two kernel launches, that's why a typical parallel reduction **takes a multi-phase approach: reduce batches within blocks, then reduce between the batches.**


cudamallocManaged() get both pinned host and device memory mapped, may cause worse performance issue.
cuMemgetinfo used to get the avaliable information of current memory.

Use the memoryAync copy to parallel host code and device data transfer. 

Consider using the CUB device code.

## GPU memory limitation
So it really depends on the hardware you are using. Newer Tesla cards have between 12-24 GB of memory, while older cards might only have 2-4GB.

The common approach to getting around this problem is to break it up into batches. If, as you say, you are simply looking for differences, you could pass in small sections of your array at a time and operate on each section individually.

If your dataset is too large to fit on the device, there isn't a lot more you can do.

**Nothing is faster than using pointers, as I learned by the huge time difference between using thrust::device_vector<T> vs cudaMalloc and thrust::pointers for sorting.** CudaMallock  then try to use the pointer is the fastest way to reduce.

Use cupy to develop code and get the equivalant machine code & C++ code.

## Thread divergence
Recall that threads from a block are bundled into fixed-size warps for execution on a CUDA core, and threads within a warp must follow the same execution trajectory. All threads must execute the same instruction at the same time. In other words, threads cannot diverge.
#### if-then-else
The most common code construct that can cause thread divergence is branching for conditionals in an if-then-else statement. If some threads in a single warp evaluate to 'true' and others to 'false', then the 'true' and 'false' threads will branch to different instructions. Some threads will want proceed to the 'then' instruction, while others the 'else'.

Intuitively, we would think statements in then and else should be executed in parallel. However, because of the requirement that threads in a warp cannot diverge, this cannot happen. The CUDA platform has a workaround that fixes the problem, but has negative performance consequences.

When executing the if-then-else statement, the CUDA platform will instruct the warp to execute the then part first, and then proceed to the else part. While executing the then part, all threads that evaluated to false (e.g. the else threads) are effectively deactivated. When execution proceeds to the else condition, the situation is reversed. As you can see, the then and else parts are not executed in parallel, but in serial. This serialization can result in a significant performance loss.

数据依赖的if else divergence才有明显性能影响；blockIdx.x/threadIdx.x决定的ifelse分叉基本不影响性能。