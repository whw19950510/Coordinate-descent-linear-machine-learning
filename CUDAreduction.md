


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