# DeviceToDevice memory copies performance (And philosophy of why to do that?) TBD

Usually, copyng data from one pointer to another, shouldn't be done without any processing in between (before writing to the destination pointer). 

For instance, imagine you have 5 consecutive kernels, that read a pointer, do some computations, and store the results in another pointer of the same size. 
- The result of each kernel is the input of the next one.
- All kernels are called using the same stream, and the same CPU thread, that calls a cudaStreamSynchronize at the end.
- The calls and the cudaStreamSynchronize are inside a loop. So the whole proces is repeated several times, including the GPU-CPU synchronization.

Why would anyone copy the results of one kernel to another pointer, and use that other pointer as an input for the next kernel? Right?

Well, now imagine the following:
- Given the original pipeline (K=Kernel, D=Data pointer): while(true) {Input->K1->D1->K2->D2->K3->D3->K4->D5->K5->Output->Synch}
- Input contains different data at every iteration, and includes an upload (cudaMemcpyAsynch HostToDevice).
- Now, D2, is not only read by K3, but also: D2->K6->D6->K7->Output2->Synch.
- Additionally, this branch of execution, is enqueued in a different CUDA Stream, by a different CPU thread that synchronices with that second Stream.

Then, how do you do that? Well, you need the following:
- The first CPU thread should some how tell the other thread that new data is abailable.
- Ideally, once this is comunicated to the other thread, the first thread should continue. Ideally, the first thread should not wait for anything, just anotate that new data is abailable and continue.
- Yo need to make sure, that the first thread does not enqueue K2 on the next iteration, before K6 has finished on the current iteration.

And now, how you ensure that last condition?
- You can synchronize the CPU threads between them, and after the cudaStreamSynchronize, make them wait for each other in a Barrier fashion.
- Or, it may happen, that K6 and K7, are very heavy kernels, that last a lot longuer than K1, K2, K3, K4 and K5 together, and you want to keep enqueueing kernels with the first thread, to ensure there are no empty spots in the execution timeline, and that you expose all the possible parallelism to the runtime (overlapping of transfers and computation and kernel overlapping). In that case, you may prefer to implement a producer consumer model, with a small buffer, between the two threads.
- If you want to execute the second pipeline only once every 4 iterations, the producer consumer model allows the second thread to schedule the execution of its kernels, when it finished with all the previous iterations, without forcing the first thread to wait for it.

So, in this case, you may want to copy D2, to one of the pointers of the buffer, and use that pointer as an input to K6.

But still, you can avoid this copy, by implementing a memory pool, where the first CPU thread asks to the pool for a pointer, to use it as output of K2, and then put it (place the pointer, so copy the address, not the entire data), in the buffer used to implement the producer consumer model, and pass it as input to K3 at the same time.

For this to work the memory pool needs to know which pointers are not used, or being used for writing, or being used for reading by anyone. So that the pointer will never be used to write and read at the same time, and will be assigned and released as input or output, correctly.

This has a problem. It's time consuming to implement, and make sure it works always correctly.

So that is why, as a fast and temporary improvement, I want to check if I can improve the NVIDIA provided cudaMemcpy DtoD copy call, with my own copy kernels. Also, it will help me clarify some aspects of the Device memory organization on Maxwell and Pascal architecture (things like maximum bandwith per Warp, interactions between Warps when reading Device memory, specially when there are a lot of them compiting to access Device memory, L1/L2 cache capabilities, effects on data locality due to thread block scheduling when there are more or less thread blocks allocated for the kernel, etc...). This can be endless exploration so I will probably close the project without all the questions solved.

cudaMemcpy() with the flag cudaMemcpyDeviceToDevice, launches a Kernel that performs the copy of data from one GPU pointer to another.

Why did I think that this copies are a Kernel? Because of the behavior observable in NSIGHT Visual Studio Timelines. This copies overlap slightly with other kernels launched on other CUDA Streams, and only at the begining and the end (as soon as there are less thread blocks active from one kernel, some frome the next kernel are scheduled).

Additionally, I have been replacing copies by handling pointers or modifying kernels, to internally update buffers, instead fo doing so with cudaMemcpy. And i ovserved, that this kernels do the writing almost for free, about 0.05ms compared to 0.33 ms for the same amount of data with cudaMemcpy. Of course, cudaMemcpy, is reading again the data, not only writing, but the time spent is more than twice.

This kernel that did the writing, was not reading and writing 32 bit elements, but 128bit elements (uint4). The kernel is memory bound, and using uint4 data elements instead of uint gave some speed-up. This makes me wonder about the total possible memory bandwith per warp. Also, if you look to other memory bound kernel optimizations, like Reduction kernel, reducing the total amount of threadblocks concurrently active, improves the performance. This obviously ensures a better locality, since you won't have the first thread block and the last thread block asking for very distant data at the same time, in the same SM (just recall an Streaming Multiprocessor can have several thread blocks executing at the same time, and they may be processing further or closer data positions).

But how is Device memory organiced? In blocks of 32 elements of 32 bits? or 64 elements?
Looking at CUDA toolkit documentation: "Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions."

So 128-byte is the same as 32 elements of 32 bits each, or 4 bytes. But why is it faster reading 128bit elements? Is due to the cache?

Doing some tests and looking into assembler code may give some answers.

# Quite relevant finding:
In NVIDIA official documentation (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization), you can find the following:
Two commands from different streams cannot run concurrently if any one of the following operations is issued in-between them by the host thread:

- a page-locked host memory allocation,
- a device memory allocation,
- a device memory set,
- a memory copy between two addresses to the same device memory,
any CUDA command to the NULL stream,
- a switch between the L1/shared memory configurations described in Compute Capability 3.x and Compute Capability 7.x.

This means that both memory set and DeviceToDevice memory copies may prevent some overlapping, making even more interesting to use your own kernels, instead of the official implementations.
