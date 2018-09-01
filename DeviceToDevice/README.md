# DeviceToDevice memory copies performance (And philosophy of why to do that?) TBD

Why I want to find the best performance? Because it might be a fast way to improve some code I'm working on, where removing the copies would require some memory pools and pointer ownership control (multiple concurrent reads, from different streams controlled each by a differrent CPU thread, single protected write from a single stream controlled by a single CPU thread). Also, it will help me clarify some aspects of the Device memory organization on Maxwell and Pascal architecture (things like maximum bandwith per Warp, interactions between Warps when reading Device memory, L1/L2 cache capabilities, effects on data locality due to thread block scheduling when there are more or less thread blocks allocated for the kernel).

If I'm not wrong, cudaMemcpy() with the flag cudaMemcpyDeviceToDevice, lanches a Kernel that performs the copy of data from one GPU pointer to another.

Why did I think that this copies are a Kernel? Because of the behavior observable in NSIGHT Visual Studio Timelines. They overlap slightly with other kernels launched on other cuda Streams,
