# CUDALearning
This GitHub repository is a place to put all my tests/projects, done with the purpose of better understanding CUDA and GPU performance behavior, as well as learning and testing new CUDA features. There might be things present in NVIDIA CUDA documentation, that I may be missing. Don't hessitate to  There might also be some exercise descriptions to teach CUDA and parallel programming, that I created.

The organization of the project is very simple. A set of folders, where each folder contains a project that serves a purpose.

Each project can be in different status:
- To be defined (TBD): the description of the project is not finished.
- Open: the goals of the project are defined and it's ready to start, but it was not yet started.
- In Progress: the execution of the project started but it's not finished.
- Blocked: the execution of the project started but it's not finished, and there is something that prevents further development.
- Finished: the execution of the project is finished, because I reached the main goals of the project. It may still change if there is something new that I discovered, or if I wanted to do some code cleaning or documentation.

## CUDALearning projects
This section enumerates the different projects, with an small description of each project and the status of the project.

### DeviceToDevice memory copies performance (And philosophy of why to do that?) TBD
Project dedicated to the exploration of the best performance possible, when copying data from GPU pointers to GPU pointers. Also, contains discussion about why and how to avoid using this copies. 

Why I want to find the best performance? Because it might be a fast way to improve some code I'm working on, where removing the copies would require some memory pools and pointer ownership control (multiple concurrent reads, from different streams controlled each by a differrent CPU thread, single protected write from a single stream controlled by a single CPU thread). Also, it will help me clarify some aspects of the Device memory organization on Maxwell and Pascal architecture (things like maximum bandwith per Warp, interactions between Warps when reading Device memory, L1/L2 cache capabilities, effects on data locality due to thread block scheduling when there are more or less thread blocks allocated for the kernel).

If I'm not wrong, cudaMemcpy() with the flag cudaMemcpyDeviceToDevice, lanches a Kernel that performs the copy of data from one GPU pointer to another.

Why did I think that this copies are a Kernel? Because of the behavior observable in NSIGHT Visual Studio Timelines. They overlap slightly with other kernels launched on other cuda Streams,
