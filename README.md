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
