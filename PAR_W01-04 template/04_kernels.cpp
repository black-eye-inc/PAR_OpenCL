/*
------------------------------------------------------------
In this exercise, we explore how OpenCL kernel functions work.
The kernel (device-side) code is stored in a separate file: "kernels/my_kernels.cl"
------------------------------------------------------------
*/

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <fstream>

/* -----------------------------------------------------------
TODO 1: Implement a helper function to load OpenCL kernel code
-----------------------------------------------------------
Create a function called LoadKernelSource that takes the path
to a .cl kernel file, uses std::ifstream to open and read the
entire file into a std::string, throws an error if the file
cannot be opened, and returns the kernel source so it can be
passed to cl::Program for runtime compilation.
// -----------------------------------------------------------*/




double GetEventExecutionTimeMS(const cl::Event &event) {
    cl_ulong start = 0, end = 0;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    return (end - start) * 1e-6; // ns to ms
}

// -----------------------------------------------------------
// Main Program
// -----------------------------------------------------------
int main() {
    // Get available OpenCL platforms and devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cout << "No OpenCL platforms found!" << std::endl;
        return 1;
    }

    // Loop through all platforms
    for (size_t p = 0; p < platforms.size(); ++p) {
        cl::Platform platform = platforms[p];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        std::cout << "\n========================================\n";
        std::cout << "Using platform: " << platformName << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); // CL_DEVICE_TYPE_ALL or (CL_DEVICE_TYPE_CPU
        if (devices.empty()) {
            std::cout << "No OpenCL devices found for this platform!" << std::endl;
            continue;
        }

        // Loop through all devices on this platform
        for (size_t d = 0; d < devices.size(); ++d) {
            cl::Device device = devices[d];
            std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
            std::cout << "Using device: " << deviceName << std::endl;

            cl::Context context(device);
            cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

            /*
            TODO 2: Start with small input vectors (10 elements)
            Modify the kernel code (see folder: 04_kernels/my_kernels.cl) to create 
            --> a "mult" kernel for parallel multiplication.
            --> a combined kernel "multadd" performing Multilication and addition e.g. C = A * B + B.
            --> a float addition "addf" kernel.
            --> a Create double addition "addd" kernel (if supported)
            see: kernels/my_kernels.cl to create these kernels.
            */
            const int N = 1000000;
            std::vector<int> A(N), B(N), C(N, 0);
            for (int i = 0; i < N; ++i) {
                A[i] = i + 1;
                B[i] = (i + 1) * 2;
            }

            size_t bytes = N * sizeof(int);

            // Buffers
            cl::Buffer bufferA(context, CL_MEM_READ_ONLY, bytes);
            cl::Buffer bufferB(context, CL_MEM_READ_ONLY, bytes);
            cl::Buffer bufferC(context, CL_MEM_READ_WRITE, bytes);

            queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, bytes, A.data());
            queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, bytes, B.data());

            // Load and build kernels
            std::string kernel_source = LoadKernelSource("04_kernels/my_kernels.cl");
            cl::Program program(context, kernel_source);

            if (program.build({device}) != CL_SUCCESS) {
                std::cerr << "Error building program:\n"
                          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                          << std::endl;
                continue; // move to next device instead of exiting
            }

            /*
            -----------------------------------------------------------
            TODO 2: Select kernel function to run. For now, use "add" and 
            when others are ready, switch between them.
            Options:
                    "add"     - addition
                    "mult"    - multiplication
                    "multadd" - combined operation
                    "addf" -  float addition
                    "addd" -  double addition
            -----------------------------------------------------------
            */
            
            

            // Launch the kernel
            cl::Event event;
            
            


            double event_time = GetEventExecutionTimeMS(event);
            std::cout << "\n--- Profiling ---\n";
            std::cout << "Kernel time: " << event_time << " ms\n";
            
            /* 
            TODO 3: Modify the host code to compute C=A*B+B using two
            separate OpenCL kernels instead of the combined "multadd" kernel.

            First, create and execute a multiplication kernel ("mult")
            to compute C=A*B. After the multiplication kernel finishes, 
            execute an addition kernel ("add") to compute C = C + B.

            -> Kernel creation is similar to: cl::Kernel kernel_add(program, "add");
            -> Two OpenCL events are required: cl::Event event_mult, event_add;
            -> Ensure that the multiplication kernel completes before launching 
               the addition kernel by calling: event_mult.wait();
            */
           
            

            /*
            TODO 4: Use the GetEventExecutionTimeMS helper function provided
            to extract the multiply and addition execution times in millisecond(ms)
            Display the addition kernel time, multiply kernel time and the total kernel time.
            */
            


            
        }
    }

    return 0;
}
