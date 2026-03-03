/*
-------------------------------------------------------------
Basic Profiling
compile using: 
g++ 02_basic_profiling.cpp -o 02_basic_profiling.exe -I"C:\OpenCL-SDK\include" -lOpenCL
-------------------------------------------------------------
*/

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

/*------------------------------------------------------------
Helper function: Detailed profiling info for any event
--------------------------------------------------------------*/
enum class ProfilingResolution { PROF_NS, PROF_US, PROF_MS };

std::string GetFullProfilingInfo(const cl::Event &event, ProfilingResolution res) {
    cl_ulong queued = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    cl_ulong submit = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    cl_ulong start  = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end    = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double scale = 1.0;
    std::string unit = "ns";
    if (res == ProfilingResolution::PROF_US) { scale = 1e-3; unit = "us"; }
    else if (res == ProfilingResolution::PROF_MS) { scale = 1e-6; unit = "ms"; }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "\n=== Profiling Breakdown (" << unit << ") ===\n"
        << "Queued : " << (submit - queued) * scale << " " << unit << "\n"
        << "Submit : " << (start - submit) * scale << " " << unit << "\n"
        << "Execute: " << (end - start) * scale << " " << unit << "\n"
        << "Total  : " << (end - queued) * scale << " " << unit << "\n";
    return oss.str();
}

int main() {
    std::vector<cl::Platform> platforms;
    // ------------------------------------------------------
    // TODO 1: Query all available OpenCL platforms, and
    // store your results in the "platforms" vector
    // Use: cl::Platform::get(&platforms)
    // ------------------------------------------------------
    



    //--------------------------------------------------------------
    // TODO 2: Display message "No OpenCL platforms found!" 
    // when "platform" vector is empty and terminate you application
    //---------------------------------------------------------------
    
    

    // ---------------------------------------
    // loop through all available platforms
    // ---------------------------------------
    for (size_t p = 0; p < platforms.size(); ++p) {

        cl::Platform platform = platforms[p];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        std::cout << "\n========================================\n";
        std::cout << "Using platform: " << platformName << "\n";

        std::vector<cl::Device> devices;
        //-----------------------------------------------
        // TODO 3: Get all devices for this platform and 
        // store in the vector called devices
        // Hint: Use platform.getDevices(CL_DEVICE_TYPE_GPU, &devices)
        // CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, etc.
        // -----------------------------------------------
        
        


        //-------------------------------------------------------------
        // TODO 4: Display message "No devices found for this platform." 
        // when "device" vector is empty and terminate you application
        //-------------------------------------------------------------
        
        

        // Loop through all devices 
        for (size_t d = 0; d < devices.size(); ++d) {

            cl::Device device = devices[d];
            std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
            std::cout << "Using device: " << deviceName << "\n";

            // ---------------------------------------------------------
            // TODO 5: creates an OpenCL context for the selected device.
            // -----------------------------------------------------------
            
            

            // ------------------------------------------------------------
            // TODO 6: Enable profiling for the queue
            // Hint: Add CL_QUEUE_PROFILING_ENABLE as the third argument
            // ------------------------------------------------------------
            
            

            
            /*
                Host data preparation: We will defines three host-side 
                integer vectors of 1024 elements: A initialized to 1, 
                B to 2, and C to 0 are stored in CPU memory.
            */
            const int N = 1024;
            std::vector<int> A(N, 1);
            std::vector<int> B(N, 2);
            std::vector<int> C(N, 0);

            /* 
            Create OpenCL buffers:OpenCL uses buffers to manage across devices
            We allocate GPU (device) memory buffers, copy the host data for A and B, and 
            prepares an output buffer C for kernel results
            */
            size_t bytes = N * sizeof(int);
            cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data());
            cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data());
            cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, bytes);

            /*
            Kernel source: Defines a parallel OpenCL kernel add where each thread, 
            identified by get_global_id(0), adds corresponding elements of input arrays A 
            and B and stores the result in output array C. We will see kernels later
            */
            std::string kernel_code = R"CLC(
                kernel void add(global const int* A, global const int* B, global int* C) {
                    int id = get_global_id(0);
                    C[id] = A[id] + B[id];
                }
            )CLC";

            /* Build kernel program:
            We will use cl::Program program(context, kernel_code) to create a program
            from the kernel source and program.build({device}) to compile it for the device, 
            allowing kernel execution and providing build logs if compilation fails.    
            */
            cl::Program program(context, kernel_code);
            if (program.build({device}) != CL_SUCCESS) {
                std::cout << "Error building program:\n"
                        << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
                continue;
            }

            /*
            Set Kernel Arguments:
            Creates a kernel object called kernel_add from the compiled program and 
            use setArg() to link device buffers A, B, and C to its arguments, 
            preparing it for execution on the device.
            */
            cl::Kernel kernel_add(program, "add");
            kernel_add.setArg(0, bufferA);
            kernel_add.setArg(1, bufferB);
            kernel_add.setArg(2, bufferC);

            /*-------------------------------------------------------------------
            // TODO 6: Attach an event called "prof_event" to the kernel command
            // Hint: Pass &prof_event as the last argument of enqueueNDRangeKernel
            syntax: queue.enqueueNDRangeKernel(kernel, offset, global, local, waitEvents, event);
            kernel-->kernel to execute, offset-->starting global ID (cl::NullRange-->start at 0)
            global-->total number of work-items (1D=cl::NDRange(N)/2D = cl::NDRange(N, N) /3D=cl::NDRange(N, N, N) )
            local-->work-items per work-group (cl::NullRange-->runtime decides)
            waitEvents-->events that must finish before this runs (nullptr-->none)
            event-->output event for synchronisation/profiling
            ----------------------------------------------------------------------------*/
            cl::Event prof_event;
            
            



            /*--------------------------------------------------------------------
            TODO 7: Display kernel execution time. Don't use Helper function 
            "GetFullProfilingInfo". use getProfilingInfo<CL_PROFILING_COMMAND_START/END>()
            Then perform the calculations. note that getProfilingInfo returns cl_ulong data type
            -----------------------------------------------------------------------*/
            

            

            //----------------------------------------------------------------------

            double exec_ns = static_cast<double>(end - start);
            double exec_ms = exec_ns * 1e-6;

            std::cout << "\nKernel execution time [ns]: " << exec_ns << std::endl;
            std::cout << "Kernel execution time [ms]: " << exec_ms << " ms" << std::endl;

            /*
            TODO 8 [Optional]: Display detailed event breakdown using the 
            helper function "GetFullProfilingInfo"
            */ 

            

            /*
            Reading results back:
            Reads the computed results from the device buffer bufferC back into the host
            vector C using a blocking transfer (CL_TRUE) so the data is ready for use 
            on the CPU.
            */
            queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, bytes, C.data());

            std::cout << "\nSample results (first 5 elements):\n";
            for (int i = 0; i < 5; ++i)
                std::cout << "C[" << i << "] = " << C[i] << "\n";
        }
    }

    /* TODO 9: Could you check execution time on different devices (CPU/GPU).
    Does it vary between consecutive launches, and how big are the differences?
    use the detailed profiling info to see how the total times compare to execution time alone?*/
    
    return 0;
}
