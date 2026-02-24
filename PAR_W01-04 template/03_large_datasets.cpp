/*
============================================================
3.	Large datasets
============================================================
*/

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>

// ============================================================
// Helper function: Detailed profiling info for any event
// ============================================================
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
    // Query all available OpenCL platforms, and
    // store your results in the "platforms" vector
    // Use: cl::Platform::get(&platforms)
    // ------------------------------------------------------
    cl::Platform::get(&platforms);
    //--------------------------------------------------------------
    // Display message "No OpenCL platforms found!" 
    // when "platform" vector is empty and terminate you application
    //---------------------------------------------------------------
    if (platforms.empty()) {
        std::cout << "No OpenCL platforms found!" << std::endl;
        return 1;
    }
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
        // Get all devices for this platform and 
        // store in the vector called devices
        // -----------------------------------------------
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        //-------------------------------------------------------------
        // Display message "No devices found for this platform." 
        // when "device" vector is empty and terminate you application
        //-------------------------------------------------------------
        if (devices.empty()) {
            std::cout << "No OpenCL devices found!" << std::endl;
            continue;
        }

        // Loop through all devices 
        for (size_t d = 0; d < devices.size(); ++d) {

            cl::Device device = devices[d];
            std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
            std::cout << "Using device: " << deviceName << "\n";

            /*
            We will query device memory limits to support chunking. We will
            Use these values to determine: 
            1) maxAlloc: the largest single buffer allocation allowed
            2) globalMem : total global memory available on the device
            The maxAlloc value helps estimate the maximum number of
            elements you can process in one chunk (based on memory limits)
            and decide when to split very large vectors into smaller segments.
            */
            cl_ulong maxAlloc  = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            cl_ulong globalMem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

            // Optional: print limits 
            std::cout << "  CL_DEVICE_MAX_MEM_ALLOC_SIZE [bytes]: " << maxAlloc << "\n";
            std::cout << "  CL_DEVICE_GLOBAL_MEM_SIZE    [bytes]: " << globalMem << "\n";


            // ---------------------------------------------------------
            // creates an OpenCL context for the selected device.
            // -----------------------------------------------------------
            cl::Context context(device);

            // ------------------------------------------------------------
            // Enable profiling for the queue
            // ------------------------------------------------------------
            cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
            
            /*
            TODO 1:
            Profile your code for different vector lengths 
            Change input vector size to test large datasets
            Try: 1000, 10000, 100000, 1000000 (1M)
            */
            const int N = 1000000;
            std::vector<int> A(N, 1);
            std::vector<int> B(N, 2);
            std::vector<int> C(N, 0);

            // ------------------------------------------------------------
            // Decide chunk size
            // We need to hold 3 buffers (A,B,C) on device at the same time.
            // Use both maxAlloc and globalMem as constraints.
            // ------------------------------------------------------------
            size_t maxElemsByAlloc  = static_cast<size_t>(maxAlloc) / sizeof(int);
            size_t maxElemsByGlobal = static_cast<size_t>(globalMem) / (3 * sizeof(int)); // 3 buffers

            size_t chunkElems = std::min({ static_cast<size_t>(N), maxElemsByAlloc, maxElemsByGlobal });

            // (Optional) add a safety margin to reduce OOM risk on some drivers
            chunkElems = std::max<size_t>(1, static_cast<size_t>(chunkElems * 0.8));

            if (chunkElems == 0) {
                std::cout << "ERROR: chunkElems computed as 0 (device memory limits too small?).\n";
                continue;
            }

            size_t chunkBytes = chunkElems * sizeof(int);

            // ------------------------------------------------------------
            // Create CHUNK-SIZED OpenCL buffers (not N-sized)
            // ------------------------------------------------------------
            cl::Buffer bufferA(context, CL_MEM_READ_ONLY,  chunkBytes);
            cl::Buffer bufferB(context, CL_MEM_READ_ONLY,  chunkBytes);
            cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, chunkBytes);

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



            /*
            TODO 3.1: Define helper lambda function to compute elapsed time from an
            OpenCL profiling event. The lambda extracts CL_PROFILING_COMMAND_START and 
            CL_PROFILING_COMMAND_END timestamps from an event and returns the elapsed 
            time in nanoseconds. NOTE: This is valid only if the command queue was created
            with CL_QUEUE_PROFILING_ENABLE.
            */
            
            


            // Accumulators across ALL chunks
            double uploadA_ms_total = 0.0;
            double uploadB_ms_total = 0.0;
            double exec_ms_total    = 0.0;
            double downloadC_ms_total = 0.0;

            // ------------------------------------------------------------
            // Chunking loop: upload --> compute --> download for each segment
            // ------------------------------------------------------------
            for (size_t offset = 0; offset < static_cast<size_t>(N); offset += chunkElems) {
                size_t thisChunkElems = std::min(chunkElems, static_cast<size_t>(N) - offset);
                size_t thisBytes      = thisChunkElems * sizeof(int);

                /*
                TODO 2: Record profiling events for data transfers and kernel execution.
                - Attach events to host --> device uploads for A and B, kernel execution, and 
                device --> host download for C, so profiling timestamps can be queried later.
                NOTE: The command queue must be created with CL_QUEUE_PROFILING_ENABLE
                for CL_PROFILING_COMMAND_START/END to be valid.
                */
                cl::Event A_event, B_event, C_event, kernel_event;

                // TODO 2.1: Attach events to input uploads (host --> device)
                
                

                // TODO 2.2: Attach event "kernel_event" to the kernel enqueue
                
                

                // TODO 2.3: Attach event to output download (device --> host)
                
                


                // Accumulate timings (ms)
                /* TODO 3.2: Compute elapsed times (in milliseconds) for each 
                profiled operation using the recorded events and Accumulate timings (ms):
                --> uploadA_ms   : host --> device transfer for buffer A
                ------> e.g. uploadA_ms = duration_ns(A_event) * 1e-6;
                --> uploadB_ms   : host --> device transfer for buffer B
                --> exec_ms      : kernel execution time
                --> downloadC_ms : device --> host transfer for buffer C
                */
                
                


            }

            // ------------------------------------------------------------
            // TODO 3.3 Compute total timings
            // --> total_ms = uploadA_ms + uploadB_ms + exec_ms + downloadC_ms
            // ------------------------------------------------------------
            
            
            
            
            // Display chunked profiling results
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "\n=== Chunked Profiling Results ===\n";
            std::cout << "Vector length N: " << N << "\n";
            std::cout << "Chunk size [elements]: " << chunkElems << "\n";
            std::cout << "Upload A total [ms]: " << uploadA_ms_total << "\n";
            std::cout << "Upload B total [ms]: " << uploadB_ms_total << "\n";
            std::cout << "Kernel exec total [ms]: " << exec_ms_total << "\n";
            std::cout << "Download C total [ms]: " << downloadC_ms_total << "\n";
            std::cout << "---------------------------------\n";
            std::cout << "Total operation time [ms]: " << total_ms << "\n";
        
        }
    
    }

    return 0;
}
