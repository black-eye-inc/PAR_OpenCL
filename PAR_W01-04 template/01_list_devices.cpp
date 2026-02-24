// Run code via terminal, typing 
// 1. """ g++ 01_list_devices.cpp -o output_here/my_devices.exe -I"C:\OpenCL-SDK\include" -lOpenCL """ --> to pre-load the .cpp file.
// 2. """ ./output_here/my_devices.exe """ --> to load the file to 

// ============================================================
// Task 1.1: Exploring OpenCL Devices
// ============================================================
// Complete this program to list all GPU devices on your system
// and display the platform and device information.

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
using namespace std;


int main() {
    std::vector<cl::Platform> platforms;
    // ============================================================
    // TODO 1: Query all available OpenCL platforms, and
    // store your results in the "platforms" vector
    // Use: cl::Platform::get(&platforms)
    // ============================================================
    
    cl::Platform::get(&platforms);

    //=============================================================
    // TODO 2: Display message "No OpenCL platforms found!" 
    // when "platform" vector is empty and terminate you application
    //=============================================================
    if (platforms.empty())
    {
        cout << "No OpenCL platforms found!\n";
        return 1;
    }

    // Loop through each available platform
    for (auto &p : platforms) {
        // ============================================================
        // TODO 3: Print the name of each platform
        // Hint: Use p.getInfo<CL_PLATFORM_NAME>()
        // ============================================================

        cout << "\n\n----------------==============  Platform \"" << p.getInfo<CL_PLATFORM_NAME>() << "\"  ==============----------------\n";
        cout << "--- Name: " << p.getInfo<CL_PLATFORM_NAME>() << endl;
        cout << "--- Vendor: " << p.getInfo<CL_PLATFORM_VENDOR>() << endl;
        cout << "--- Version: " << p.getInfo<CL_PLATFORM_VERSION>() << endl;
        cout << "--- Profile: " << p.getInfo<CL_PLATFORM_PROFILE>() << endl;
        

        std::vector<cl::Device> devices;
        // ============================================================
        // TODO 4: Get all GPU devices for this platform and 
        // store in a vector called devices
        // Hint: Use platform.getDevices(CL_DEVICE_TYPE_GPU, &devices)
        // CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, etc.
        // ============================================================

        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        

        //=============================================================
        // TODO 5: Display message "No devices found for this platform." 
        // when "device" vector is empty and terminate you application
        //=============================================================
        
        cout << "--- GPU Devices for this platform:\n";
        if (devices.empty())
        {
            cout << "--- --- None found.\n";
        }
        

        // ============================================================
        // TODO 6: Loop through each GPU device and print its info
        // Required info:Device name, Max compute units, Max work-group size
        // and Max clock frequency
        // Hint: use device.getInfo<>() with:
        //   CL_DEVICE_NAME returns data type std::string
        //   CL_DEVICE_MAX_COMPUTE_UNITS returns a data type cl_uint
        //   CL_DEVICE_MAX_WORK_GROUP_SIZE  returns a data type size_t
        //   CL_DEVICE_MAX_CLOCK_FREQUENCY returns a data type cl_uint
        // ============================================================
        else
        {
            for (auto &d : devices) 
            {
                cout << "--- --- Device \"" << d.getInfo<CL_DEVICE_NAME>() << "\":\n";
                cout << "--- --- --- Name: " << d.getInfo<CL_DEVICE_NAME>() << endl;
                cout << "--- --- --- Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << endl;
                cout << "--- --- --- Max Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
                cout << "--- --- --- Max Work Group Size: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
                cout << "--- --- --- Max Clock Frequency: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
            }
        }
    }

    // ============================================================
    // TODO 7: For discussion with colleagues:
    // Compare the specs of different CPU and GPU devices.
    // Which device do you expect to perform better in parallel tasks?
    // ============================================================

    return 0;
}
