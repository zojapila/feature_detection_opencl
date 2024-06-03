/**********************************************************************
Copyright �2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#include "FeatureDetection.hpp"
#include <cmath>


int
FeatureDetection::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!" << std::endl;
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    // allocate memory for input & output image data  */
    inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

    // error check
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");


    // allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

    // error check
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");


    // initialize the Image data to NULL
    memset(outputImageData, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();

    // error check
    CHECK_ALLOCATION(pixelData, "Failed to read pixel Data!");

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationOutput = (cl_uchar*)malloc(width * height * pixelSize);

    // error check
    CHECK_ALLOCATION(verificationOutput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * pixelSize);

    return SDK_SUCCESS;

}


int
FeatureDetection::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * pixelSize);

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!" << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
FeatureDetection::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("FeatureDetection_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


int
FeatureDetection::setupCL()
{
    cl_int err = CL_SUCCESS;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    err = cl::Platform::get(&platforms);
    CHECK_OPENCL_ERROR(err, "Platform::get() failed.");

    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        if(sampleArgs->isPlatformEnabled())
        {
            i = platforms.begin() + sampleArgs->platformId;
        }
        else
        {
            for(i = platforms.begin(); i != platforms.end(); ++i)
            {
                if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                           "Advanced Micro Devices, Inc."))
                {
                    break;
                }
            }
        }
    }

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*i)(),
        0
    };

    context = cl::Context(dType, cps, NULL, NULL, &err);
    CHECK_OPENCL_ERROR(err, "Context::Context() failed.");

    devices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
    CHECK_OPENCL_ERROR(err, "Context::getInfo() failed.");

    std::cout << "Platform :" << (*i).getInfo<CL_PLATFORM_VENDOR>().c_str() << "\n";
    int deviceCount = (int)devices.size();
    int j = 0;
    for (std::vector<cl::Device>::iterator i = devices.begin(); i != devices.end();
            ++i, ++j)
    {
        std::cout << "Device " << j << " : ";
        std::string deviceName = (*i).getInfo<CL_DEVICE_NAME>();
        std::cout << deviceName.c_str() << "\n";
    }
    std::cout << "\n";

    if (deviceCount == 0)
    {
        std::cout << "No device available\n";
        return SDK_FAILURE;
    }

    if(validateDeviceId(sampleArgs->deviceId, deviceCount))
    {
        std::cout << "validateDeviceId() failed" << std::endl;
        return SDK_FAILURE;
    }


    // Check for image support
    imageSupport = devices[sampleArgs->deviceId].getInfo<CL_DEVICE_IMAGE_SUPPORT>
                   (&err);
    CHECK_OPENCL_ERROR(err, "Device::getInfo() failed.");

    // If images are not supported then return
    if(!imageSupport)
    {
        OPENCL_EXPECTED_ERROR("Images are not supported on this device!");
    }

    commandQueue = cl::CommandQueue(context, devices[sampleArgs->deviceId], 0,
                                    &err);
    CHECK_OPENCL_ERROR(err, "CommandQueue::CommandQueue() failed.");
    /*
    * Create and initialize memory objects
    */
    inputImage2D = cl::Image2D(context,
                               CL_MEM_READ_ONLY,
                               cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                               width,
                               height,
                               0,
                               NULL,
                               &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (inputImage2D)");


    // Create memory objects for output Image
    outputImage2D = cl::Image2D(context,
                                CL_MEM_READ_WRITE,
                                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2D)");
    outputImage2D1 = cl::Image2D(context,
                                CL_MEM_READ_WRITE,
                                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2D)");
    outputImage2D22 = cl::Image2D(context,
                                CL_MEM_READ_WRITE,
                                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2D)");
    outputImage2D2 = cl::Image2D(context,
                                CL_MEM_READ_WRITE,
                                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2D)");
  
    

    finalOutputImage2D = cl::Image2D(context,
                                CL_MEM_WRITE_ONLY,
                                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (finalOutputImage2D)");

    device.push_back(devices[sampleArgs->deviceId]);

    // create a CL program using the kernel source
    SDKFile kernelFile;
    std::string kernelPath = getPath();

    if(sampleArgs->isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs->loadBinary.c_str());
        if(kernelFile.readBinaryFromFile(kernelPath.c_str()) != SDK_SUCCESS)
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }
        cl::Program::Binaries programBinary(1,std::make_pair(
                                                (const void*)kernelFile.source().data(),
                                                kernelFile.source().size()));

        program = cl::Program(context, device, programBinary, NULL, &err);
        CHECK_OPENCL_ERROR(err, "Program::Program(Binary) failed.");

    }
    else
    {
        kernelPath.append("FeatureDetection_Kernels.cl");
        if(!kernelFile.open(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        // create program source
        cl::Program::Sources programSource(1,
                                           std::make_pair(kernelFile.source().data(),
                                                   kernelFile.source().size()));

        // Create program object
        program = cl::Program(context, programSource, &err);
        CHECK_OPENCL_ERROR(err, "Program::Program() failed.");

    }

    std::string flagsStr = std::string("");

    // Get additional options
    if(sampleArgs->isComplierFlagsSpecified())
    {
        SDKFile flagsFile;
        std::string flagsPath = getPath();
        flagsPath.append(sampleArgs->flags.c_str());
        if(!flagsFile.open(flagsPath.c_str()))
        {
            std::cout << "Failed to load flags file: " << flagsPath << std::endl;
            return SDK_FAILURE;
        }
        flagsFile.replaceNewlineWithSpaces();
        const char * flags = flagsFile.source().c_str();
        flagsStr.append(flags);
    }

    if(flagsStr.size() != 0)
    {
        std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
    }

    err = program.build(device, flagsStr.c_str());

    if(err != CL_SUCCESS)
    {
        if(err == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[sampleArgs->deviceId]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str << std::endl;
            std::cout << " ************************************************\n";
        }
    }
    CHECK_OPENCL_ERROR(err, "Program::build() failed.");

    // Create kernel
    kernel = cl::Kernel(program, "feature_detection",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");

    kernel2 = cl::Kernel(program, "feature_detection2",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    kernel22 = cl::Kernel(program, "feature_detection22",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    kernel3 = cl::Kernel(program, "feature_detection3",  &err);
    CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");
    // kernel4 = cl::Kernel(program, "feature_detection4",  &err);
    // CHECK_OPENCL_ERROR(err, "Kernel::Kernel() failed.");


    // Check group size against group size returned by kernel
    kernelWorkGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    kernelWorkGroupSize2 = kernel2.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");
    kernelWorkGroupSize22 = kernel22.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");
    kernelWorkGroupSize3 = kernel3.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");
    // kernelWorkGroupSize4 = kernel4.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
    //                       (devices[sampleArgs->deviceId], &err);
    // CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelWorkGroupSize << std::endl;
        }

        if(blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

    if((blockSizeX * blockSizeY) > kernelWorkGroupSize2)
        {
            if(!sampleArgs->quiet)
            {
                std::cout << "Out of Resources!" << std::endl;
                std::cout << "Group Size specified : "
                        << blockSizeX * blockSizeY << std::endl;
                std::cout << "Max Group Size supported on the kernel : "
                        << kernelWorkGroupSize2 << std::endl;
                std::cout << "Falling back to " << kernelWorkGroupSize2 << std::endl;
            }

            if(blockSizeX > kernelWorkGroupSize2)
            {
                blockSizeX = kernelWorkGroupSize2;
                blockSizeY = 1;
            }
        }
           if((blockSizeX * blockSizeY) > kernelWorkGroupSize22)
        {
            if(!sampleArgs->quiet)
            {
                std::cout << "Out of Resources!" << std::endl;
                std::cout << "Group Size specified : "
                        << blockSizeX * blockSizeY << std::endl;
                std::cout << "Max Group Size supported on the kernel : "
                        << kernelWorkGroupSize22 << std::endl;
                std::cout << "Falling back to " << kernelWorkGroupSize22 << std::endl;
            }

            if(blockSizeX > kernelWorkGroupSize22)
            {
                blockSizeX = kernelWorkGroupSize22;
                blockSizeY = 1;
            }
        }
    
    if((blockSizeX * blockSizeY) > kernelWorkGroupSize3)
        {
            if(!sampleArgs->quiet)
            {
                std::cout << "Out of Resources!" << std::endl;
                std::cout << "Group Size specified : "
                        << blockSizeX * blockSizeY << std::endl;
                std::cout << "Max Group Size supported on the kernel : "
                        << kernelWorkGroupSize3 << std::endl;
                std::cout << "Falling back to " << kernelWorkGroupSize3 << std::endl;
            }

            if(blockSizeX > kernelWorkGroupSize3)
            {
                blockSizeX = kernelWorkGroupSize3;
                blockSizeY = 1;
            }
        }
    
    // if((blockSizeX * blockSizeY) > kernelWorkGroupSize4)
    //     {
    //         if(!sampleArgs->quiet)
    //         {
    //             std::cout << "Out of Resources!" << std::endl;
    //             std::cout << "Group Size specified : "
    //                     << blockSizeX * blockSizeY << std::endl;
    //             std::cout << "Max Group Size supported on the kernel : "
    //                     << kernelWorkGroupSize4 << std::endl;
    //             std::cout << "Falling back to " << kernelWorkGroupSize4 << std::endl;
    //         }

    //         if(blockSizeX > kernelWorkGroupSize4)
    //         {
    //             blockSizeX = kernelWorkGroupSize4;
    //             blockSizeY = 1;
    //         }
    //     }
    

    return SDK_SUCCESS;
}

// TODO: AAA TU KONIEC

// int FeatureDetection::runCLKernels() {
//     cl_int status;

//     // Inicjalizacja zmiennych origin i region
//     cl::size_t<3> origin;
//     origin[0] = 0;
//     origin[1] = 0;
//     origin[2] = 0;

//     cl::size_t<3> region;
//     region[0] = width;
//     region[1] = height;
//     region[2] = 1;

//     // Enqueue Write Image
//     cl::Event writeEvt;
//     status = commandQueue.enqueueWriteImage(
//                  inputImage2D,
//                  CL_TRUE,
//                  origin,
//                  region,
//                  0,
//                  0,
//                  inputImageData,
//                  NULL,
//                  &writeEvt);
//     CHECK_OPENCL_ERROR(status,
//                        "CommandQueue::enqueueWriteImage failed. (inputImage2D)");

//     status = commandQueue.flush();
//     CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

//     // Czekaj na zakończenie operacji zapisu
//     status = writeEvt.wait();
//     CHECK_OPENCL_ERROR(status, "Event::wait failed.");

//     // Set appropriate arguments to the kernel1
//     status = kernel.setArg(0, inputImage2D);
//     CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

//     cl::Buffer resultBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);

//     status = kernel.setArg(1, outputImage2D);
//     CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

//     // Uruchomienie wszystkich instancji kernela1
//     cl::NDRange globalThreads(width, height);
//     // cl::NDRange localThreads(blockSizeX, blockSizeY);

//     // status = commandQueue.enqueueNDRangeKernel(
//     //             kernel,
//     //             cl::NullRange,
//     //             globalThreads,
//     //             localThreads,
//     //             0,
//     //             NULL);
//     // CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");
//     cl::Device device = commandQueue.getInfo<CL_QUEUE_DEVICE>();
//     size_t maxLocalSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

// // Dostosowanie lokalnego rozmiaru grupy roboczej, aby nie przekraczać maksymalnego rozmiaru
// // cl::NDRange localThreads(std::min(maxLocalSize, static_cast<size_t>(blockSizeX)), 
// //                          std::min(maxLocalSize, static_cast<size_t>(blockSizeY)));
// cl::NDRange localThreads(1, 1);
// // Uruchomienie wszystkich instancji kernela1
// status = commandQueue.enqueueNDRangeKernel(
//             kernel,
//             cl::NullRange,
//             globalThreads,
//             localThreads,
//             0,
//             NULL);
// CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

//     status = commandQueue.flush();
//     CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

//     // Czekaj na zakończenie operacji kernela1
//     commandQueue.finish();
//      if(writeOutputImage(OUTPUT_IMAGE) != SDK_SUCCESS)
//     {
//         return SDK_FAILURE;
//     }

//     // Set appropriate arguments to the kernel2
//     status = kernel2.setArg(0, outputImage2D);
//     CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

//     status = kernel2.setArg(1, finalOutputImage2D);
//     CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

//     // Uruchomienie wszystkich instancji kernela2
//     status = commandQueue.enqueueNDRangeKernel(
//                 kernel2,
//                 cl::NullRange,
//                 globalThreads,
//                 localThreads,
//                 NULL,
//                 NULL);
//     CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

//     status = commandQueue.flush();
//     CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

//     // Czekaj na zakończenie operacji kernela2
//     commandQueue.finish();

//     // Enqueue Read Image
//     cl::Event readEvt;
//     status = commandQueue.enqueueReadImage(
//                  finalOutputImage2D,
//                  CL_FALSE,
//                  origin,
//                  region,
//                  0,
//                  0,
//                  outputImageData,
//                  NULL,
//                  &readEvt);
//     CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadImage failed.");

//     status = commandQueue.flush();
//     CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

//     // Czekaj na zakończenie operacji odczytu
//     status = readEvt.wait();
//     CHECK_OPENCL_ERROR(status, "Event::wait failed.");

//     return SDK_SUCCESS;
// }
cl_int FeatureDetection::runCLKernels() {
    cl_int status;

    // Inicjalizacja zmiennych origin i region
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    // Enqueue Write Image
    cl::Event writeEvt;
    status = commandQueue.enqueueWriteImage(
                 inputImage2D,
                 CL_TRUE,
                 origin,
                 region,
                 0,
                 0,
                 inputImageData,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueWriteImage failed. (inputImage2D)");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // Czekaj na zakończenie operacji zapisu
    status = writeEvt.wait();
    CHECK_OPENCL_ERROR(status, "Event::wait failed.");

    // Set appropriate arguments to the kernel1
    status = kernel.setArg(0, inputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    cl::Buffer resultBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);

    status = kernel.setArg(1, outputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

    // Uruchomienie wszystkich instancji kernela1
    cl::NDRange globalThreads(width, height);
    // cl::NDRange localThreads(blockSizeX, blockSizeY);
    cl::NDRange localThreads(1024, 1);

    status = commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                globalThreads,
                localThreads,
                0,
                NULL);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // Czekaj na zakończenie operacji kernela1
    commandQueue.finish();

    if(writeOutputImage(OUTPUT_IMAGE) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Set appropriate arguments to the kernel2
    status = kernel2.setArg(0, outputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    cl::Buffer resultBuffer2(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);

    status = kernel2.setArg(1, outputImage2D1);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

    // Uruchomienie wszystkich instancji kernela2
    status = commandQueue.enqueueNDRangeKernel(
                kernel2,
                cl::NullRange,
                globalThreads,
                localThreads,
                NULL,
                NULL);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // Czekaj na zakończenie operacji kernela2
    commandQueue.finish();
        // Set appropriate arguments to the kernel3
    status = kernel22.setArg(0, outputImage2D1);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    cl::Buffer resultBuffer22(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);

    status = kernel22.setArg(1, outputImage2D22);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    // Uruchomienie wszystkich instancji kernela3
    status = commandQueue.enqueueNDRangeKernel(
                kernel22,
                cl::NullRange,
                globalThreads,
                localThreads,
                NULL,
                NULL);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // Czekaj na zakończenie operacji kernela3
    commandQueue.finish();

    // Set appropriate arguments to the kernel3
    status = kernel3.setArg(0, inputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");
    status = kernel3.setArg(1, outputImage2D22);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    cl::Buffer resultBuffer3(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);

    status = kernel3.setArg(2, finalOutputImage2D);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

    // Uruchomienie wszystkich instancji kernela3
    status = commandQueue.enqueueNDRangeKernel(
                kernel3,
                cl::NullRange,
                globalThreads,
                localThreads,
                NULL,
                NULL);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // Czekaj na zakończenie operacji kernela3
    commandQueue.finish();

    // // Set appropriate arguments to the kernel4
    // status = kernel4.setArg(0, outputImage2D2);
    // CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    // status = kernel4.setArg(1, finalOutputImage2D);
    // CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");

    // // Uruchomienie wszystkich instancji kernela4
    // status = commandQueue.enqueueNDRangeKernel(
    //             kernel4,
    //             cl::NullRange,
    //             globalThreads,
    //             localThreads,
    //             NULL,
    //             NULL);
    // CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    // status = commandQueue.flush();
    // CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // // Czekaj na zakończenie operacji kernela4
    // commandQueue.finish();

    // Enqueue Read Image
    cl::Event readEvt;
    status = commandQueue.enqueueReadImage(
                 finalOutputImage2D,
                 CL_FALSE,
                 origin,
                 region,
                 0,
                 0,
                 outputImageData,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadImage failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    // Czekaj na zakończenie operacji odczytu
    status = readEvt.wait();
    CHECK_OPENCL_ERROR(status, "Event::wait failed.");

    return SDK_SUCCESS;
}



int
FeatureDetection::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initialization failed");

    Option* iteration_option = new Option;
    if(!iteration_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;
    sampleArgs->AddOption(iteration_option);
    delete iteration_option;
    return SDK_SUCCESS;
}

int
FeatureDetection::setup()
{
    // Allocate host memory and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    std::cout << filePath << std::endl;
    if(readInputImage(filePath) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if (status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;

}

int
FeatureDetection::run()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    if(writeOutputImage(FINAL_OUTPUT_IMAGE) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
FeatureDetection::cleanup()
{

    // release program resources (input memory etc.)
    FREE(inputImageData);
    FREE(outputImageData);
    FREE(verificationOutput);

    return SDK_SUCCESS;
}

void
FeatureDetection::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}


int
main(int argc, char * argv[])
{
    FeatureDetection clFeatureDetection;

    if(clFeatureDetection.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clFeatureDetection.sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(clFeatureDetection.sampleArgs->isDumpBinaryEnabled())
    {
        return clFeatureDetection.genBinaryImage();
    }
    else
    {
        // Setup
        int status = clFeatureDetection.setup();
        if(status != SDK_SUCCESS)
        {
            return status;
        }

        // Run
        if(clFeatureDetection.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        // // VerifyResults
        // if(clFeatureDetection.verifyResults() != SDK_SUCCESS)
        // {
        //     return SDK_FAILURE;
        // }

        // Cleanup
        if(clFeatureDetection.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clFeatureDetection.printStats();
    }

    return SDK_SUCCESS;
}



