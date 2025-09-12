#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upsweep(int n, int stride, int* dev_odata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n || index >= (n / stride / 2))
            {
                return;
            }

            // stride = 2^d, d \in [0, log2(n) - 1]
            int strideMult2 = stride * 2;
            if (index < (n / strideMult2))
            {
                if ((n/strideMult2) == 1)
                {
                    // Last element clear at final stage
                    dev_odata[index + strideMult2 - 1] = 0;
                    return;
                }

                int writeIndex = index * strideMult2;
                dev_odata[writeIndex + strideMult2 - 1] += dev_odata[writeIndex + stride - 1];
            }
        }

        __global__ void downsweep(int n, int stride, int* dev_odata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n || index >= (n / stride))
            {
                return;
            }

            // let n = 8
            // stage = 0, stride = 8, n/stride = 1
            // stage = 1, stride = 4, n/stride = 2
            // stage = 2, stride = 2, n/stride = 4
            int strideDiv2 = stride / 2;

            if (index < (n / stride))
            {
                int writeIndex = index * stride;

                int left = dev_odata[writeIndex + strideDiv2 - 1];
                int right = dev_odata[writeIndex + stride - 1];

                dev_odata[writeIndex + strideDiv2 - 1] = right;
                dev_odata[writeIndex + stride - 1] += left;
            }
        }

        __global__ void clearLastElem(int n, int* dev_odata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
            {
                return;
            }

            if (index == n - 1)
            {
                dev_odata[index] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scanDispatch(int blocks, int blockSize, int n, int stages, int &stride, int *dev_odata)
        {
            // upsweep, write to temp buffer
            for (int d = 0; d <= stages; d++)
            {
                upsweep<<<blocks, blockSize>>>(n, stride, dev_odata);
                stride <<= 1;
            }

            for (int d = 0; d <= stages; d++)
            {
                downsweep<<<blocks, blockSize>>>(n, stride, dev_odata);
                stride >>= 1;
            }
        }


        void scan(int n, int *odata, const int *idata) {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_odata;

            int sizeOfData = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_odata, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int stages = ilog2(paddedN) - 1;
            int stride = 1;

            timer().startGpuTimer();

            scanDispatch(blocks, blockSize, paddedN, stages, stride, dev_odata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeOfData, cudaMemcpyDeviceToHost);
            
            cudaFree(dev_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */

        __global__ void assignBitmap(int n, int *dev_bitmap)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
            {
                return;
            }

            dev_bitmap[index] = (dev_bitmap[index] > 0) ? 1 : 0;
        }

        __global__ void scatter(int n, int *dev_bitmap, int *dev_odata, int *dev_idata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
            {
                return;
            }

            if (dev_bitmap[index] > 0)
            {
                int scatterIndex = dev_odata[index];
                dev_odata[scatterIndex] = dev_idata[index];
            }
        }

        int compact(int n, int *odata, const int *idata) 
        {
            int paddedN = 1 << ilog2ceil(n);
            int sizeOfData = paddedN * sizeof(int);

            int *dev_idata;
            int *dev_odata;
            int *dev_bitmap;

            cudaMalloc((void**)&dev_idata, sizeOfData);
            cudaMalloc((void**)&dev_odata, sizeOfData);
            cudaMalloc((void**)&dev_bitmap, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_bitmap, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int stages = ilog2(paddedN) - 1;
            int stride = 1;

            timer().startGpuTimer();

            // Write to dev_bitmap, input is idata memcpyed to dev_bitmap
            assignBitmap<<<blocks, blockSize>>>(paddedN, dev_bitmap);

            // Copy dev_bitmap info to dev_odata, this is needed so we can run scan
            cudaMemcpy(dev_odata, dev_bitmap, sizeOfData, cudaMemcpyDeviceToDevice);

            // Scan writes output to dev_odata
            scanDispatch(blocks, blockSize, paddedN, stages, stride, dev_odata);

            // Write to dev_odata with inputs idata, bitmap, and scanOutput, which is dev_odata at this point.
            scatter<<<blocks, blockSize>>>(paddedN, dev_bitmap, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeOfData, cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bitmap);

            return odata[paddedN - 1];
        }
    }
}
