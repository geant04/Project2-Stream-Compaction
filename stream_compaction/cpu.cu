#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void computeScan(int n, int *odata, const int *idata)
        {
            odata[0] = 0;

            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            computeScan(n, odata, idata);

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int odataIndex = 0;
            for (int i = 0; i < n; i++)
            {
                int inputValue = idata[i];

                if (inputValue != 0)
                {
                    odata[odataIndex] = inputValue;
                    odataIndex++;
                }
            }

            timer().endCpuTimer();
            return odataIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // t/f array is != 0 check
            int *bitmap = new int[n];

            for (int i = 0; i < n; i++)
            {
                bitmap[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Scan this stuff
            int *scanOut = new int[n];
            computeScan(n, scanOut, bitmap);

            // Scatter
            int elements = 0;

            for (int i = 0; i < n; i++)
            {
                if (bitmap[i] != 0)
                {
                    int finalOutIndex = scanOut[i];
                    odata[finalOutIndex] = idata[i];
                    elements++;
                }
            }

            timer().endCpuTimer();
            delete[] bitmap;
            delete[] scanOut;
            return elements;
        }
    }
}
