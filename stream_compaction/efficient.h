#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        
        void optimizedScan(int n, int *odata, const int *idata);

        void optimizedMemScan(int n, int *odata, const int *data);

        int compact(int n, int *odata, const int *idata);
    }
}
