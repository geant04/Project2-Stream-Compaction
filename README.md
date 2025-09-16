CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Anthony Ge
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Personal)


## Parallel Algorithms Introduction
#### CPU Scan
#### Naive GPU Scan
#### Work-efficient GPU Scan
#### Optimized Work-efficient GPU Scan
#### Shared Memory Optimized Work-Efficient GPU Scan (not working)


## Performance Analysis

### Block Size Analysis + Dynamic Block Size Optimization
#### Testing Block Sizes
To test for optimal block size, I ran my work-efficient scan on different block sizes ranging from 64 to 512.

![block comparison](img/blockComparison.png)
![block comparison](img/blockComparisonNonPow2.png)

Through some quick testing, I found that a block size of 256 found optimal results for an array size of 2^22, and similarly 2^22-3 for the non-power-of-two test. When implementing the assignment, I arbitrarily chose 128 for my block size.

A quick idea for why larger block sizes can be better is because more warps can run per block. The SM can therefore utilize more warps to hide global memory latency that often come per stage in our work-efficient scan.

---
#### Dynamic Block Size
An optimization I used in my "optimized work efficient" is to **dynamically reduce the block-size** based on the number of threads we need to run in the first place. For example, if we're at a stage in our scan that runs on 64 pairs, but my block size is 128, I dynamically match to 64. This allows less work to run on the GPU, and therefore faster kernel throughput. Using 128 can lead to immediately inactive warps, along with the added overhead of dispatching more warps too.

Here's an example snippet from my code for upsweep, notice variables ```optimizedBlocks, optimizedBlockSize```:
```
for (int d = 0; d < stages; d++)
{
    upsweep<<<optimizedBlocks, optimizedBlockSize>>>(paddedN, stride, dev_odata); // dispatch kernel using dynamic # of blocks/block size

    stride <<= 1; // double stride for next pass

    // Threads to run are halved
    threadsToRun = paddedN >> (d + 1); // in each stage, we halve threads to run. This is important!
    optimizedN = threadsToRun;

    threadsToRun = (threadsToRun >= blockSize) ? blockSize : threadsToRun; // we keep our original blockSize if our # of threads to run is greater
    threadsToRun = (threadsToRun <= 32) ? 32 : threadsToRun; // no need to dispatch < 32 threads, a warp has 32 threads

    optimizedBlockSize = threadsToRun;
    optimizedBlocks = (optimizedN + optimizedBlockSize - 1) / optimizedBlockSize;
}
```
The same idea applies for downsweep. This leads to a much better overall throughput result.

Using NSight Compute, for an example size of 2^24, we can compare the ms duration of a upsweep stage of work-efficient vs. optimized work efficient, where one threadSize is 128 vs. 64 in a case when we need to run on 64 threads:

<table>
  <tr>
    <th>Work-efficient, (131072,1,1)x(128,1,1)</th>
    <th>Optimized, dynamic block num/size</th>
  </tr>
  <tr>
    <td> <img src="img/workEfficientKernel.png" width="500"></td>
    <td> <img src="img/optimizedWorkEfficientKernel.png" width="500"></td>
  </tr> 
</table>

The speed improvements are very apparent even after 8 stages, comparing **0.17ms vs 0.03ms between our baseline work-efficient and our dynamic,** which dynamically reduces our block size based on the number of threads to compute. 

---
#### Index Remap in Upsweep/Downsweep in Optimized Work-Efficient
The above optimization is only possible by restructing our sweep kernels to avoid modulos when checking for active threads to run. Instead, **only the first # of threads to run by index are ran, ensuring that using less blocks effectively gets rid of unused warps.**

```
// Within upsweep, my logic for running a thread is as below:

int strideMult2 = stride * 2;
if (index < (n / strideMult2))
{
    // .. Do work
}

// Compare to (index % stride), re-mapping our indices that do work helps us dispatch our blocks more smartly, letting us launch less warps since we only care about index < #.
```

---

### GPU Scan Implementations Analysis
For this analysis, I compared my naive, work-efficient, optimized work-efficient, and thrust algorithms on increasingly large power-of-two array sizes. 

![scan comparison](img/scanComparison.png)
![scan comparison non pow2](img/scanComparisonNonPow2.png)
---

### CPU Perf > GPU? Huh?
Though it's hard to see in the graphs, for array sizes < (2^18), or 262,144, the CPU scan implementation consistently outperformed all GPU implementations besides thrust.

Here are the results I logged directly, measurements in ms: 
| Array Size | CPU | Naive | Work Efficient | Opt. Work Efficient |
| ---------- | -------- | ----- | -------------- | ------------- |
| 1024 (2^10)      | **0.00294**  | 0.137 | 0.163          | 0.145         |
| 4096 (2^12)      | **0.0123**   | 0.134 | 0.212          | 0.222         |
| 16384 (2^14)     | **0.0425**   | 0.201 | 0.325          | 0.248         |
| 65536 (2^16)     | **0.198**    | 0.258 | 0.368          | 0.388         |

These results can most likely be attributed to the many stalls that occur during global memory read/writes, along with the several dispatches required for the GPU implementations to work. Meanwhile, the CPU has less overhead issues and can modify information much faster.

---
### Naive > Work-efficient? Seems odd, right?
In all cases as well, the work-efficient was always slower than the naive until (2^22), when work-efficient performed better on average by 0.04ms.

| Array Size | CPU | Naive | Work Efficient | Opt. Work Efficient |
| ---------- | -------- | ----- | -------------- | ------------- |
| 4194304 (2^22)    | 11.3146  | **1.873** | **1.833**          | 0.924         |

These results obviously surprised, as I assumed naturally that the work-efficient implementation should be faster than naive. However, this is mostly attributed to the work taken per kernel launch along with the observation that the work-efficient implementation simply launches more dispatches.

We can first observe, in NSight Compute, that every kernel ran in naive runs for roughly the same duration per dispatch. On 2^12 elements, we see each naive kernel take roughly ~2.53us duration. In total, we run for ~36.51 us for log(n) + 1 dispatchs to convert inclusive to exclusive.

![](img/naiveTimings.png)


With our work-efficient implementation which will have many unused warps due to its static block size/block number, we can interestingly note that most dispatches on the upsweep/downswep also take ~2.56us. Because of having both upsweep/downsweep, **we perform double the number of dispatches, 2 * log(n), compared to naive.** Thus, it's easier to understand why work-efficient runs worse, in total taking ~65.18us.

![](img/workEfficientTimings.png)

*There are so many dispatches that it's not even worth fitting in the image.*

It's ultimately surprising to see that our optimized work efficient perform better on average, given the circumstances and the same number of dispatches ran. It's just the case that each kernel is fast enough that the bottleneck is on kernel performance instead of # of dispatches.

---
### The General Bottleneck on GPU
Independent of calling many dispatches, the common bottlenecks in all kernels are the memory stalls from reading global memory. In NSight Compute using the Warp Stall Sampling metric (enabled in Metrics, I kind of just clicked full to get the most detail), we can see that most occurences happen during data load instructions.

![](img/naiveStall.png)

Here is a screenshot of naive showing ~41% stall sampling, easily greater than the other stalls in the kernel.

![](img/upsweepStall.png)

This screenshot is from the upsweep kernel, showing similar results.

Problems like these can therefore be circumvented using shared memory, which is much faster to access than global memory. Ideally, we would be able to quickly populated shared mem, sync threads, then only use shared mem for our kernel computes. 

---

Using an alternative method of performing scan using shared memory by performing sub-scans on blocks, running another scan on an array of block reductions and then re-adding respective block elemens to original values in blocks, I was able to reach remarkable speedups!

![](img/optimizedSharedMem.png)

In this case, for N=2^12 (4096) **we went from 0.422ms to 0.244ms.** However I was only able to achieve this up until 2^14, before I started running into issues regarding max block size and SM memory limits. I unfortunately was not able to fix it and thus omitted this implementation in any of my performance readings, though I'm still optimistic that this can provide some nice perf wins.

---
### Test Program Output
Erm...