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

### Block Size Analysis
To test for optimal block size, I ran my work-efficient scan on different block sizes ranging from 64 to 512.

![block comparison](img/blockComparison.png)
![block comparison](img/blockComparisonNonPow2.png)

---

### GPU Scan Implementations Analysis
For this analysis, I compared my naive, work-efficient, optimized work-efficient, and thrust algorithms on increasingly large power-of-two array sizes. 

![scan comparison](img/scanComparison.png)
![scan comparison non pow2](img/scanComparisonNonPow2.png)

#### Brief Explanation of the Phenomena

#### NSight Compute Kernel Comparison (ms)
I noticed many variations in the performance between my different algorithms and took a capture in NSight on an array size of ???? to verify kernel durations.

