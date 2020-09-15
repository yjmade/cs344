/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

const unsigned int NUMTHREAD = 1024;

template<typename funcT>
__global__ void reduce_kernel(float* d_array, size_t length, funcT op){
    int steps=ceilf(log2f(length));
    unsigned int total_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_length = blockDim.x;
    if (total_id/blockDim.x == blockIdx.x - 1){
        // last block
        block_length = length % blockDim.x;
    }
    // copy data to shared memory
    __shared__ float shared_array[NUMTHREAD];
    if (threadIdx.x<block_length){
        shared_array[threadIdx.x] = d_array[total_id];
    }
    __syncthreads();

    for (unsigned int step=0; step<steps; step++){
        unsigned int step_size = (block_length+1)/2;
        if ((threadIdx.x + step_size) < block_length)
        {
            shared_array[threadIdx.x] = op(shared_array[threadIdx.x], shared_array[threadIdx.x+step_size]);
        }
        block_length=step_size;
        __syncthreads();
    }
    d_array[blockIdx.x]=shared_array[0];
}
// __global__ void reduce_max(float* const d_array, float* max_value)
// __global__ void fused_reduce_min_max(float* const d_array, float* min_value, float* max_value)
__device__ float (*my_fminf)(float, float) = fminf;
__device__ float (*my_fmaxf)(float, float) = fmaxf;

template<typename funcT>
float reduce(const float* const d_array, const size_t length, funcT op){
    uint steps = ceilf(log2f(length));
    int numThreads = NUMTHREAD;
    uint iters = ceilf(steps/10.);


    float *d_result;
    checkCudaErrors(cudaMalloc(&d_result, sizeof(float)*length));
    checkCudaErrors(cudaMemcpy(d_result, d_array, sizeof(float) * length, cudaMemcpyDeviceToDevice));
    size_t current_length = length;
    for (int iter = 0; iter < iters; iter++)
    {
        // std::cout<<"current length"<< current_length<<std::endl;
        size_t block_count = (current_length+numThreads-1)/numThreads;
        if (block_count==1)
            numThreads = current_length;
        // std::cout << "kernel size" << block_count<<","<<numThreads << std::endl;
        reduce_kernel<<<block_count, numThreads>>>(d_result, current_length, op);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        current_length = block_count;
    }
    // float *h_result;
    // h_result=(float *)malloc(sizeof(float));
    float h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(float) * 1, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return h_result;

}

__global__ void histogram_kernel(unsigned int* d_output, const float *const d_input, size_t length, const float min_value, const float interval)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    float value = d_input[id];
    if(id<length){
        unsigned int bin_id = (value - min_value) / interval;
        atomicAdd(d_output + bin_id, 1);
    }
}

__global__ void print_int_kernel(const uint *const d_input, uint length){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    int value = d_input[id];
    if (id < length)
    {
        printf("id[%d]=%d\n", id, value);
    }
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                 unsigned int *const d_cdf,
                                 float &min_logLum,
                                 float &max_logLum,
                                 const size_t numRows,
                                 const size_t numCols,
                                 const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    const size_t length=numRows*numCols;
    float (*h_fminf)(float, float);
    cudaMemcpyFromSymbol(&h_fminf, my_fminf, sizeof(void *)); // https://stackoverflow.com/questions/38133314/function-as-templated-parameter-in-cuda
    float (*h_fmaxf)(float, float);
    cudaMemcpyFromSymbol(&h_fmaxf, my_fmaxf, sizeof(void *));
    min_logLum = reduce(d_logLuminance, length, h_fminf);
    max_logLum = reduce(d_logLuminance, length, h_fmaxf);
    std::cout << "min_value:" << min_logLum<< std::endl;
    std::cout << "max_value:" << max_logLum<< std::endl;
    float logLum_range = max_logLum - min_logLum;

    float offset_per_bin = logLum_range / numBins;
    // checkCudaErrors(cudaMemset())
    histogram_kernel<<<(length + NUMTHREAD - 1) / NUMTHREAD, NUMTHREAD>>>(d_cdf, d_logLuminance, length, min_logLum, offset_per_bin);
    // print_int_kernel<<<(length + NUMTHREAD - 1) / NUMTHREAD, NUMTHREAD>>>(d_cdf, numBins);
    // float result[length];
    // checkCudaErrors(cudaMemcpy(result, d_logLuminance, sizeof(float)*length, cudaMemcpyDeviceToHost));
    // printf("min_value %.5f\n", min_logLum);
    // // max_logLum = reduce_max(d_logLuminance, length);
    // for(int i=0; i < length; i++){
    //     // std::cout<<*(d_logLuminance)<<",";
    //     printf("%f, ", result[i]);
    // }
    // printf("\n");
}
