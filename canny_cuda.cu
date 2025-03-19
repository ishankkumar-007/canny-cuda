// to run:
// nvcc canny_cuda.cu -arch=sm_60 -lcudart && ./a.out

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;

#define BLOCK_SIZE 16

// Global Variables
int width = 512, height = 512;
double sig = 1.0;

// CUDA Kernels
__global__ void gaussianDerivativeKernel(double* maskX, double* maskY, int dim, double sig) {
    int p = threadIdx.x - dim / 2;
    int q = threadIdx.y - dim / 2;
     
    if (p >= -dim / 2 && p <= dim / 2 && q >= -dim / 2 && q <= dim / 2) {
        int index = (p + dim / 2) * dim + (q + dim / 2);
        maskX[index] = q * exp(-1 * ((p * p + q * q) / (2 * sig * sig)));
        maskY[index] = p * exp(-1 * ((p * p + q * q) / (2 * sig * sig)));
    }
}

__global__ void convolutionKernel(double* pic, double* mag, double* x, double* y, double* maskX, double* maskY, int width, int height, int dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < height && j < width) {
        double sumX = 0, sumY = 0;
        for (int p = -dim / 2; p <= dim / 2; p++) {
            for (int q = -dim / 2; q <= dim / 2; q++) {
                int newI = i + p;
                int newJ = j + q;
                if (newI >= 0 && newI < height && newJ >= 0 && newJ < width) {
                    int maskIndex = (p + dim / 2) * dim + (q + dim / 2);
                    int picIndex = newI * width + newJ;
                    sumX += pic[picIndex] * maskX[maskIndex];
                    sumY += pic[picIndex] * maskY[maskIndex];
                }
            }
        }
        
        int index = i * width + j;
        x[index] = sumX;
        y[index] = sumY;
        mag[index] = sqrt(sumX * sumX + sumY * sumY);
    }
}

void cannyEdgeDetectionCUDA(double* h_pic, double* h_mag, int width, int height, double sig) {
    int size = width * height * sizeof(double);
    double *d_pic, *d_mag, *d_x, *d_y, *d_maskX, *d_maskY;
    int dim = 6 * sig + 1;
    int maskSize = dim * dim * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_pic, size);
    cudaMalloc((void**)&d_mag, size);
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMalloc((void**)&d_maskX, maskSize);
    cudaMalloc((void**)&d_maskY, maskSize);

    // Copy input image to device
    cudaMemcpy(d_pic, h_pic, size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Compute Gaussian derivative masks
    gaussianDerivativeKernel<<<1, dim3(dim, dim)>>>(d_maskX, d_maskY, dim, sig);
    cudaDeviceSynchronize();

    // Apply convolution to compute gradients and magnitude
    convolutionKernel<<<gridDim, blockDim>>>(d_pic, d_mag, d_x, d_y, d_maskX, d_maskY, width, height, dim);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_mag, d_mag, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pic);
    cudaFree(d_mag);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_maskX);
    cudaFree(d_maskY);
}

int main() {
    // Allocate host memory
    double *pic = new double[width * height];
    double *mag = new double[width * height];

    // Read input PGM image
    ifstream infile("chess.pgm", ios::binary);
    if (!infile) {
        cout << "Error: Could not open input file." << endl;
        return 1;
    }

    // Read PGM header
    string magic;
    int maxVal;
    infile >> magic >> width >> height >> maxVal;
    infile.ignore(1); // Ignore single whitespace

    // Ensure PGM format is valid
    if (magic != "P5") {
        cout << "Error: Only P5 format is supported." << endl;
        return 1;
    }

    // Read pixel data as unsigned char and convert to double
    unsigned char* inputBuffer = new unsigned char[width * height];
    infile.read((char*)inputBuffer, width * height);
    infile.close();

    for (int i = 0; i < width * height; i++) {
        pic[i] = static_cast<double>(inputBuffer[i]);
    }
    delete[] inputBuffer;

    // Run Canny Edge Detection on GPU
    cannyEdgeDetectionCUDA(pic, mag, width, height, sig);

    // Normalize magnitude values to 0-255
    double maxValMag = 0;
    for (int i = 0; i < width * height; i++) {
        if (mag[i] > maxValMag) {
            maxValMag = mag[i];
        }
    }
    
    unsigned char* outputBuffer = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++) {
        outputBuffer[i] = static_cast<unsigned char>((mag[i] / maxValMag) * 255.0);
    }

    // Write output PGM file
    ofstream outfile("output.pgm", ios::binary);
    if (!outfile) {
        cout << "Error: Could not create output file." << endl;
        return 1;
    }

    // Write PGM header
    outfile << "P5\n" << width << " " << height << "\n255\n";
    
    // Write pixel data
    outfile.write((char*)outputBuffer, width * height);
    outfile.close();

    // Free host memory
    delete[] pic;
    delete[] mag;
    delete[] outputBuffer;

    cout << "Canny Edge Detection completed. Output saved to output.pgm" << endl;
    return 0;
}
