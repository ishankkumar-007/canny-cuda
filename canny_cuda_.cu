// Compile & Run:
// nvcc canny_cuda_speedup_fixed.cu -arch=sm_60 -lcudart && ./a.out

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>  // Include for max_element()

using namespace std;
using namespace chrono;

#define BLOCK_SIZE 16

// Global Variables
int width, height;
double sig = 1.0;

// =============================
// Read PGM Image
// =============================
bool readPGM(const string& filename, double*& img) {
    ifstream infile(filename, ios::binary);
    if (!infile) {
        cout << "Error: Could not open input file." << endl;
        return false;
    }

    string magic;
    int maxVal;
    infile >> magic >> width >> height >> maxVal;
    infile.ignore(1);  // Skip the newline character after the header

    if (magic != "P5") {
        cout << "Error: Only P5 format is supported." << endl;
        return false;
    }

    unsigned char* buffer = new unsigned char[width * height];
    infile.read((char*)buffer, width * height);
    infile.close();

    img = new double[width * height];
    for (int i = 0; i < width * height; i++) {
        img[i] = static_cast<double>(buffer[i]);
    }
    delete[] buffer;
    return true;
}

// =============================
// Save PGM Image
// =============================
void savePGM(const string& filename, double* image, int width, int height) {
    ofstream outfile(filename, ios::binary);
    outfile << "P5\n" << width << " " << height << "\n255\n";

    unsigned char* outputBuffer = new unsigned char[width * height];

    // Ensure proper normalization of pixel values
    double maxVal = *max_element(image, image + (width * height));
    if (maxVal == 0) maxVal = 1; // Avoid division by zero

    for (int i = 0; i < width * height; i++) {
        outputBuffer[i] = static_cast<unsigned char>((image[i] / maxVal) * 255.0);
    }
    outfile.write((char*)outputBuffer, width * height);
    outfile.close();
    delete[] outputBuffer;
}

// =============================
// Serial CPU Implementation
// =============================
void cannyEdgeDetectionSerial(double* pic, double* mag, int width, int height) {
    int dim = 6 * sig + 1, cent = dim / 2;
    vector<vector<double>> maskX(dim, vector<double>(dim));
    vector<vector<double>> maskY(dim, vector<double>(dim));

    for (int p = -cent; p <= cent; p++) {
        for (int q = -cent; q <= cent; q++) {
            maskX[p + cent][q + cent] = q * exp(-1 * ((p * p + q * q) / (2 * sig * sig)));
            maskY[p + cent][q + cent] = p * exp(-1 * ((p * p + q * q) / (2 * sig * sig)));
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double sumX = 0, sumY = 0;
            for (int p = -cent; p <= cent; p++) {
                for (int q = -cent; q <= cent; q++) {
                    int newI = i + p, newJ = j + q;
                    if (newI >= 0 && newI < height && newJ >= 0 && newJ < width) {
                        sumX += pic[newI * width + newJ] * maskX[p + cent][q + cent];
                        sumY += pic[newI * width + newJ] * maskY[p + cent][q + cent];
                    }
                }
            }
            mag[i * width + j] = sqrt(sumX * sumX + sumY * sumY);
        }
    }
}

// =============================
// CUDA Kernels
// =============================
__global__ void convolutionKernel(double* pic, double* mag, int width, int height) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < height && j < width) {
        double sumX = 0, sumY = 0;
        for (int p = -3; p <= 3; p++) {
            for (int q = -3; q <= 3; q++) {
                int newI = i + p;
                int newJ = j + q;
                if (newI >= 0 && newI < height && newJ >= 0 && newJ < width) {
                    int picIndex = newI * width + newJ;
                    sumX += pic[picIndex] * q;
                    sumY += pic[picIndex] * p;
                }
            }
        }
        
        int index = i * width + j;
        mag[index] = sqrt(sumX * sumX + sumY * sumY);
    }
}

// =============================
// GPU Implementation (Returns Kernel Execution Time)
// =============================
double cannyEdgeDetectionCUDA(double* h_pic, double* h_mag, int width, int height) {
    int size = width * height * sizeof(double);
    double *d_pic, *d_mag;

    cudaMalloc((void**)&d_pic, size);
    cudaMalloc((void**)&d_mag, size);

    cudaMemcpy(d_pic, h_pic, size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    convolutionKernel<<<gridDim, blockDim>>>(d_pic, d_mag, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(h_mag, d_mag, size, cudaMemcpyDeviceToHost);

    cudaFree(d_pic);
    cudaFree(d_mag);

    return kernelTime / 1000.0;
}

// =============================
// Main Function
// =============================
int main() {
    double* pic;
    if (!readPGM("chess.pgm", pic)) return 1;

    double* magSerial = new double[width * height];
    double* magGPU = new double[width * height];

    auto start = high_resolution_clock::now();
    cannyEdgeDetectionSerial(pic, magSerial, width, height);
    auto stop = high_resolution_clock::now();
    double serialTime = duration<double>(stop - start).count();
    cout << "Serial Execution Time: " << serialTime << " sec\n";

    double parallelTime = cannyEdgeDetectionCUDA(pic, magGPU, width, height);
    cout << "CUDA Kernel Execution Time: " << parallelTime << " sec\n";

    double speedup = serialTime / parallelTime;
    cout << "Speedup (Serial / CUDA Kernel): " << speedup << "x\n";

    savePGM("output_serial.pgm", magSerial, width, height);
    savePGM("output_gpu.pgm", magGPU, width, height);

    delete[] pic;
    delete[] magSerial;
    delete[] magGPU;
    return 0;
}
