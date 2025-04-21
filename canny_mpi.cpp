// Compile & Run:
// mpic++ canny_mpi.cpp -o canny_mpi && mpirun -np 4 ./canny_mpi

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

int width, height;
double sig = 1.0;

bool readPGM(const string &filename, double *&img)
{
    ifstream infile(filename, ios::binary);
    if (!infile)
        return false;

    string magic;
    int maxVal;
    infile >> magic >> width >> height >> maxVal;
    infile.ignore();

    if (magic != "P5")
        return false;

    unsigned char *buffer = new unsigned char[width * height];
    infile.read((char *)buffer, width * height);
    infile.close();

    img = new double[width * height];
    for (int i = 0; i < width * height; i++)
        img[i] = static_cast<double>(buffer[i]);

    delete[] buffer;
    return true;
}

void savePGM(const string &filename, double *image, int width, int height)
{
    ofstream outfile(filename, ios::binary);
    outfile << "P5\n"
            << width << " " << height << "\n255\n";

    unsigned char *outputBuffer = new unsigned char[width * height];
    double maxVal = *max_element(image, image + (width * height));
    if (maxVal == 0)
        maxVal = 1;

    for (int i = 0; i < width * height; i++)
        outputBuffer[i] = static_cast<unsigned char>((image[i] / maxVal) * 255.0);

    outfile.write((char *)outputBuffer, width * height);
    delete[] outputBuffer;
}

void convolve(double *input, double *output, int local_height, int width, int dim, double **maskX, double **maskY)
{
    int cent = dim / 2;

    for (int i = cent; i < local_height - cent; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double sumX = 0.0, sumY = 0.0;
            for (int p = -cent; p <= cent; p++)
            {
                for (int q = -cent; q <= cent; q++)
                {
                    int y = i + p;
                    int x = j + q;
                    if (x >= 0 && x < width && y >= 0 && y < local_height)
                    {
                        sumX += input[y * width + x] * maskX[p + cent][q + cent];
                        sumY += input[y * width + x] * maskY[p + cent][q + cent];
                    }
                }
            }
            output[(i - cent) * width + j] = sqrt(sumX * sumX + sumY * sumY);
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    double *fullImage = nullptr;
    double *fullOutput = nullptr;

    int dim = 6 * sig + 1;
    int cent = dim / 2;

    double totalStart = MPI_Wtime(); // Total timer start

    if (rank == 0)
    {
        if (!readPGM("chess.pgm", fullImage))
        {
            cout << "Failed to load image\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsPerProc = height / numProcs;
    int extra = height % numProcs;
    int local_height = rowsPerProc + (rank < extra ? 1 : 0) + 2 * cent;

    vector<int> sendCounts(numProcs), displs(numProcs);
    int offset = 0;
    for (int i = 0; i < numProcs; i++)
    {
        int rows = rowsPerProc + (i < extra ? 1 : 0);
        sendCounts[i] = (rows + 2 * cent) * width;
        displs[i] = (offset - cent) * width;
        offset += rows;
    }

    double *localInput = new double[local_height * width];
    double *localOutput = new double[(local_height - 2 * cent) * width];

    if (rank == 0)
    {
        for (int i = 0; i < numProcs; i++)
        {
            int startRow = displs[i] / width;
            int sendRows = sendCounts[i] / width;

            if (i == 0)
            {
                memcpy(localInput, fullImage, sendCounts[0] * sizeof(double));
            }
            else
            {
                MPI_Send(&fullImage[startRow * width], sendCounts[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        MPI_Recv(localInput, local_height * width, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double **maskX = new double *[dim];
    double **maskY = new double *[dim];
    for (int i = 0; i < dim; ++i)
    {
        maskX[i] = new double[dim];
        maskY[i] = new double[dim];
    }

    for (int p = -cent; p <= cent; ++p)
    {
        for (int q = -cent; q <= cent; ++q)
        {
            double val = exp(-((p * p + q * q) / (2 * sig * sig)));
            maskX[p + cent][q + cent] = q * val;
            maskY[p + cent][q + cent] = p * val;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double localStart = MPI_Wtime();

    convolve(localInput, localOutput, local_height, width, dim, maskX, maskY);

    double localEnd = MPI_Wtime();
    double localTime = localEnd - localStart;

    vector<int> recvCounts(numProcs), recvDispls(numProcs);
    offset = 0;
    for (int i = 0; i < numProcs; i++)
    {
        int rows = rowsPerProc + (i < extra ? 1 : 0);
        recvCounts[i] = rows * width;
        recvDispls[i] = offset * width;
        offset += rows;
    }

    if (rank == 0)
        fullOutput = new double[width * height];

    MPI_Gatherv(localOutput, (local_height - 2 * cent) * width, MPI_DOUBLE,
                fullOutput, recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double totalEnd = MPI_Wtime();
    double totalTime = totalEnd - totalStart;

    // Gather and print per-process timings
    vector<double> allTimes(numProcs);
    MPI_Gather(&localTime, 1, MPI_DOUBLE, allTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "\n========= Timing Report =========\n";
        for (int i = 0; i < numProcs; i++)
            cout << "Process " << i << " computation time: " << allTimes[i] << " seconds\n";

        cout << "Total execution time (including IO and communication): " << totalTime << " seconds\n";

        savePGM("output_mpi.pgm", fullOutput, width, height);
        cout << "Output saved as output_mpi.pgm\n";
        cout << "=================================\n";
    }

    for (int i = 0; i < dim; ++i)
    {
        delete[] maskX[i];
        delete[] maskY[i];
    }
    delete[] maskX;
    delete[] maskY;
    delete[] localInput;
    delete[] localOutput;
    if (rank == 0)
    {
        delete[] fullImage;
        delete[] fullOutput;
    }

    MPI_Finalize();
    return 0;
}
