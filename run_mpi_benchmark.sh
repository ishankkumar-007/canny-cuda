#!/bin/bash

# Compile the MPI C++ program
mpicxx -o canny_mpi canny_mpi.cpp -O2

# Output CSV file
output_file="mpi_timing_results.csv"
echo "Processes,TotalTime" > $output_file

# List of process counts
process_counts=(1 2 4 6 8 10 12 16 20)

# Run the program for each process count
for np in "${process_counts[@]}"; do
    echo "Running with $np processes..."

    # Run the program and capture output
    output=$(mpirun -np $np ./canny_mpi)

    # Extract total time using grep and awk
    total_time=$(echo "$output" | grep "Total execution time" | awk '{print $(NF-1)}')

    # Write to CSV
    echo "$np,$total_time" >> $output_file
done

echo "Benchmark completed. Results saved to $output_file"
