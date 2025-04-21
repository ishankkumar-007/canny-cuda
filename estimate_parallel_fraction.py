import pandas as pd

# Load performance data
data = pd.read_csv("./mpi_timing_results.csv")

# Speedup Calculation
base_time_critical = data["TotalTime"][0]

data["Speedup"] = base_time_critical / data["TotalTime"]

# Compute Parallel Fraction (P) for each thread count
def compute_parallel_fraction(N, S):
    """Compute parallel fraction using Amdahl's Law: P = (1 - 1/S) / (1 - 1/N)"""
    if N == 1:  # If single thread, parallel fraction is undefined
        return 0
    return (1 - 1/S) / (1 - 1/N) if S > 1 else 0  # Avoid division by zero

data["Parallel_Fraction"] = data.apply(lambda row: compute_parallel_fraction(row["Processes"], row["Speedup"]), axis=1)

# Print results
# print(data[["Processes", "Parallel_Fraction", "Speedup"]])
print(data)

# # Save the results to a new CSV file
# data.to_csv("performance_with_parallel_fraction.csv", index=False)
