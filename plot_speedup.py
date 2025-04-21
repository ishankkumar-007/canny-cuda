import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("./mpi_timing_results.csv")

# Speedup Calculation
base_time = data["TotalTime"][0]

data["Speedup"] = base_time / data["TotalTime"]

# Plot Speedup
plt.figure(figsize=(10, 6))
plt.plot(data["Processes"], data["Speedup"], marker='o', label="Canny Edge Detection MPI")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.title("Speedup vs Processes")
plt.legend()
plt.grid()
plt.savefig("./mpi_speedup_plot.png")
plt.show()
