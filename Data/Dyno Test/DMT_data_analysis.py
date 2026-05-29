import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## STRAIN GAUGE ##
# File path
file_path = r"pretty good on calibrated dyno.txt"

# Read data:
# - skip first header line
# - use tab separation
# - manually define columns
df = pd.read_csv(
    file_path,
    sep=r"\t+",
    engine="python",
    skiprows=1,
    header=None,
    names=[
        "Scan",
        "Time",
        "Ch1",
        "Ch2",
        "Ch3",
        "Ch5",
        "Ch6",
        "Ch7",
        "Ch8"
    ]
)

# Remove completely empty columns if they appear
df = df.dropna(axis=1, how="all")

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert time to seconds
df["Time_s"] = df["Time"] / 1e6

# Zero offset straingauge readings and double check positive/negative

# Convert channels to straingauges
straingauges = ["St1", "St2", "St3", "St5", "St6", "St9", "St8"]

# Aluminium 7075 T6 properties (https://ntrs.nasa.gov/api/citations/19670004550/downloads/19670004550.pdf)
yield_strength = 503 #MPa
UTS = 572 #MPa
max_elongation = 0.11
Youngs_modulus = 71.7 #GPa

# Convert strains to stresses

# Import expected stresses (for what load setting)


## ENGINE ##
# Data logger
data_logging_frequency = 20 #Hz
timestep = 1/20 #s

# Import engine data


gear = 1
rpm = 1

# fix gear indexing (6th is -1)
for i in gear:
    if i == -1: 
        i = 6


# Gearbox
sprocket_ratio = 40/14
primary_reduction_ratio	= 71/32
gearbox = [41/14, 37/18, 34/21, 32/24, 30/26, 28/27]
num_gears = len(gearbox)

# Parameters
rpm_max = 12000 # boundary value

power = 0.002573*rpm**6 + -0.113906*rpm**5 + 2.070404*rpm**4 + -19.936337*rpm**3 + 107.415328*rpm**2 + -300.565432*rpm + 356.299961
torque = 0.002450*rpm**6 + -0.116268*rpm**5 + 2.280364*rpm**4 + -23.689543*rpm**3 + 136.857560*rpm**2 + -413.597675*rpm + 543.075438

# Calculate expected loads on sideplates based on torque

# Speed
r_tire = 18.1 * 0.0254 / 2 # m (conversion from in to m)
speed = rpm/30*np.pi()*primary_reduction_ratio*gearbox[gear-1]*sprocket_ratio*r_tire # m/s


## PLOTTING ##
# Ploting strain gauge recording
plt.figure(figsize=(12, 6))

channels = ["Ch1", "Ch2", "Ch3", "Ch5", "Ch6", "Ch7", "Ch8"]

for ch in channels:
    plt.plot(df["Time_s"], df[ch], label=ch)

plt.xlabel("Time [s]")
plt.ylabel("Microstrain")
plt.title("Strain Gauge Data")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()