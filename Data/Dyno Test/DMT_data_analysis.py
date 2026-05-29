import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

## STRAIN GAUGE ##
# File path
file_path = Path(__file__).resolve().parent / "pretty good on calibrated dyno.txt"

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

# Import expected stresses from FEA (for what load setting)


## ENGINE ##

# Import engine data
engine_data_dir = Path(__file__).resolve().parent / "Testing data"
engine_csv_files = [engine_data_dir / f"{i}.csv" for i in range(50, 63)]
manual_time_gaps = {i: 0.0 for i in range(50, 63)}  # extra gap before each file starts
manual_time_offsets = {i: 0.0 for i in range(50, 63)}  # per-file manual alignment shift


def _read_aim_csv_with_metadata(csv_file, time_start=0.0):
    metadata = pd.read_csv(csv_file, nrows=13, header=None, dtype=str)
    metadata = {
        str(row[0]).strip(): str(row[1]).strip()
        for _, row in metadata.iterrows()
        if len(row) >= 2 and str(row[0]).strip() != ""
    }

    rate = float(metadata.get("Sample Rate", 20.0))
    duration = float(metadata.get("Duration", np.nan))

    header_row = pd.read_csv(csv_file, header=None, skiprows=14, nrows=1)
    header = [str(x).strip() for x in header_row.iloc[0].tolist()]

    df = pd.read_csv(csv_file, header=None, skiprows=15)
    df.columns = header
    df = df.iloc[1:].reset_index(drop=True)  # drop the unit row

    df = df.apply(pd.to_numeric, errors="coerce")
    row_count = len(df)
    time_step = 1.0 / rate

    df["Time"] = np.arange(row_count, dtype=float) * time_step + time_start
    if not np.isnan(duration):
        df["Time_duration"] = np.linspace(time_start, time_start + duration, row_count, endpoint=False)
    else:
        df["Time_duration"] = df["Time"]

    file_number = int(csv_file.stem)
    df["Time_manual"] = df["Time"] + manual_time_offsets.get(file_number, 0.0)

    return df, time_start + row_count * time_step, metadata


engine_data_frames = []
engine_time_start = 0.0
for csv_file in engine_csv_files:
    file_number = int(csv_file.stem)
    file_gap = manual_time_gaps.get(file_number, 0.0)
    engine_time_start += file_gap
    file_df, engine_time_start, metadata = _read_aim_csv_with_metadata(csv_file, engine_time_start)
    engine_data_frames.append(file_df)

engine_data = pd.concat(engine_data_frames, ignore_index=True)

# Extra time arrays for plotting and alignment
engine_time = engine_data["Time"].to_numpy()
engine_time_duration = engine_data["Time_duration"].to_numpy()
engine_time_manual = engine_data["Time_manual"].to_numpy()

# Extract and process engine data
gear = engine_data["Gear"].copy()
gear = gear.replace(-1, 6) # fix gear indexing (6th is -1)
gear = gear.fillna(0).astype(int).to_numpy()


rpm_idle = 1300.0
rpm_join = 4000.0
rpm_max = 12000.0

rpm = np.clip(engine_data["MS3 RPM"].to_numpy(),0, rpm_max)  # RPM clipped
lateral_accel = engine_data["LateralAcc"].to_numpy()
vertical_accel = engine_data["VerticalAcc"].to_numpy()
tps = np.clip(engine_data["MS3 TPS"].to_numpy() / 100, 0, 1)
# Nonlinear TPS scaling: model is a smooth cubic-ish throttle opening law
gamma = 2.2


# Gearbox
sprocket_ratio = 40 / 14
primary_reduction_ratio = 71 / 32
gearbox = np.array([41 / 14, 37 / 18, 34 / 21, 32 / 24, 30 / 26, 28 / 27])
num_gears = len(gearbox)

valid_gear_index = np.where((gear >= 1) & (gear <= num_gears), gear - 1, -1)
gear_ratio = np.where(valid_gear_index >= 0, gearbox[valid_gear_index], np.nan)

# Parameters: WOT curves (user polynomials) and extended low-rpm behaviour
rpm_idle = 1300.0
rpm_join = 4000.0

def wot_power_kw(rpm):
    x = np.asarray(rpm, dtype=float) / 1000.0
    return (
        0.002573 * x**6
        - 0.113906 * x**5
        + 2.070404 * x**4
        - 19.936337 * x**3
        + 107.415328 * x**2
        - 300.565432 * x
        + 356.299961
    )


def wot_torque_nm(rpm):
    x = np.asarray(rpm, dtype=float) / 1000.0
    return (
        0.002450 * x**6
        - 0.116268 * x**5
        + 2.280364 * x**4
        - 23.689543 * x**3
        + 136.857560 * x**2
        - 413.597675 * x
        + 543.075438
    )


def smoothstep(rpm, rpm0=rpm_idle, rpm1=rpm_join):
    u = (np.asarray(rpm, dtype=float) - rpm0) / (rpm1 - rpm0)
    u = np.clip(u, 0.0, 1.0)
    return 3 * u**2 - 2 * u**3


T4000 = float(wot_torque_nm(rpm_join))
P4000 = float(wot_power_kw(rpm_join))


def extended_torque_nm(rpm):
    rpm = np.asarray(rpm, dtype=float)
    base = wot_torque_nm(rpm)
    s = smoothstep(rpm)
    low = s * T4000
    return np.where(rpm < rpm_join, low, base)


def extended_power_kw(rpm):
    rpm = np.asarray(rpm, dtype=float)
    base = wot_power_kw(rpm)
    s = smoothstep(rpm)
    low = s * P4000
    return np.where(rpm < rpm_join, low, base)

# Compute driven power (hp) and torque (Nm), scaled by TPS
power = (tps**gamma) * extended_power_kw(rpm)
torque = (tps**gamma) * extended_torque_nm(rpm)

# Force zero when gear == 0
gear_zero_mask = (gear == 0)
power[gear_zero_mask] = 0.0
torque[gear_zero_mask] = 0.0

shaft_torque = torque * primary_reduction_ratio * gear_ratio * sprocket_ratio

# Calculate expected loads on sideplates based on torque
sprocket_diameter = 0.22457 # m (from CAD)

# Speed
r_tire = 18.1 * 0.0254 / 2 # m (conversion from in to m)
speed = rpm/30*np.pi*primary_reduction_ratio*gear_ratio*sprocket_ratio*r_tire # m/s


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

# Plotting engine data
plt.figure(figsize=(14, 7))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(engine_time, rpm/100, color="tab:blue", label="RPM in 100/min")
ax1.plot(engine_time, tps*100, color="gold", label="TPS in %")
ax1.plot(engine_time, torque, color="tab:orange", label="Torque")
ax1.plot(engine_time, power, color="tab:red", label="Power")
ax2.step(engine_time, gear, where="post", color="tab:green", label="Gear", linewidth=1.5)

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("RPM / TPS / Torque / Power", color="tab:blue")
ax2.set_ylabel("Gear", color="tab:green")
ax1.set_title("Engine data: RPM, Gear, TPS, Torque, and Power")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

ax1.grid(True)
plt.tight_layout()
plt.show()
