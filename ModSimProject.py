import matplotlib
matplotlib.use('TkAgg')
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import os
import tempfile
from PIL import Image
import mplcursors
plt.ioff()

# Constants
FLOOR_AREA = 124.87  # m²
ROOF_AREA = 137.94  # m²
THICKNESS_ROOF = 0.001  # m (metal sheet only)
THICKNESS_INSULATION = 0.05  # m (insulation layer, e.g., fiberglass)
DENSITY_METAL = 7850  # kg/m³
DENSITY_INSULATION = 50  # kg/m³ (typical for fiberglass)
SPECIFIC_HEAT_METAL = 500  # J/kg·K
SPECIFIC_HEAT_INSULATION = 840  # J/kg·K (fiberglass)
THERMAL_CONDUCTIVITY_METAL = 50  # W/m·K (steel)
THERMAL_CONDUCTIVITY_INSULATION = 0.5  # W/m·K (fiberglass)
MASS_ROOF = (ROOF_AREA * THICKNESS_ROOF * DENSITY_METAL) + (ROOF_AREA * THICKNESS_INSULATION * DENSITY_INSULATION)  # kg
H_COMBINED = 40  # W/m²·K
WALL_AREA = 131.61  # m²
THICKNESS_WALL = 0.1  # m (concrete wall)
DENSITY_WALL = 2400  # kg/m³ (concrete)
SPECIFIC_HEAT_WALL = 900  # J/kg·K (concrete)
MASS_WALL = WALL_AREA * THICKNESS_WALL * DENSITY_WALL  # kg
WINDOW_AREA = 3.716  # m²
H_WALL = 5  # W/m²·K
H_WINDOW = 20  # W/m²·K
AIR_DENSITY = 1.2  # kg/m³
AIR_SPECIFIC_HEAT = 1005  # J/kg·K
VOLUME = 2.5 * FLOOR_AREA  # m³
MASS_AIR = AIR_DENSITY * VOLUME  # kg
EFFECTIVE_MASS = MASS_AIR + MASS_ROOF * 0.5 + MASS_WALL * 0.3  # Include 50% roof, 30% wall mass
EFFECTIVE_SPECIFIC_HEAT = (
    MASS_AIR * AIR_SPECIFIC_HEAT + MASS_ROOF * SPECIFIC_HEAT_METAL * 0.5 + MASS_WALL * SPECIFIC_HEAT_WALL * 0.3) / EFFECTIVE_MASS
STEFAN_BOLTZMANN = 5.67e-8  # W/m²·K⁴
EMISSIVITY = 0.7
VENTILATION_RATE = 0.1

# Absorptivity values for different roof colors
ROOF_ABSORPTIVITY = {
    'white': 0.20,
    'gray': 0.53,
    'black': 0.94,
    'red': 0.55,
    'orange': 0.60,
    'green': 0.65,
    'blue': 0.72,
    'beige': 0.30,
    'brown': 0.75,
    'unpainted': 0.45
}

# Coordinates for Metro Manila cities
CITY_COORDINATES = {
    "Manila": (14.5995, 120.9842),
    "Caloocan": (14.6514, 120.9721),
    "Las Piñas": (14.4500, 120.9822),
    "Makati": (14.5547, 121.0244),
    "Malabon": (14.6680, 120.9567),
    "Mandaluyong": (14.5794, 121.0359),
    "Marikina": (14.6507, 121.1029),
    "Muntinlupa": (14.3810, 121.0490),
    "Navotas": (14.6667, 120.9417),
    "Parañaque": (14.4791, 121.0198),
    "Pasay": (14.5378, 121.0014),
    "Pasig": (14.5764, 121.0851),
    "Quezon City": (14.6760, 121.0437),
    "San Juan": (14.6042, 121.0300),
    "Taguig": (14.5176, 121.0509),
    "Valenzuela": (14.7011, 120.9830)
}

CITIES = list(CITY_COORDINATES.keys())

def get_weather_data(city, sim_date, hours=24):
    """Fetch hourly temperature and solar radiation for the specified date."""
    try:
        lat, lon = CITY_COORDINATES[city]
        current_date = datetime.now().date()
        sim_date = datetime.strptime(sim_date, '%Y-%m-%d').date()

        if sim_date >= current_date:
            # Future or current date: Use forecast API
            url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m,shortwave_radiation&timezone=Asia/Manila"
            )
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            temp = data['hourly']['temperature_2m'][:hours]
            solar = data['hourly']['shortwave_radiation'][:hours]
        else:
            # Past date: Use archive API
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                f"&start_date={sim_date}&end_date={sim_date}"
                f"&hourly=temperature_2m,shortwave_radiation&timezone=Asia/Manila"
            )
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            temp = data['hourly']['temperature_2m'][:hours]
            solar = data['hourly']['shortwave_radiation'][:hours]

        return temp, solar

    except requests.RequestException as e:
        print(f"Error fetching weather data for {city} on {sim_date}: {e}")
        if isinstance(e, requests.HTTPError) and e.response.status_code == 400:
            print("Likely cause: Invalid API request (e.g., future date for archive API or invalid parameters).")
        return None, None

def calculate_U_roof():
    """Calculate overall heat transfer coefficient (U-value) for composite roof."""
    R_metal = THICKNESS_ROOF / THERMAL_CONDUCTIVITY_METAL
    R_insulation = THICKNESS_INSULATION / THERMAL_CONDUCTIVITY_INSULATION
    R_conv_inside = 1 / H_COMBINED
    R_conv_outside = 1 / H_COMBINED
    R_total = R_metal + R_insulation + R_conv_inside + R_conv_outside
    U_roof = 1 / R_total
    return U_roof

def simulate_indoor_temp(city, color, sim_date, hours=24):
    """Simulate indoor temperature for a given city, roof color, and date."""
    if city not in CITIES:
        raise ValueError(f"City {city} not supported. Choose from {CITIES}")
    if color not in ROOF_ABSORPTIVITY:
        raise ValueError(f"Color {color} not supported. Choose from {list(ROOF_ABSORPTIVITY.keys())}")

    temps_outside, solar_radiation = get_weather_data(city, sim_date)
    if temps_outside is None or solar_radiation is None:
        return None, None

    alpha = ROOF_ABSORPTIVITY[color]
    temp_inside = [temps_outside[0]]
    U_roof = calculate_U_roof()

    for t in range(1, hours):
        T_out = temps_outside[t]
        T_in = temp_inside[-1]
        I_solar = solar_radiation[t]

        # Validate inputs
        if not (-50 <= T_out <= 100) or I_solar < 0 or np.isnan(T_out) or np.isnan(I_solar):
            print(f"Invalid data at hour {t}: T_out={T_out}, I_solar={I_solar}. Skipping.")
            temp_inside.append(T_in)
            continue

        # Heat transfer calculations
        Q_solar = I_solar * ROOF_AREA * alpha * 0.1
        Q_roof = U_roof * ROOF_AREA * (T_out - T_in)
        Q_window = H_WINDOW * WINDOW_AREA * (T_in - T_out)

        # Radiative heat loss
        T_roof_K = T_in + 273.15
        T_sky_K = T_out - 1 + 273.15 if (t >= 23 or t <= 3) else T_out + 273.15
        Q_radiation = EMISSIVITY * STEFAN_BOLTZMANN * ROOF_AREA * (T_roof_K**4 - T_sky_K**4)
        Q_radiation = np.clip(Q_radiation, -3000, 3000)

        # Ventilation (at night, hours 23-3)
        Q_ventilation = 0
        if t >= 23 or t <= 3 and T_in > T_out:
            Q_ventilation = VENTILATION_RATE * VOLUME * AIR_DENSITY * AIR_SPECIFIC_HEAT * (T_in - T_out) / 3600
            Q_ventilation = np.clip(Q_ventilation, -500, 500)

        # Net heat and temp change
        Q_net = Q_solar + Q_roof - Q_window - Q_radiation - Q_ventilation
        delta_T = (Q_net * 3600) / (EFFECTIVE_MASS * EFFECTIVE_SPECIFIC_HEAT)
        delta_T = np.clip(delta_T, -3, 3)
        T_new = T_in + delta_T
        T_new = np.clip(T_new, -50, 100)
        temp_inside.append(T_new)

    return temps_outside[:hours], temp_inside

def print_temperature_table(city, color, T_out, T_in, sim_date):
    """Print a table of outdoor and indoor temperatures for a single roof color and save as PNG."""
    # Print to console
    print(f"\nTemperature Data for {city} with {color.capitalize()} Roof on {sim_date}")
    print("-" * 50)
    print(f"{'Hour':<6} {'Outdoor Temp (°C)':<20} {'Indoor Temp (°C)':<20}")
    print("-" * 50)
    for h, (out, ins) in enumerate(zip(T_out, T_in)):
        print(f"{h:<6} {out:<20.2f} {ins:<20.2f}")
    print("-" * 50)

    # Create table data
    table_data = [[str(h), f"{out:.2f}", f"{ins:.2f}"] for h, (out, ins) in enumerate(zip(T_out, T_in))]
    columns = ['Hour', 'Outdoor Temp (°C)', 'Indoor Temp (°C)']

    # Create figure and table
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 3,
        colWidths=[0.15, 0.35, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Clear text
    table.scale(1.2, 1.8)
    plt.title(f"Temperature Data for {city} with {color.capitalize()} Roof on {sim_date}", pad=10, fontsize=14)
    plt.tight_layout(pad=0.5)

    # Save table as PNG
    plt.savefig("temperature_table.png", bbox_inches='tight', dpi=150, pad_inches=0.05)
    plt.close()

def print_all_colors_temperature_table(city, sim_date):
    """Print a table of outdoor temperature and indoor temperatures for all roof colors and save as PNG."""
    T_out, _ = simulate_indoor_temp(city, 'white', sim_date)  # Get outdoor temp
    if T_out is None:
        print("Cannot create table due to missing weather data.")
        return

    # Collect indoor temperatures for all colors in the same order as plot_all_colors
    all_temps = []
    color_order = ['white', 'gray', 'black', 'red', 'orange', 'green', 'blue', 'beige', 'brown', 'unpainted']
    for color in color_order:
        _, T_in = simulate_indoor_temp(city, color, sim_date)
        if T_in is None:
            print(f"Skipping {color} due to missing data.")
            continue
        all_temps.append(T_in)

    if not all_temps:
        print("No valid temperature data available for table.")
        return

    print(f"\nTemperature Data for All Roof Colors in {city} on {sim_date}")
    header = f"{'Hour':<6} {'Outdoor (°C)':<12}"
    for color in color_order:
        header += f"{color.capitalize():<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for h in range(24):
        row = f"{h:<6} {T_out[h]:<12.2f}"
        for T_in in all_temps:
            row += f"{T_in[h]:<12.2f}"
        print(row)
    print("-" * len(header))

    # Create table data
    table_data = []
    for h in range(24):
        row = [str(h), f"{T_out[h]:.2f}"]
        for T_in in all_temps:
            row.append(f"{T_in[h]:.2f}")
        table_data.append(row)
    columns = ['Hour', 'Outdoor (°C)'] + [color.capitalize() for color in color_order]

    # Create figure and table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')  # Hide axes
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(columns),
        colWidths=[0.08] + [0.085] * (len(columns) - 1)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    plt.title(f"Temperature Data for All Roof Colors in {city} on {sim_date}", pad=10, fontsize=14)
    plt.tight_layout(pad=0.5)

    # Save table as PNG
    plt.savefig("all_colors_temperature_table.png", bbox_inches='tight', dpi=150, pad_inches=0.05)
    plt.close()

def create_gif(frame_files, output_gif, duration=500):
    """Combine PNG frames into a GIF."""
    images = [Image.open(f) for f in frame_files]
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def plot_simulation(city, color, sim_date):
    """Plot indoor vs outdoor temperatures with interactivity and create an animated GIF."""
    T_out, T_in = simulate_indoor_temp(city, color, sim_date)
    if T_out is None or T_in is None:
        print("Cannot plot due to missing weather data.")
        return

    print_temperature_table(city, color, T_out, T_in, sim_date)

    color_map = {
        'white': '#F5F5DC',
        'gray': '#696969',
        'black': '#333333',
        'red': 'red',
        'orange': 'orange',
        'green': 'green',
        'blue': 'blue',
        'beige': 'tan',
        'brown': 'brown',
        'unpainted': '#4682B4'
    }

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_files = []

        # Generate frames for each hour
        for h in range(24):
            plt.figure(figsize=(10, 6))
            hours = np.arange(h + 1)

            # Plot up to current hour
            plt.plot(
                hours, T_out[:h + 1], label="Outdoor Temp (°C)",
                linestyle='--', color='#4B0082', marker='o', markersize=5
            )
            plt.plot(
                hours, T_in[:h + 1], label=f"Indoor Temp - {color.capitalize()} Roof (°C)",
                linestyle='-', color=color_map[color], linewidth=2, marker='o', markersize=5
            )

            plt.title(f"Indoor Temperature Simulation for {city} ({color.capitalize()} Roof) on {sim_date}")
            plt.xlabel("Hour")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            plt.grid(True)
            plt.xticks(np.arange(0, 24, 1), [str(i) for i in range(24)], rotation=45)
            plt.xlim(0, 23)
            plt.ylim(min(min(T_out), min(T_in)) - 1, max(max(T_out), max(T_in)) + 1)
            plt.tight_layout()

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{h:03d}.png")
            plt.savefig(frame_path)
            frame_files.append(frame_path)
            plt.close()

        # Create GIF
        create_gif(frame_files, "simulation_plot.gif", duration=500)

    # Generate static plot with interactivity
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = np.arange(24)
    line_out, = ax.plot(
        hours, T_out, label="Outdoor Temp (°C)",
        linestyle='--', color='#4B0082', marker='o', markersize=5
    )
    line_in, = ax.plot(
        hours, T_in, label=f"Indoor Temp - {color.capitalize()} Roof (°C)",
        linestyle='-', color=color_map[color], linewidth=2, marker='o', markersize=5
    )
    ax.set_title(f"Indoor Temperature Simulation for {city} ({color.capitalize()} Roof) on {sim_date}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(hours)
    ax.set_xticklabels([str(h) for h in hours], rotation=45)
    ax.set_xlim(0, 23)
    plt.tight_layout()

    # Add interactivity
    cursor = mplcursors.cursor([line_out, line_in], hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = int(round(sel.target[0]))
        if 0 <= index < 24:
            sel.annotation.set_text(
                f"Hour: {index}\nOutdoor: {T_out[index]:.2f} °C\nIndoor: {T_in[index]:.2f} °C"
            )

    plt.savefig("simulation_plot.png")
    plt.show(block=True)

def visualize_thermal(city, color, sim_date):
    """Visualize indoor and outdoor temperatures as a heatmap with interactivity and create an animated GIF."""
    T_out, T_in = simulate_indoor_temp(city, color, sim_date)
    if T_out is None or T_in is None:
        print("Cannot visualize due to missing weather data.")
        return

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_files = []

        # Generate frames for each hour
        for h in range(24):
            plt.figure(figsize=(12, 3))
            # Show temperatures from Hour 0 to current hour
            data = np.array([T_in[:h+1], T_out[:h+1]])
            plt.imshow(
                data, aspect='auto', cmap='inferno',
                extent=[0, h+1, -0.5, 1.5], vmin=min(min(T_out), min(T_in)), vmax=max(max(T_out), max(T_in))
            )
            plt.hlines(0.5, 0, h+1, colors='white', linewidth=1)
            plt.colorbar(label='Temperature (°C)')
            plt.title(f"Thermal Visualization for {city} ({color.capitalize()} Roof) on {sim_date}")
            plt.xlabel("Hour")
            plt.yticks([0, 1], ['Outdoor', 'Indoor'])
            plt.xticks(np.arange(0, 25, 1))
            plt.xlim(0, 24)
            plt.ylim(-0.5, 1.5)
            plt.tight_layout()

            # Save frame
            frame_path = os.path.join(temp_dir, f"heatmap_frame_{h:03d}.png")
            plt.savefig(frame_path)
            frame_files.append(frame_path)
            plt.close()

        # Add final frame with full heatmap
        plt.figure(figsize=(12, 3))
        data = np.array([T_in, T_out])
        plt.imshow(
            data, aspect='auto', cmap='inferno',
            extent=[0, 24, -0.5, 1.5], vmin=min(min(T_out), min(T_in)), vmax=max(max(T_out), max(T_in))
        )
        plt.hlines(0.5, 0, 24, colors='white', linewidth=1)
        plt.colorbar(label='Temperature (°C)')
        plt.title(f"Thermal Visualization for {city} ({color.capitalize()} Roof) on {sim_date}")
        plt.xlabel("Hour")
        plt.yticks([0, 1], ['Outdoor', 'Indoor'])
        plt.xticks(np.arange(0, 25, 1))
        plt.xlim(0, 24)
        plt.ylim(-0.5, 1.5)
        plt.tight_layout()

        # Save final frame
        final_frame_path = os.path.join(temp_dir, "heatmap_frame_final.png")
        plt.savefig(final_frame_path)
        frame_files.append(final_frame_path)
        plt.close()

        # Create GIF
        create_gif(frame_files, "thermal_heatmap.gif", duration=500)

    # Generate static heatmap with interactivity
    fig, ax = plt.subplots(figsize=(12, 3))
    data = np.array([T_in, T_out])  # Indoor first, then Outdoor
    im = ax.imshow(
        data, aspect='auto', cmap='inferno', extent=[0, 24, -0.5, 1.5],
        vmin=min(min(T_out), min(T_in)), vmax=max(max(T_out), max(T_in))
    )
    ax.hlines(0.5, 0, 24, colors='white', linewidth=1)
    plt.colorbar(im, label='Temperature (°C)')
    ax.set_title(f"Thermal Visualization for {city} ({color.capitalize()} Roof) on {sim_date}")
    ax.set_xlabel("Hour")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Outdoor', 'Indoor'])
    ax.set_xticks(np.arange(0, 25, 1))
    ax.set_ylim(-0.5, 1.5)
    plt.tight_layout()

    # Add interactivity with swapped cursor data interpretation
    cursor = mplcursors.cursor(im, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        hour = int(round(x))
        row_idx = int(round(y))
        if 0 <= hour < 24 and 0 <= row_idx < 2:
            label = 'Outdoor' if row_idx == 0 else 'Indoor'
            temp = data[1, hour] if row_idx == 0 else data[0, hour]
            sel.annotation.set_text(
                f"Hour: {hour}\n{label}: {temp:.2f} °C"
            )

    plt.savefig("thermal_heatmap.png")
    plt.show(block=True)

def plot_all_colors(city, sim_date):
    """Plot indoor temperatures for all roof colors and outdoor temperature."""
    T_out, _ = simulate_indoor_temp(city, 'white', sim_date)  # Get outdoor temp
    if T_out is None:
        print("Cannot plot due to missing weather data.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    hours = np.arange(24)

    # Plot outdoor temperature
    ax.plot(
        hours, T_out, label="Outdoor Temp (°C)",
        linestyle='--', color='#4B0082', linewidth=2, marker='o', markersize=5
    )

    # Define colors for plotting
    color_map = {
        'white': '#F5F5DC',
        'gray': '#696969',
        'black': '#333333',
        'red': 'red',
        'orange': 'orange',
        'green': 'green',
        'blue': 'blue',
        'beige': 'tan',
        'brown': 'brown',
        'unpainted': '#4682B4'
    }

    # Simulate and plot for each roof color
    lines = []
    color_order = ['white', 'gray', 'black', 'red', 'orange', 'green', 'blue', 'beige', 'brown', 'unpainted']
    for color in color_order:
        _, T_in = simulate_indoor_temp(city, color, sim_date)
        if T_in is None:
            continue
        line, = ax.plot(
            hours, T_in, label=f"Indoor - {color.capitalize()} Roof",
            linestyle='-', color=color_map[color], linewidth=1.5, marker='o', markersize=4, alpha=0.8
        )
        lines.append(line)

    ax.set_title(f"Indoor Temperature Simulation for All Roof Colors in {city} on {sim_date}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True)
    ax.set_xticks(hours)
    ax.set_xticklabels([str(h) for h in hours], rotation=45)
    ax.set_xlim(0, 23)
    min_temp = min(min(T_out), min(min(simulate_indoor_temp(city, c, sim_date)[1] or [float('inf')]) for c in ROOF_ABSORPTIVITY.keys()))
    max_temp = max(max(T_out), max(max(simulate_indoor_temp(city, c, sim_date)[1] or [float('-inf')]) for c in ROOF_ABSORPTIVITY.keys()))
    ax.set_ylim(min_temp - 1, max_temp + 1)
    plt.tight_layout()

    # Add interactivity
    cursor = mplcursors.cursor(lines + [ax.get_lines()[0]], hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = int(round(sel.target[0]))
        if 0 <= index < 24:
            label = sel.artist.get_label()
            temp = sel.target[1]
            sel.annotation.set_text(
                f"Hour: {index}\n{label}: {temp:.2f} °C"
            )

    plt.savefig("all_colors_simulation.png", bbox_inches='tight')
    plt.show(block=True)

def get_user_input():
    """Get validated user input for city, color, simulation date, and plot preferences."""
    # City input
    print("Available cities:", ", ".join(CITIES))
    while True:
        city = input("Enter city: ").strip().title()
        if city in CITIES:
            break
        print(f"Invalid city. Choose from: {', '.join(CITIES)}")

    # Color input
    print("Available roof colors:", ", ".join(ROOF_ABSORPTIVITY.keys()))
    while True:
        color = input("Enter roof color: ").strip().lower()
        if color in ROOF_ABSORPTIVITY:
            break
        print(f"Invalid color. Choose from: {', '.join(ROOF_ABSORPTIVITY.keys())}")

    # Date input
    min_date = datetime(2020, 1, 1).date()
    max_date = (datetime.now() + timedelta(days=7)).date()
    print(f"Enter simulation date (YYYY-MM-DD) between {min_date} and {max_date}:")
    while True:
        date_input = input("Enter date: ").strip()
        try:
            sim_date = datetime.strptime(date_input, '%Y-%m-%d').date()
            if min_date <= sim_date <= max_date:
                break
            print(f"Date must be between {min_date} and {max_date}.")
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD (e.g., 2025-04-17).")

    # Combined plot input
    while True:
        combined_plot = input("Do you want a graph comparing all roof colors with outdoor temperature? (yes/no): ").strip().lower()
        if combined_plot in ['yes', 'no']:
            break
        print("Please enter 'yes' or 'no'.")

    return city, color, sim_date.strftime('%Y-%m-%d'), combined_plot == 'yes'

if __name__ == "__main__":
    city, color, sim_date, want_combined_plot = get_user_input()
    plot_simulation(city, color, sim_date)
    visualize_thermal(city, color, sim_date)
    if want_combined_plot:
        print_all_colors_temperature_table(city, sim_date)
        plot_all_colors(city, sim_date)