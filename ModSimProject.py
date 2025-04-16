import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Constants
FLOOR_AREA = 124.87  # m²
ROOF_AREA = 137.94  # m²
THICKNESS_ROOF = 0.001  # m (metal sheet)
THICKNESS_INSULATION = 0.05  # m (fiberglass)
DENSITY_METAL = 7850  # kg/m³
DENSITY_INSULATION = 50  # kg/m³
SPECIFIC_HEAT_METAL = 500  # J/kg·K
SPECIFIC_HEAT_INSULATION = 840  # J/kg·K
THERMAL_CONDUCTIVITY_METAL = 50  # W/m·K
THERMAL_CONDUCTIVITY_INSULATION = 0.04  # W/m·K
H_COMBINED = 10  # W/m²·K
WALL_AREA = 131.61  # m²
WINDOW_AREA = 3.716  # m²
H_WALL = 6  # W/m²·K
H_WINDOW = 15  # W/m²·K
AIR_DENSITY = 1.2  # kg/m³
AIR_SPECIFIC_HEAT = 1005  # J/kg·K
VOLUME = 2.5 * FLOOR_AREA  # m³
MASS_AIR = AIR_DENSITY * VOLUME  # kg
STEFAN_BOLTZMANN = 5.67e-8  # W/m²·K⁴
EMISSIVITY = 0.9
INTERNAL_HEAT = 0  # W
SHADING_FACTOR = 0.3
VENT_RATE = 0.05  # m³/s

# Absorptivity values
ROOF_ABSORPTIVITY = {
    'white': 0.30,
    'gray': 0.63,
    'black': 0.94,
    'red': 0.60,
    'orange': 0.66,
    'green': 0.75,
    'blue': 0.70,
    'beige': 0.20,
    'brown': 0.66,
    'unpainted': 0.75
}

# Metro Manila cities
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
ROOF_TYPES = ['composite', 'bare']

def get_weather_data(city):
    """Fetch hourly temperature and solar radiation."""
    try:
        lat, lon = CITY_COORDINATES[city]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,shortwave_radiation&timezone=Asia/Manila"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['hourly']['temperature_2m'], data['hourly']['shortwave_radiation']
    except requests.RequestException as e:
        print(f"Error fetching weather data for {city}: {e}")
        return None, None

def calculate_U_roof():
    """Calculate U-value for composite roof."""
    R_metal = THICKNESS_ROOF / THERMAL_CONDUCTIVITY_METAL
    R_insulation = THICKNESS_INSULATION / THERMAL_CONDUCTIVITY_INSULATION
    R_total = R_metal + R_insulation
    return 1 / R_total

def simulate_indoor_temp(city, color, roof_type, hours=24):
    """Simulate indoor temperature with selected roof type."""
    if city not in CITIES:
        raise ValueError(f"City {city} not supported. Choose from {CITIES}")
    if color not in ROOF_ABSORPTIVITY:
        raise ValueError(f"Color {color} not supported. Choose from {list(ROOF_ABSORPTIVITY.keys())}")
    if roof_type not in ROOF_TYPES:
        raise ValueError(f"Roof type {roof_type} not supported. Choose from {ROOF_TYPES}")

    temps_outside, solar_radiation = get_weather_data(city)
    if temps_outside is None or solar_radiation is None:
        return None, None

    # Set roof properties
    if roof_type == 'composite':
        mass_roof = (ROOF_AREA * THICKNESS_ROOF * DENSITY_METAL) + (ROOF_AREA * THICKNESS_INSULATION * DENSITY_INSULATION)
        U_roof = calculate_U_roof()
    else:  # bare
        mass_roof = ROOF_AREA * THICKNESS_ROOF * DENSITY_METAL
        U_roof = H_COMBINED

    alpha = ROOF_ABSORPTIVITY[color]
    temp_inside = [temps_outside[0]]  # Start equal to outdoor
    for t in range(1, hours):
        T_out = temps_outside[t]
        T_in = temp_inside[-1]
        I_solar = min(solar_radiation[t], 900)  # Cap solar radiation

        # Validate inputs
        if not (10 <= T_out <= 40) or I_solar < 0 or np.isnan(T_out) or np.isnan(I_solar):
            print(f"Invalid data at hour {t}: T_out={T_out}, I_solar={I_solar}. Skipping.")
            temp_inside.append(T_in)
            continue

        # Roof temperature and heat transfer
        Q_solar = I_solar * ROOF_AREA * alpha * SHADING_FACTOR
        if roof_type == 'bare':
            T_roof = T_out + (Q_solar / (H_COMBINED * ROOF_AREA)) * 2.5
            Q_roof = H_COMBINED * ROOF_AREA * (T_roof - T_in)
            Q_solar = Q_solar * 0.2
        else:  # composite
            Q_roof = U_roof * ROOF_AREA * (T_out - T_in)
            Q_solar *= 0.7  # Reduced to prevent overheating

        # Other heat transfers
        Q_wall = H_WALL * WALL_AREA * (T_out - T_in)
        Q_window = H_WINDOW * WINDOW_AREA * (T_in - T_out)
        Q_vent = VENT_RATE * AIR_DENSITY * AIR_SPECIFIC_HEAT * (T_out - T_in)

        # Radiative heat loss
        Q_radiation = 0
        if I_solar < 10:
            T_roof_K = T_in + 273.15
            T_sky_K = T_out - 5 + 273.15
            if not (200 <= T_roof_K <= 400) or not (200 <= T_sky_K <= 400):
                print(f"Warning at hour {t}: T_roof_K={T_roof_K}, T_sky_K={T_sky_K}. Skipping radiation.")
            else:
                Q_radiation = EMISSIVITY * STEFAN_BOLTZMANN * ROOF_AREA * (T_roof_K**4 - T_sky_K**4)

        # Net heat and temp change
        Q_net = Q_solar + Q_roof + Q_wall - Q_window - Q_vent - Q_radiation + INTERNAL_HEAT
        delta_T = (Q_net * 3600) / (MASS_AIR * AIR_SPECIFIC_HEAT)
        delta_T = np.clip(delta_T, -5, 5)
        T_new = T_in + delta_T
        T_new = np.clip(T_new, T_out - 1 if roof_type == 'composite' else T_out, 42)
        temp_inside.append(T_new)

    return temps_outside[:hours], temp_inside

def print_temperature_table(city, color, roof_type, T_out, T_in):
    """Print temperature table."""
    print(f"\nTemperature Data for {city} with {color.capitalize()} {roof_type.capitalize()} Roof")
    print("-" * 50)
    print(f"{'Hour':<6} {'Outdoor Temp (°C)':<20} {'Indoor Temp (°C)':<20}")
    print("-" * 50)
    for h, (out, ins) in enumerate(zip(T_out, T_in)):
        print(f"{h:<6} {out:<20.2f} {ins:<20.2f}")
    print("-" * 50)

def plot_simulation(city, color, roof_type):
    """Plot indoor vs outdoor temperatures."""
    T_out, T_in = simulate_indoor_temp(city, color, roof_type)
    if T_out is None or T_in is None:
        print("Cannot plot due to missing weather data.")
        return

    print_temperature_table(city, color, roof_type, T_out, T_in)

    plt.figure(figsize=(10, 6))
    hours = np.arange(24)
    plt.plot(hours, T_out, label="Outdoor Temp (°C)", linestyle='--')
    plt.plot(hours, T_in, label=f"Indoor Temp - {color.capitalize()} {roof_type.capitalize()} Roof (°C)", linewidth=2)
    plt.title(f"Indoor Temperature Simulation for {city}")
    plt.xlabel("Hour")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_thermal(city, color, roof_type):
    """Visualize indoor temperature as heatmap."""
    _, T_in = simulate_indoor_temp(city, color, roof_type)
    if T_in is None:
        print("Cannot visualize due to missing weather data.")
        return

    plt.figure(figsize=(12, 2))
    plt.imshow([T_in], aspect='auto', cmap='inferno', extent=[0, 24, 0, 1])
    plt.colorbar(label='Indoor Temp (°C)')
    plt.title(f"Thermal Visualization of Indoor Temp for {city} ({color.capitalize()} {roof_type.capitalize()} Roof)")
    plt.xlabel("Hour")
    plt.yticks([])
    plt.xticks(np.arange(0, 25, 1))
    plt.tight_layout()
    plt.show()

def get_user_input():
    """Get validated user input."""
    print("Available cities:", ", ".join(CITIES))
    while True:
        city = input("Enter city: ").strip().title()
        if city in CITIES:
            break
        print(f"Invalid city. Choose from: {', '.join(CITIES)}")

    print("Available roof colors:", ", ".join(ROOF_ABSORPTIVITY.keys()))
    while True:
        color = input("Enter roof color: ").strip().lower()
        if color in ROOF_ABSORPTIVITY:
            break
        print(f"Invalid color. Choose from: {', '.join(ROOF_ABSORPTIVITY.keys())}")

    print("Available roof types: composite, bare")
    while True:
        roof_type = input("Enter roof type (composite or bare): ").strip().lower()
        if roof_type in ROOF_TYPES:
            break
        print(f"Invalid roof type. Choose from: {', '.join(ROOF_TYPES)}")

    return city, color, roof_type

if __name__ == "__main__":
    city, color, roof_type = get_user_input()
    plot_simulation(city, color, roof_type)
    visualize_thermal(city, color, roof_type)