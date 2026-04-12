"""
PM Deposition Modelling in Human Respiratory System
====================================================
Monte Carlo simulation of particulate matter (PM 0.1 - PM 10) deposition
in the human lung, based on the stochastic IDEAL-2 model (Koblinger & Hofmann).

Reference paper: "A Study of Environmental Impact of Air Pollution on Human Health:
PM Deposition Modelling" - Mayilvahanan I, Naresh K & Gandhimathi A (KCT, Coimbatore)

Author of this implementation: Claude (reconstructed from paper methodology)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# ============================================================================
# CONSTANTS
# ============================================================================

BOLTZMANN_K = 1.380649e-23       # J/K
TEMPERATURE = 310.15             # K (37°C body temperature)
AIR_VISCOSITY = 1.81e-5          # Pa·s (dynamic viscosity of air at 37°C)
AIR_DENSITY = 1.125              # kg/m³ (air density at 37°C)
PARTICLE_DENSITY = 1000.0        # kg/m³ (unit density as per paper: 1.0 g/cm³)
GRAVITY = 9.81                   # m/s²
MEAN_FREE_PATH = 0.066e-6        # m (mean free path of air molecules)

# PM sizes in micrometers
PM_SIZES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 10.0]

# Number of particles per simulation
N_PARTICLES = 10000

# Age group categories from the paper
AGE_GROUPS = {
    'D (21-30 yrs)': {'tidal_volume': 2800e-6, 'breathing_freq': 15, 'label': '21-30'},
    'C (31-35 yrs)': {'tidal_volume': 2600e-6, 'breathing_freq': 14, 'label': '31-35'},
    'B (36-40 yrs)': {'tidal_volume': 2400e-6, 'breathing_freq': 13, 'label': '36-40'},
    'A (41-50 yrs)': {'tidal_volume': 2200e-6, 'breathing_freq': 12, 'label': '41-50'},
}

# Lung regions mapped to airway generations
REGION_MAP = {
    'Extra-thoracic': (0, 3),       # Nose, pharynx, larynx
    'Conducting (Tubular)': (3, 16), # Trachea through terminal bronchioles
    'Alveolar': (16, 23),            # Respiratory bronchioles through alveolar sacs
}


# ============================================================================
# WEIBEL SYMMETRIC LUNG MODEL (Model A, 23 generations)
# ============================================================================

@dataclass
class AirwayGeneration:
    """Properties of a single airway generation."""
    generation: int
    length: float        # meters
    diameter: float      # meters
    n_airways: int       # number of parallel airways
    branching_angle: float  # radians

def build_weibel_lung() -> List[AirwayGeneration]:
    """
    Build Weibel Type A symmetric lung model.
    Data from Weibel (1963) "Morphometry of the Human Lung".
    """
    # Weibel model data: (generation, length_mm, diameter_mm, branching_angle_deg)
    weibel_data = [
        (0,  120.0,  18.0,   0),    # Trachea
        (1,   47.6,  12.2,  35),    # Main bronchi
        (2,   19.0,   8.3,  30),    # Lobar bronchi
        (3,    7.6,   5.6,  25),    # Segmental bronchi
        (4,   12.7,   4.5,  25),
        (5,   10.7,   3.5,  22),
        (6,    9.0,   2.8,  20),
        (7,    7.6,   2.3,  18),
        (8,    6.4,   1.86, 18),
        (9,    5.4,   1.54, 16),
        (10,   4.6,   1.30, 15),
        (11,   3.9,   1.09, 14),
        (12,   3.3,   0.95, 13),
        (13,   2.7,   0.82, 12),
        (14,   2.3,   0.74, 11),
        (15,   2.0,   0.66, 10),
        (16,   1.65,  0.56, 10),    # Terminal bronchioles
        (17,   1.41,  0.45,  9),    # Respiratory bronchioles
        (18,   1.17,  0.39,  8),
        (19,   0.99,  0.33,  7),
        (20,   0.83,  0.27,  7),    # Alveolar ducts
        (21,   0.70,  0.23,  6),
        (22,   0.59,  0.20,  5),
        (23,   0.50,  0.18,  5),    # Alveolar sacs
    ]

    airways = []
    for gen, length_mm, diam_mm, angle_deg in weibel_data:
        airways.append(AirwayGeneration(
            generation=gen,
            length=length_mm * 1e-3,           # mm -> m
            diameter=diam_mm * 1e-3,            # mm -> m
            n_airways=2**gen,
            branching_angle=np.radians(angle_deg)
        ))
    return airways


# ============================================================================
# PARTICLE PHYSICS
# ============================================================================

def cunningham_slip(dp: float) -> float:
    """
    Cunningham slip correction factor for small particles.
    dp: particle diameter in meters
    """
    Kn = 2 * MEAN_FREE_PATH / dp  # Knudsen number
    return 1 + Kn * (2.34 + 1.05 * np.exp(-0.39 / Kn))


def diffusion_coefficient(dp: float) -> float:
    """
    Stokes-Einstein diffusion coefficient.
    dp: particle diameter in meters
    Returns: D in m²/s
    """
    Cc = cunningham_slip(dp)
    return (BOLTZMANN_K * TEMPERATURE * Cc) / (3 * np.pi * AIR_VISCOSITY * dp)


def settling_velocity(dp: float) -> float:
    """
    Terminal settling velocity under gravity (Stokes regime).
    dp: particle diameter in meters
    Returns: v_s in m/s
    """
    Cc = cunningham_slip(dp)
    return (PARTICLE_DENSITY * dp**2 * GRAVITY * Cc) / (18 * AIR_VISCOSITY)


def flow_velocity(airway: AirwayGeneration, tidal_volume: float, breathing_freq: float) -> float:
    """
    Mean air flow velocity in an airway generation.
    """
    # Volumetric flow rate (m³/s) — inspiratory flow
    Q = tidal_volume * breathing_freq / 60.0  # approximate
    # Cross-sectional area of all parallel airways
    A_total = airway.n_airways * np.pi * (airway.diameter / 2)**2
    if A_total == 0:
        return 0
    return Q / A_total


# ============================================================================
# DEPOSITION MECHANISMS
# ============================================================================

def deposition_brownian(dp: float, airway: AirwayGeneration, v_flow: float) -> float:
    """
    Deposition efficiency by Brownian diffusion.
    Uses Ingham (1975) formula for laminar flow in a cylindrical tube.
    """
    D = diffusion_coefficient(dp)
    R = airway.diameter / 2

    if v_flow <= 0 or R <= 0:
        return 0.0

    # Dimensionless deposition parameter (Delta)
    residence_time = airway.length / max(v_flow, 1e-10)
    delta = D * residence_time / (R**2)

    if delta > 0.1:
        # High diffusion regime
        eff = 1 - 0.819 * np.exp(-14.63 * delta) - 0.0976 * np.exp(-89.22 * delta) \
              - 0.0325 * np.exp(-228.0 * delta)
    else:
        # Low diffusion regime (Gormley & Kennedy)
        mu = (8 * delta) ** (1/3)
        eff = 1 - 0.819 * np.exp(-3.657 * mu) - 0.0976 * np.exp(-22.3 * mu)

    return np.clip(eff, 0, 1)


def deposition_sedimentation(dp: float, airway: AirwayGeneration, v_flow: float) -> float:
    """
    Deposition efficiency by gravitational sedimentation.
    Uses Pich (1972) formula for horizontal tubes.
    """
    v_s = settling_velocity(dp)
    R = airway.diameter / 2

    if v_flow <= 0 or R <= 0:
        return 0.0

    # Dimensionless settling parameter
    epsilon = (3 * v_s * airway.length) / (8 * R * v_flow)

    # Account for airway inclination (average gravity component)
    gravity_factor = np.cos(airway.branching_angle) if airway.branching_angle > 0 else 1.0
    epsilon *= gravity_factor

    if epsilon >= 1:
        return 1.0

    # Pich formula
    eff = (2 / np.pi) * (2 * epsilon * np.sqrt(1 - epsilon**(2/3))
                          - epsilon**(1/3) * np.sqrt(1 - epsilon**(2/3))
                          + np.arcsin(epsilon**(1/3)))

    return np.clip(eff, 0, 1)


def deposition_impaction(dp: float, airway: AirwayGeneration, v_flow: float) -> float:
    """
    Deposition efficiency by inertial impaction at bifurcations.
    Uses the Yeh & Schum (1980) empirical model.
    """
    Cc = cunningham_slip(dp)

    # Stokes number
    tau = (PARTICLE_DENSITY * dp**2 * Cc) / (18 * AIR_VISCOSITY)
    Stk = tau * v_flow / (airway.diameter / 2)

    if airway.branching_angle == 0:
        return 0.0  # No impaction at trachea (no bifurcation)

    # Empirical impaction formula (Zhang et al.)
    theta = airway.branching_angle
    eff = 1 - (2 / np.pi) * np.arccos(Stk * np.sin(theta))

    # Stk must be sufficient for impaction
    if Stk * np.sin(theta) > 1:
        eff = 1.0
    elif Stk * np.sin(theta) < 0:
        eff = 0.0

    return np.clip(eff, 0, 1)


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def simulate_single_particle(
    dp_um: float,
    airways: List[AirwayGeneration],
    tidal_volume: float,
    breathing_freq: float,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Simulate the journey of a single particle through the lung.

    Uses statistical weight method: instead of binary deposition,
    the particle's weight is reduced at each generation by the
    deposition probability.

    Returns: dict with regional deposition fractions
    """
    dp = dp_um * 1e-6  # convert µm to m
    weight = 1.0
    regional_deposition = {region: 0.0 for region in REGION_MAP}

    for airway in airways:
        if weight < 1e-10:
            break

        gen = airway.generation
        v_flow = flow_velocity(airway, tidal_volume, breathing_freq)

        # Add stochastic variation to airway geometry (±15% as in IDEAL-2)
        length_var = airway.length * (1 + 0.15 * rng.standard_normal())
        diam_var = airway.diameter * (1 + 0.10 * rng.standard_normal())

        # Create a varied airway for this particle
        varied_airway = AirwayGeneration(
            generation=gen,
            length=max(length_var, airway.length * 0.5),
            diameter=max(diam_var, airway.diameter * 0.5),
            n_airways=airway.n_airways,
            branching_angle=airway.branching_angle
        )

        # Calculate deposition by each mechanism
        p_diff = deposition_brownian(dp, varied_airway, v_flow)
        p_sed = deposition_sedimentation(dp, varied_airway, v_flow)
        p_imp = deposition_impaction(dp, varied_airway, v_flow)

        # Combined deposition probability (independent mechanisms)
        p_total = 1 - (1 - p_diff) * (1 - p_sed) * (1 - p_imp)
        p_total = np.clip(p_total, 0, 1)

        # Determine which region this generation belongs to
        for region, (gen_start, gen_end) in REGION_MAP.items():
            if gen_start <= gen < gen_end:
                deposited = weight * p_total
                regional_deposition[region] += deposited
                break

        # Reduce statistical weight
        weight *= (1 - p_total)

    # Remaining weight = exhaled (not deposited)
    return regional_deposition


def run_simulation(
    n_particles: int = N_PARTICLES,
    age_group: str = 'D (21-30 yrs)',
    pm_sizes: List[float] = PM_SIZES,
    seed: int = 42
) -> Dict[float, Dict[str, Tuple[float, float]]]:
    """
    Run full Monte Carlo simulation for all PM sizes.

    Returns: {pm_size: {region: (mean_deposition%, std%)}}
    """
    rng = np.random.default_rng(seed)
    airways = build_weibel_lung()
    tv = AGE_GROUPS[age_group]['tidal_volume']
    bf = AGE_GROUPS[age_group]['breathing_freq']

    results = {}

    for pm in pm_sizes:
        region_deposits = {region: [] for region in REGION_MAP}

        for i in range(n_particles):
            dep = simulate_single_particle(pm, airways, tv, bf, rng)
            for region in REGION_MAP:
                region_deposits[region].append(dep[region] * 100)  # as percentage

        results[pm] = {}
        for region in REGION_MAP:
            vals = np.array(region_deposits[region])
            results[pm][region] = (float(np.mean(vals)), float(np.std(vals)))

        total_dep = sum(results[pm][r][0] for r in REGION_MAP)
        print(f"  PM {pm:>4.1f} µm → Total deposition: {total_dep:5.1f}% "
              f"(ET: {results[pm]['Extra-thoracic'][0]:5.1f}%, "
              f"Tubular: {results[pm]['Conducting (Tubular)'][0]:5.1f}%, "
              f"Alveolar: {results[pm]['Alveolar'][0]:5.1f}%)")

    return results


# ============================================================================
# MONTE CARLO FORMULA (from paper)
# ============================================================================

def monte_carlo_formula(n_particles: int, unit_density: float,
                        tidal_volume_cm3: float, breathing_freq: float) -> float:
    """
    Paper's formula:
    MC = (30 × N × ρ) / (TV × BF)
    For 8-hour average per day.
    """
    return (30 * n_particles * unit_density) / (tidal_volume_cm3 * breathing_freq)


# ============================================================================
# 3D VISUALIZATION (Fig 3.2 from paper)
# ============================================================================

def generate_3d_scatter(results: Dict, save_path: str = 'pm_3d_scatter.png'):
    """
    Generate 3D scatter plot of PM distribution in alveolar system.
    Reproduces Fig 3.2 from the paper.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        0.1:  '#8B00FF',  # violet
        0.5:  '#0000FF',  # blue
        1.0:  '#00BFFF',  # cyan
        1.5:  '#00FF00',  # green
        2.0:  '#FFFF00',  # yellow
        2.5:  '#FF8C00',  # orange
        10.0: '#FF0000',  # red
    }

    labels = {
        0.1: 'PM 0.1 (Ultrafine)',
        0.5: 'PM 0.5',
        1.0: 'PM 1.0',
        1.5: 'PM 1.5',
        2.0: 'PM 2.0',
        2.5: 'PM 2.5 (Fine)',
        10.0: 'PM 10 (Coarse)',
    }

    rng = np.random.default_rng(123)

    for pm in PM_SIZES:
        # Number of scatter points proportional to alveolar deposition
        alv_dep = results[pm]['Alveolar'][0]
        n_points = max(int(alv_dep * 30), 20)

        # Generate lung-shaped distribution
        # Alveolar region is roughly spherical in the lower lung
        spread = alv_dep * 3 + 5
        theta = rng.uniform(0, 2 * np.pi, n_points)
        phi = rng.uniform(0, np.pi, n_points)
        r = rng.exponential(spread, n_points)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) - 10  # shift downward (lungs are below trachea)

        ax.scatter(x, y, z, c=colors[pm], s=8, alpha=0.6, label=labels[pm])

    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (depth, mm)', fontsize=10)
    ax.set_title('3D Particulate Spread in Alveolar System\n(Monte Carlo Simulation - Multiple Runs)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, markerscale=3)
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved 3D scatter plot: {save_path}")


# ============================================================================
# BAR CHART VISUALIZATION (Fig 4.1 from paper)
# ============================================================================

def generate_deposition_bar_charts(all_age_results: Dict, save_path: str = 'pm_deposition_bars.png'):
    """
    Generate grouped bar charts showing PM deposition % by region and age group.
    Reproduces Fig 4.1 from the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
    regions = list(REGION_MAP.keys())
    region_titles = ['Deposition in Extra-thoracic Region',
                     'Deposition in Conducting Airways',
                     'Deposition in Alveolar Region']

    bar_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bar_width = 0.18

    for idx, (region, title) in enumerate(zip(regions, region_titles)):
        ax = axes[idx]
        x = np.arange(len(PM_SIZES))

        for i, (age_key, age_data) in enumerate(all_age_results.items()):
            means = [age_data[pm][region][0] for pm in PM_SIZES]
            label = AGE_GROUPS[age_key]['label'] + ' yrs'
            ax.bar(x + i * bar_width, means, bar_width,
                   label=label, color=bar_colors[i], alpha=0.85, edgecolor='white')

        ax.set_xlabel('Particle Size (µm)', fontsize=11)
        ax.set_ylabel('Deposition (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x + bar_width * 1.5)
        ax.set_xticklabels([f'{pm}' for pm in PM_SIZES], fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Percent Occupancy of PM in Respiratory System by Age Group\n(Monte Carlo Simulation, N=10,000 particles)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved deposition bar charts: {save_path}")


# ============================================================================
# DEPOSITION HEATMAP
# ============================================================================

def generate_heatmap(results: Dict, age_label: str, save_path: str = 'pm_heatmap.png'):
    """
    Generate a heatmap of deposition % by PM size and lung region.
    """
    regions = list(REGION_MAP.keys())
    region_short = ['Extra-thoracic', 'Tubular', 'Alveolar']
    data = np.zeros((len(PM_SIZES), len(regions)))

    for i, pm in enumerate(PM_SIZES):
        for j, region in enumerate(regions):
            data[i, j] = results[pm][region][0]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(region_short, fontsize=11)
    ax.set_yticks(range(len(PM_SIZES)))
    ax.set_yticklabels([f'PM {pm}' for pm in PM_SIZES], fontsize=11)

    # Annotate cells
    for i in range(len(PM_SIZES)):
        for j in range(len(regions)):
            val = data[i, j]
            color = 'white' if val > data.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_title(f'PM Deposition (%) in Respiratory Regions — {age_label}',
                 fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Deposition %', shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved heatmap: {save_path}")


# ============================================================================
# RESULTS TABLE
# ============================================================================

def print_results_table(results: Dict, age_label: str):
    """Print formatted results table matching Table 4.1 from the paper."""
    print(f"\n{'='*70}")
    print(f"  PM Deposition Results — {age_label}")
    print(f"{'='*70}")
    print(f"  {'PM (µm)':<10} {'Extra-thoracic':>16} {'Tubular':>16} {'Alveolar':>16}")
    print(f"  {'-'*58}")

    for pm in PM_SIZES:
        et = results[pm]['Extra-thoracic']
        tb = results[pm]['Conducting (Tubular)']
        al = results[pm]['Alveolar']
        print(f"  PM {pm:<6.1f} {et[0]:>11.1f}% ±{et[1]:<4.1f} "
              f"{tb[0]:>11.1f}% ±{tb[1]:<4.1f} "
              f"{al[0]:>11.1f}% ±{al[1]:<4.1f}")

    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  PM DEPOSITION MONTE CARLO SIMULATION")
    print("  Based on Mayilvahanan et al. (KCT, Coimbatore)")
    print("  Stochastic lung model with Weibel geometry")
    print(f"  Particles per simulation: {N_PARTICLES:,}")
    print("=" * 70)

    all_age_results = {}

    for age_key in AGE_GROUPS:
        label = AGE_GROUPS[age_key]['label']
        print(f"\n▶ Simulating age group: {label} years")
        print(f"  Tidal volume: {AGE_GROUPS[age_key]['tidal_volume']*1e6:.0f} cm³, "
              f"Breathing freq: {AGE_GROUPS[age_key]['breathing_freq']}/min")

        results = run_simulation(
            n_particles=N_PARTICLES,
            age_group=age_key,
            pm_sizes=PM_SIZES
        )
        all_age_results[age_key] = results

    # Print detailed table for D-category (21-30 yrs) — matches paper's Table 4.1
    d_key = 'D (21-30 yrs)'
    print_results_table(all_age_results[d_key], "D-Category Workers (21-30 yrs)")

    # Apply paper's Monte Carlo formula for 8hr exposure
    print("\n▶ Monte Carlo 8-Hour Exposure Formula (Paper's method):")
    print(f"  Formula: MC = (30 × N × ρ) / (TV × BF)")
    for n in [250, 1000, 2500]:
        mc = monte_carlo_formula(n, 1.0, 2800, 15)
        print(f"  N={n:>5} particles → MC factor = {mc:.4f}")

    # Generate visualizations
    print("\n▶ Generating visualizations...")
    generate_3d_scatter(all_age_results[d_key], 'pm_3d_scatter.png')
    generate_deposition_bar_charts(all_age_results, 'pm_deposition_bars.png')
    generate_heatmap(all_age_results[d_key], 'D-Category (21-30 yrs)', 'pm_heatmap.png')

    print("\n✓ Simulation complete!")
    return all_age_results


if __name__ == '__main__':
    results = main()