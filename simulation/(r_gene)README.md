# Simulation Module (`simulation/route_generator.py`)

## Overview

This module generates realistic SUMO route files (.rou.xml) for traffic simulation. It creates diverse traffic scenarios with support for curriculum learning and emergency vehicles.

## Features

### Traffic Scenarios
- **Morning Rush**: Heavy inbound traffic (residential → business)
- **Evening Rush**: Heavy outbound traffic (business → residential)
- **Weekend**: Balanced, moderate traffic in all directions
- **Random**: Randomly selects one of the above

### Curriculum Learning
- **Stage 0** (LOW): ~400 vehicles/hour - Light traffic for initial learning
- **Stage 1** (MEDIUM): ~600 vehicles/hour - Moderate traffic
- **Stage 2** (HIGH): ~800 vehicles/hour - Heavy traffic for advanced training

### Vehicle Types
- **Cars** (70%): Standard passenger vehicles
- **Two-wheelers** (20%): Motorcycles with higher unpredictability
- **Buses** (10%): Larger, slower vehicles
- **Emergency Vehicles**: Ambulances with priority (30% of episodes)

### Resource Management
- Automatic cleanup of old route files
- Prevents disk space issues during long training runs

## Usage

### Basic Usage

```python
from simulation.route_generator import RouteGenerator

# Create generator
generator = RouteGenerator()

# Generate a route file
filename = generator.generate_unique_filename()
info = generator.generate_route_file(
    filename,
    scenario="MORNING_RUSH",
    curriculum_stage=1
)

print(f"Generated {info['total_flow']} vehicles/hour")
print(f"Emergency vehicle: {info['emergency'] is not None}")
```

### With Curriculum Learning

```python
# Stage 0: Light traffic for early training
info = generator.generate_route_file(
    "route_stage0.rou.xml",
    scenario="RANDOM",
    curriculum_stage=0  # ~400 veh/hour
)

# Stage 2: Heavy traffic for advanced training
info = generator.generate_route_file(
    "route_stage2.rou.xml",
    scenario="MORNING_RUSH",
    curriculum_stage=2  # ~800 veh/hour
)
```

### Cleanup Old Files

```python
# Keep only the 50 most recent route files
cleaned_count = generator.cleanup_old_routes(max_files=50)
print(f"Removed {cleaned_count} old files")
```

### Backward Compatibility

```python
# Convenience functions for compatibility with old code
from simulation.route_generator import generate_route_file, cleanup_old_routes

info = generate_route_file("route.rou.xml", scenario="WEEKEND")
cleanup_old_routes()
```

## Generated File Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <!-- Vehicle Type Definitions -->
  <vType id="car" accel="2.9" decel="7.5" sigma="0.5" length="5" maxSpeed="50"/>
  <vType id="2-wheeler" accel="2.5" decel="6.0" sigma="0.7" length="2.5" maxSpeed="40"/>
  <vType id="bus" accel="1.5" decel="4.0" sigma="0.5" length="12" maxSpeed="30"/>
  <vType id="ambulance" accel="4.0" decel="8.0" sigma="0.2" length="7" maxSpeed="60" vClass="emergency"/>

  <!-- Route Definitions -->
  <route id="W_E" edges="W1_to_J1 J1_to_E1"/>
  <route id="E_W" edges="E1_to_J1 J1_to_W1"/>
  <route id="N_S" edges="N1_to_J1 J1_to_S1"/>
  <route id="S_N" edges="S1_to_J1 J1_to_N1"/>

  <!-- Traffic Flows -->
  <flow id="flow_WE" type="car" route="W_E" begin="0" end="1800" vehsPerHour="650"/>
  <flow id="flow_EW" type="car" route="E_W" begin="0" end="1800" vehsPerHour="350"/>
  <flow id="flow_NS" type="car" route="N_S" begin="0" end="1800" vehsPerHour="150"/>
  <flow id="flow_SN" type="car" route="S_N" begin="0" end="1800" vehsPerHour="150"/>

  <!-- Emergency Vehicle (if generated) -->
  <vehicle id="ambulance_a1b2c3d4" type="ambulance" route="W_E" depart="450" departLane="best"/>
</routes>
```

## API Reference

### `RouteGenerator`

#### `__init__(route_dir=None)`
Initialize the generator.
- `route_dir`: Directory for route files (defaults to config)

#### `generate_route_file(filename, scenario, curriculum_stage, complexity_multiplier)`
Generate a SUMO route file.
- `filename`: Output file path
- `scenario`: Traffic pattern ("MORNING_RUSH", "EVENING_RUSH", "WEEKEND", "RANDOM")
- `curriculum_stage`: 0, 1, or 2
- `complexity_multiplier`: Additional traffic density multiplier (default 1.0)

Returns: Dictionary with scenario information

#### `generate_unique_filename(prefix="route")`
Generate a unique filename with UUID.
- `prefix`: Filename prefix

Returns: Full path to unique file

#### `cleanup_old_routes(max_files=None)`
Remove old route files.
- `max_files`: Maximum files to keep (defaults to config)

Returns: Number of files removed

## Testing

Run the comprehensive test suite:

```bash
python test_route_generator.py
```

Tests include:
- Module import
- RouteGenerator initialization
- All traffic scenarios
- All curriculum stages
- Emergency vehicle generation
- File cleanup
- XML structure validation
- Backward compatibility

## Dependencies

```bash
pip install numpy  # For config (already installed)
```

## Configuration

All parameters are defined in `configs/parameters.py`:

- **ScenarioConfig**: Traffic scenarios and vehicle types
- **EpisodeConfig**: Simulation duration
- **CurriculumConfig**: Learning stages
- **ResourceConfig**: Cleanup settings
- **Paths**: File locations

Modify these to customize behavior without changing code.

## Integration

This module integrates with:
- **configs/parameters.py**: Gets all configuration
- **SUMO**: Generates files for SUMO simulation
- **RL training**: Provides diverse training scenarios
- **Curriculum learning**: Progressive difficulty scaling

## Notes

- Route files are created in `data/generated_routes/`
- Files use UUID for uniqueness (prevents conflicts)
- Emergency vehicles appear in ~30% of scenarios
- Traffic volumes are randomized within scenario ranges
- All timing values are in seconds
- All flow rates are in vehicles per hour