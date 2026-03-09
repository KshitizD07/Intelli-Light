# Configuration Module (`configs/parameters.py`)

## Overview

This module contains **all configuration parameters** for the IntelliLight system, organized into logical classes for easy management and tuning.

## Design Philosophy

- **Single Source of Truth**: All system parameters are defined here
- **No Magic Numbers**: Every constant is documented and named
- **Easy Tuning**: Change behavior by editing this file, not the code
- **Real-world Ready**: Parameters are organized for deployment scenarios

## Configuration Classes

### 1. `SUMOConfig`
SUMO simulation engine settings
- Binary path and optimization flags
- File locations and timeouts

### 2. `NetworkTopology`
Traffic network structure (must match SUMO network files)
- Detector IDs by direction
- Lane IDs by direction
- Traffic light programs

### 3. `SignalTiming`
Traffic signal timing parameters
- Green light durations
- Safety intervals (yellow, all-red)
- Fairness constraints

### 4. `ScenarioConfig`
Traffic scenario generation
- Traffic scenarios (rush hour, weekend)
- Vehicle type distributions
- Emergency vehicle settings

### 5. `EpisodeConfig`
Simulation episode settings
- Episode duration
- Simulation time step

### 6. `TrainingConfig`
RL training hyperparameters
- PPO algorithm settings
- Training schedule
- Vectorized environment settings

### 7. `ObservationConfig`
State observation parameters
- Observation space shape
- Normalization bounds
- Feature indices

### 8. `ActionConfig`
Action space definition
- Available actions
- Action dimensions

### 9. `RewardConfig`
Multi-objective reward function
- Component weights (wait time, throughput, fairness)
- Penalty thresholds
- Normalization factors

### 10. `CurriculumConfig`
Progressive training difficulty
- Curriculum stages (light → heavy traffic)
- Transition thresholds

### 11. `ResourceConfig`
System resource management
- File cleanup policies
- Memory limits
- Process management

### 12. `LoggingConfig`
Logging and monitoring
- Log levels and formats
- Performance tracking

### 13. `DeploymentConfig`
Real-world deployment settings
- Failsafe configuration
- Health monitoring
- Edge device settings

### 14. `Paths`
Auto-generated file paths
- Creates all necessary directories
- Provides consistent path references

## Usage

```python
# Import configuration classes
from configs.parameters import SUMOConfig, SignalTiming, RewardConfig

# Access parameters
binary = SUMOConfig.BINARY
min_green = SignalTiming.MIN_GREEN
wait_weight = RewardConfig.WAIT_TIME_WEIGHT

# Use paths
from configs.parameters import Paths
model_dir = Paths.MODELS_DIR
```

## Testing

Run the test script to verify configuration:

```bash
python test_parameters.py
```

This will:
- Test all imports
- Validate parameter values
- Check directory creation
- Verify configuration consistency

## Customization Guide

### For Different Intersections
Modify `NetworkTopology` to match your SUMO network:
- Update detector IDs
- Update lane IDs
- Update traffic light ID

### For Different Traffic Patterns
Modify `ScenarioConfig.TRAFFIC_VOLUMES`:
- Add new scenarios
- Adjust vehicle volumes
- Change vehicle type distribution

### For Different Training Goals
Modify `RewardConfig` weights:
- Increase `THROUGHPUT_WEIGHT` for efficiency
- Increase `FAIRNESS_WEIGHT` for equity
- Increase `EMERGENCY_WEIGHT` for safety

### For Faster/Slower Training
Modify `TrainingConfig`:
- Increase `N_ENVS` for faster training (needs more CPU)
- Adjust `LEARNING_RATE` for convergence speed
- Change `TOTAL_TIMESTEPS` for training duration

## Validation

The `validate_configuration()` function checks:
- Green light durations are consistent
- Vehicle distribution sums to 1.0
- Observation shape matches expected dimensions
- Curriculum stages match transitions

## Directory Structure Created

```
IntelliLight/
├── data/
│   └── generated_routes/     # Auto-generated traffic files
├── configs/
│   └── sumo/                 # SUMO network files
├── models/
│   └── checkpoints/          # Saved models
└── logs/                     # Training logs
```

## Notes

- All timing values are in **seconds**
- All distance values are in **meters**
- Vehicle counts are **normalized** for RL
- Paths are created automatically on import