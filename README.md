# DyBond

Dynamic Bonding plugin for **HOOMD-blue v7**. This repository contains updates to the DyBond plugin to make it compatible with the Python API, modular architecture, and GPU/HIP features of HOOMD-blue v7.

---

## Overview

**DyBond** enables molecular dynamics simulations in HOOMD-blue where chemical or physical bonds form and break dynamically. Bonding decisions are made during the simulation run using distance thresholds and Metropolis Monte Carlo criteria, allowing the simulation of reversible polymer networks, self-assembling systems, supramolecular chemistry, and more.

### Key Features
- **Dynamic Bond Formation & Cleavage**: Define bond-forming and bond-breaking rates ($k_{\text{on}}$, $k_{\text{off}}$) and range limitations ($r_{\text{min}}$, $r_{\text{max}}$).
- **Self-Avoidance**: `selfAvoidFlag` (required boolean parameter) specifies whether to exclude candidate partners belonging to the same molecule, where molecules are traced from the *static* bond topology when the updater is configured. Set it to `False` to permit intramolecular bonding, which is how loop defects form in a network of multi-arm building blocks, or `True` to enforce intermolecular-only bonding. The flag is honored identically by the CPU and GPU candidate searches.
- **Temperature Source**: `DyBondUpdater(..., use_set_temperature=False)`. By default the temperature switch and the Metropolis factor use the instantaneous translational kinetic temperature from the `thermo` compute. With `use_set_temperature=True` they use the `kT` of the integrator's single Langevin or Brownian method, looked up at every sweep (one Python call per sweep), so assigning `method.kT` (a new value or Variant), replacing the method, or replacing `sim.operations.integrator` takes effect at the next sweep; leaving no such method raises `ValueError`, as does an integrator `dt` different from the one at attach (the rate probabilities are built with the attach-time `dt`).
- **HOOMD-blue 7 Compatibility**: Full compatibility with the `hoomd.Simulation` workflow, triggers, and operations API.
- **GPU Acceleration**: Native GPU-accelerated bonding checks and event compaction built using CUDA/HIP.

---

## Installation

To set up, compile, and install the plugin:

### 1. Set Up Python & HOOMD-blue Environment
Ensure you have Python and **HOOMD-blue v4 or newer** installed. The required build and test dependencies are:
* **HOOMD-blue** (v4.0.0 or newer)
* **CMake** (v3.9 or newer)
* **Eigen3**
* **pybind11**
* **NumPy**
* **GSD**
* **pytest**

Please refer to the [HOOMD-blue Installation Documentation](https://hoomd-blue.readthedocs.io/) for detailed guides on setting up your environment (e.g., using Conda/Miniforge).

### 2. Build and Install
Activate your target Python environment and run the installation script from the repository root:
```bash
conda activate <env_name>
bash install.sh
```
*Note: `install.sh` performs the following steps automatically:*
- *Clears build caches to prevent cross-environment compiler/linker pollution.*
- *Verifies that all required dependencies are installed, attempting to automatically download any missing packages via `mamba` or `conda` if you are in an active Conda environment.*
- *Performs a safety check to verify that the active CUDA compiler (`nvcc`) version matches the CUDA version that your HOOMD-blue installation was built with to prevent compiler errors.*
- *Compiles the dynamic bonding plugin and installs it into the active environment's `site-packages`.*

---

## Quick Start Example

Here is a basic script demonstrating how to configure and run the DyBond plugin in a HOOMD-blue v7 simulation:

```python
import hoomd
from hoomd.md import nlist, bond
from hoomd.dybond import DyBond, DyBondUpdater

# 1. Initialize simulation and state
device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=42)
sim.create_state_from_gsd("init.gsd")

# 2. Setup standard MD integrator and forces
integrator = hoomd.md.Integrator(dt=0.005)
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
integrator.methods.append(langevin)

# Define standard potential (e.g. Harmonic) for the dynamic bonds
harmonic = bond.Harmonic()
harmonic.params['patch'] = dict(k=30.0, r0=1.0)
integrator.forces.append(harmonic)
sim.operations.integrator = integrator

# 3. Setup Cell neighbor list & Thermodynamic quantities compute
cell = nlist.Cell(buffer=0.4)
thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermo)

# 4. Define Dynamic Bond behavior
# Form bonds between 'patch_A' and 'patch_B' with given rates and ranges
dybond_pot = DyBond(
    name="patch", 
    bond=harmonic, 
    kon=100.0, 
    koff=2.06e-7, 
    rmin=0.8, 
    rmax=1.2,
    nameA="patch_A",
    nameB="patch_B",
    selfAvoidFlag=True
)

# 5. Add DyBondUpdater to simulation
# The updater checks for bond formation and cleavage at regular periods
dybond_updater = DyBondUpdater(
    trigger=hoomd.trigger.Periodic(10), 
    nlist=cell, 
    thermo=thermo
)
dybond_updater.append(dybond_pot)
sim.operations.updaters.append(dybond_updater)

# 6. Run simulation
sim.run(10000)
```

---

## Project Structure

```bash
DyBond/
├── src/
│   ├── cpp/              # C++ and CUDA source files (CPU and GPU/HIP logic)
│   └── dybond/           # Python package wrapper modules
├── tests/                # Unit tests (reference and integrated HOOMD plugin tests)
├── CMakeLists.txt        # CMake build configuration
├── install.sh            # Environment-agnostic compile and install script
├── LICENSE               # MIT License
└── README.md             # This file
```

---

## Modifications

- derivative-contributor: G. Hocky (with Claude), 2026-10-15, developed as `feature/set-temperature` from 09aef76 and carried onto main af24ffc —
  added an opt-in temperature source for the rate switch and Metropolis factor.
  `DyBondUpdater` (C++) gained `setTemperatureCallback(callable)`; when set, each sweep calls
  it with the timestep (GIL acquired) and passes the bonding code a
  one-element properties array whose kinetic-energy entry is `kT * translational_dof / 2`, so
  the unchanged CPU functions and GPU kernels (which compute `2 properties[0] / dof`) see the set
  temperature. Python `DyBondUpdater` gained `use_set_temperature` (default `False`; settable
  after attach, validated before it changes anything); its callback (held through a weak
  reference) evaluates the kT of the active `sim.operations.integrator`'s single
  Langevin/Brownian method, raising `ValueError` for any other count or when the integrator
  `dt` differs from the attach-time `dt` used in the rate probabilities. Tests in `tests/test_set_temperature.py`. Motivation: the kinetic
  temperature of a few-thousand-particle system fluctuates by ~1 % (sqrt(2/(3N))), which a steep
  switch turns into large rate fluctuations shared by every bond.
