import hoomd
import hoomd.md
import hoomd.dybond as dyb_gpu
import numpy as np

# 1. Initialize GPU device and simulation
device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device, seed=42)

# 2. Build snapshot with two particles of types A and B
snap = hoomd.Snapshot()
snap.particles.N = 2
snap.particles.types = ["A", "B"]
snap.particles.typeid[:] = [0, 1]  # A and B
snap.particles.position[:] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
snap.particles.mass[:] = [1.0, 1.0]
snap.configuration.box = [10.0, 10.0, 10.0, 0, 0, 0]
snap.bonds.types = ["A-B"]

sim.create_state_from_snapshot(snap)

# 3. Create MD integrator, neighbor list, and forces
dt = 0.001
integrator = hoomd.md.Integrator(dt=dt)

cell_nlist = hoomd.md.nlist.Cell(buffer=0.4)

# Dynamic bonding uses a standard HOOMD bond force to evaluate bond potential energy
harmonic_bond = hoomd.md.bond.Harmonic()
harmonic_bond.params["A-B"] = dict(k=200.0, r0=1.0)
integrator.forces.append(harmonic_bond)

# Integrate equations of motion using Langevin dynamics
integrator.methods.append(
    hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
)
sim.operations.integrator = integrator

# Thermodynamic compute is required to supply temperature to the updater
thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermo)

# 4. Set up dynamic bonding spec
AB_dybond = dyb_gpu.DyBond(
    name="A-B",
    bond=harmonic_bond,
    kon=150.0,            # Transition rate on (steps^-1)
    koff=50.0,           # Transition rate off (steps^-1)
    rmin=0.5,            # Minimum binding distance
    rmax=1.3,            # Maximum binding distance
    nameA="A",
    nameB="B",
    selfAvoidFlag=True,
    metropolisFlag=True # Enable energy penalty factor
)

# 5. Create DyBondUpdater and append the spec
updater = dyb_gpu.DyBondUpdater(
    trigger=hoomd.trigger.Periodic(1), # Run every step
    nlist=cell_nlist,
    thermo=thermo
)
updater.append(AB_dybond)
sim.operations.updaters.append(updater)

# 6. Run the simulation
print("Initial number of dynamic bonds:", AB_dybond.bonds)
sim.run(1000)
print("Number of dynamic bonds after 1000 steps:", AB_dybond.bonds)
