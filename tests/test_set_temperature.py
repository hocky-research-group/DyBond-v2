"""DyBond rates driven by the thermostat's set temperature instead of the kinetic temperature.

``DyBondUpdater(use_set_temperature=True)`` evaluates the temperature switch and the
Metropolis factor at the kT of the integrator's Langevin (or Brownian) method, read at every
sweep, so a later ``method.kT = ...`` takes effect on the next sweep. The default
(``False``) keeps the instantaneous translational kinetic temperature.

The tests hold two particles at the bond rest length with a kinetic temperature far above
Tmelt (huge mass: the thermostat cannot cool them within the test) and a set temperature far
below it. Binding is certain when cold (kon dt period = 1) and impossible when hot
(kon_melt = 0); unbinding is impossible when cold (koff = 0) and certain when hot
(koff_melt dt period = 1).

In every mode the per-sweep probabilities kon dt period and koff dt period use the dt of the
active integrator at that sweep, so kon and koff stay rates per unit time when dt changes.

Run inside the HOOMD container:
    pytest test_set_temperature.py -v
"""

import numpy as np
import pytest

hoomd = pytest.importorskip("hoomd")
pytest.importorskip("hoomd.dybond")
from hoomd.dybond import DyBondUpdater

DT = 0.001
MASS = 1e20
T_MELT = 1.0
T_COLD = 0.5
T_HOT = 2.0
DT_SHORT = 1e-6
DT_LONG = 1.0


def _devices():
    """Every device the host can build, as (id, factory) pairs."""
    devices = [("cpu", hoomd.device.CPU)]
    try:
        hoomd.device.GPU()
    except Exception:
        return devices
    return devices + [("gpu", hoomd.device.GPU)]


DEVICES = [pytest.param(f, id=i) for i, f in _devices()]


def _hot_pair_sim(device, set_kT, n_methods=1, method="langevin", bonded=False):
    """Two frozen particles at r = r0 with kinetic temperature >= 3; thermostat set to set_kT.

    device (callable): hoomd device factory. set_kT (float): thermostat kT.
    n_methods (int): number of Langevin methods to add (0, 1, or 2; each on one particle when 2).
    method (str): "langevin", "brownian", or "nve" for the single method's class.
    bonded (bool): start with a P-P bond between the two particles.
    Returns (sim, langevin_or_None, nlist, thermo, harmonic bond).
    """
    snap = hoomd.Snapshot()
    snap.particles.N = 2
    snap.particles.types = ["P"]
    snap.particles.typeid[:] = [0, 0]
    snap.particles.position[:] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    snap.particles.mass[:] = MASS
    # KE = MASS v^2 = 9 -> T_kin = 2 KE / dof >= 3 for dof <= 6
    v = np.sqrt(9.0 / MASS)
    snap.particles.velocity[:] = [[0.0, v, 0.0], [0.0, -v, 0.0]]
    snap.configuration.box = [20.0, 20.0, 20.0, 0, 0, 0]
    snap.bonds.types = ["P-P"]
    if bonded:
        snap.bonds.N = 1
        snap.bonds.typeid[:] = [0]
        snap.bonds.group[:] = [[0, 1]]
    sim = hoomd.Simulation(device=device(), seed=11)
    sim.create_state_from_snapshot(snap)
    sim.operations.tuners.clear()
    integ = hoomd.md.Integrator(dt=DT)
    nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=[])
    wca = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    wca.params[("P", "P")] = dict(epsilon=0.0, sigma=1.0)
    wca.r_cut[("P", "P")] = 2.0 ** (1.0 / 6.0)
    integ.forces.append(wca)
    harm = hoomd.md.bond.Harmonic()
    harm.params["P-P"] = dict(k=200.0, r0=1.0)
    integ.forces.append(harm)
    thermostat = None
    if n_methods == 1 and method == "langevin":
        thermostat = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=set_kT)
        integ.methods.append(thermostat)
    elif n_methods == 1 and method == "brownian":
        thermostat = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=set_kT)
        thermostat.gamma["P"] = MASS    # overdamped steps of ~ sqrt(2 kT dt / gamma): frozen
        integ.methods.append(thermostat)
    elif n_methods == 1 and method == "nve":
        integ.methods.append(hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All()))
    elif n_methods == 2:
        for tag in (0, 1):
            integ.methods.append(hoomd.md.methods.Langevin(filter=hoomd.filter.Tags([tag]),
                                                           kT=set_kT))
    sim.operations.integrator = integ
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    return sim, thermostat, nlist, thermo, harm


def _updater(nlist, thermo, harm, **kwargs):
    """DyBond updater for P-P with a sharp switch at T_MELT, sweeping every step.

    nlist, thermo, harm: simulation objects from _hot_pair_sim. kwargs: passed to the updater.
    """
    upd = DyBondUpdater(hoomd.trigger.Periodic(1), nlist, thermo, **kwargs)
    upd[("P", "P")] = dict(bond=harm, bond_typename="P-P", kon=1.0 / DT, koff=0.0,
                           rmin=0.5, rmax=1.3, r0=1.0, metropolisFlag=True,
                           selfAvoidFlag=False, Tmelt=T_MELT, alpha=200.0,
                           kon_melt=0.0, koff_melt=1.0 / DT)
    return upd


def _num_bonds(sim):
    """Number of bonds in the current state."""
    return sim.state.get_snapshot().bonds.N


@pytest.mark.parametrize("device", DEVICES)
def test_default_follows_kinetic_temperature(device):
    """Flag off (default): the hot kinetic temperature blocks binding despite a cold thermostat."""
    sim, _, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD)
    sim.operations.updaters.append(_updater(nlist, thermo, harm))
    sim.run(3)
    assert _num_bonds(sim) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["langevin", "brownian"])
def test_set_temperature_flag_follows_thermostat(device, method):
    """Flag on: the cold set temperature allows binding although the particles are hot."""
    sim, _, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD, method=method)
    upd = _updater(nlist, thermo, harm, use_set_temperature=True)
    sim.operations.updaters.append(upd)
    sim.run(3)
    assert upd.use_set_temperature
    assert _num_bonds(sim) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_external_kT_change_takes_effect_next_sweep(device):
    """Assigning method.kT outside DyBond (a new Variant) switches the rates at the next sweep."""
    sim, langevin, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD)
    sim.operations.updaters.append(_updater(nlist, thermo, harm, use_set_temperature=True))
    sim.run(3)
    assert _num_bonds(sim) == 1
    langevin.kT = T_HOT
    sim.run(3)
    assert _num_bonds(sim) == 0
    langevin.kT = hoomd.variant.Constant(T_COLD)
    sim.run(3)
    assert _num_bonds(sim) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_ramp_variant_is_evaluated_at_the_current_timestep(device):
    """A kT ramp crossing Tmelt: bonded while cold, broken once the ramp passes Tmelt."""
    sim, langevin, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD)
    langevin.kT = hoomd.variant.Ramp(A=T_COLD, B=T_HOT, t_start=0, t_ramp=20)
    sim.operations.updaters.append(_updater(nlist, thermo, harm, use_set_temperature=True))
    sim.run(3)                      # kT ~ 0.6
    assert _num_bonds(sim) == 1
    sim.run(20)                     # kT = 2.0
    assert _num_bonds(sim) == 0


@pytest.mark.parametrize("n_methods,method", [(0, "langevin"), (1, "nve"), (2, "langevin")])
def test_set_temperature_needs_exactly_one_thermostat_method(n_methods, method):
    """Flag on with no Langevin/Brownian method, or with two, fails at attach."""
    sim, _, nlist, thermo, harm = _hot_pair_sim(hoomd.device.CPU, set_kT=T_COLD,
                                                n_methods=n_methods, method=method)
    sim.operations.updaters.append(_updater(nlist, thermo, harm, use_set_temperature=True))
    with pytest.raises(ValueError, match="use_set_temperature"):
        sim.run(1)


@pytest.mark.parametrize("device", DEVICES)
def test_replaced_thermostat_method_is_used_next_sweep(device):
    """Removing the thermostat method and appending a new one: DyBond follows the new method."""
    sim, langevin, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD)
    sim.operations.updaters.append(_updater(nlist, thermo, harm, use_set_temperature=True))
    sim.run(3)
    assert _num_bonds(sim) == 1
    integ = sim.operations.integrator
    integ.methods.remove(langevin)
    integ.methods.append(hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=T_HOT))
    sim.run(3)
    assert _num_bonds(sim) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_removed_thermostat_method_fails_at_next_sweep(device):
    """Replacing the only thermostat with a method that has no kT raises at the next sweep."""
    sim, langevin, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD)
    sim.operations.updaters.append(_updater(nlist, thermo, harm, use_set_temperature=True))
    sim.run(2)
    integ = sim.operations.integrator
    integ.methods.remove(langevin)
    integ.methods.append(hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All()))
    with pytest.raises(ValueError, match="use_set_temperature"):
        sim.run(2)


def test_rejected_flag_assignment_leaves_the_previous_source():
    """Setting the flag after attach without a single thermostat raises and changes nothing."""
    sim, _, nlist, thermo, harm = _hot_pair_sim(hoomd.device.CPU, set_kT=T_COLD, n_methods=2)
    upd = _updater(nlist, thermo, harm)
    sim.operations.updaters.append(upd)
    sim.run(2)
    with pytest.raises(ValueError, match="use_set_temperature"):
        upd.use_set_temperature = True
    assert upd.use_set_temperature is False
    sim.run(3)
    assert _num_bonds(sim) == 0     # still follows the hot kinetic temperature


def _replacement_integrator(set_kT, nlist, dt=DT):
    """A new integrator with the same forces as _hot_pair_sim and one Langevin at set_kT.

    set_kT (float): thermostat kT. nlist: neighbour list shared with the DyBond updater.
    dt (float): timestep of the new integrator.
    """
    integ = hoomd.md.Integrator(dt=dt)
    wca = hoomd.md.pair.LJ(nlist=nlist, mode="shift")
    wca.params[("P", "P")] = dict(epsilon=0.0, sigma=1.0)
    wca.r_cut[("P", "P")] = 2.0 ** (1.0 / 6.0)
    integ.forces.append(wca)
    harm = hoomd.md.bond.Harmonic()
    harm.params["P-P"] = dict(k=200.0, r0=1.0)
    integ.forces.append(harm)
    integ.methods.append(hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=set_kT))
    return integ


@pytest.mark.parametrize("device", DEVICES)
def test_replaced_integrator_thermostat_is_used_next_sweep(device):
    """Replacing sim.operations.integrator: DyBond follows the active integrator's thermostat."""
    sim, _, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD)
    sim.operations.updaters.append(_updater(nlist, thermo, harm, use_set_temperature=True))
    sim.run(3)
    assert _num_bonds(sim) == 1
    sim.operations.integrator = _replacement_integrator(T_HOT, nlist)
    sim.run(3)
    assert _num_bonds(sim) == 0


def _rate_updater(nlist, thermo, harm, kon, koff, **kwargs):
    """DyBond updater for P-P with constant rates (no temperature switch), sweeping every step.

    nlist, thermo, harm: simulation objects from _hot_pair_sim. kon, koff (float): binding and
    unbinding rates per unit time. kwargs: passed to the updater.
    """
    upd = DyBondUpdater(hoomd.trigger.Periodic(1), nlist, thermo, **kwargs)
    upd[("P", "P")] = dict(bond=harm, bond_typename="P-P", kon=kon, koff=koff,
                           rmin=0.5, rmax=1.3, r0=1.0, metropolisFlag=True,
                           selfAvoidFlag=False)
    return upd


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_set_temperature", [False, True], ids=["kinetic", "set_kT"])
@pytest.mark.parametrize("how", ["replace", "in_place"])
@pytest.mark.parametrize("event", ["bind", "unbind"])
def test_rates_follow_integrator_dt_after_attach(device, use_set_temperature, how, event):
    """kon and koff are rates per unit time: each sweep uses the active integrator's dt.

    Attached at DT_SHORT, the rate 1 / DT_LONG gives a per-sweep probability of 1e-6, so the
    state holds; once dt becomes DT_LONG, by assignment or a new integrator, the probability is
    1 and the bond forms (or breaks) at the next sweep.
    """
    bonded = event == "unbind"
    sim, _, nlist, thermo, harm = _hot_pair_sim(device, set_kT=T_COLD, bonded=bonded)
    sim.operations.integrator.dt = DT_SHORT
    kon, koff = (0.0, 1.0 / DT_LONG) if bonded else (1.0 / DT_LONG, 0.0)
    sim.operations.updaters.append(_rate_updater(nlist, thermo, harm, kon, koff,
                                                 use_set_temperature=use_set_temperature))
    sim.run(3)
    assert _num_bonds(sim) == int(bonded)
    if how == "replace":
        sim.operations.integrator = _replacement_integrator(T_COLD, nlist, dt=DT_LONG)
    else:
        sim.operations.integrator.dt = DT_LONG
    sim.run(1)
    assert _num_bonds(sim) == int(not bonded)
