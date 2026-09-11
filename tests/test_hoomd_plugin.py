"""Integration tests for the compiled DyBond HOOMD plugin.

These require a working HOOMD build with the ``hoomd.dybond`` plugin installed
(see ``Lab_notebook.md`` for the container build recipe). They are skipped
automatically where HOOMD is not importable, so the pure-Python reference tests
in ``test_dybond_reference.py`` can still run in a plain environment.

Run inside the HOOMD container:
    pytest test_hoomd_plugin.py -v
"""

import itertools

import numpy as np
import pytest

hoomd = pytest.importorskip("hoomd")
dyb = pytest.importorskip("hoomd.dybond")
from hoomd.dybond import DyBond as GPU_DyBond, DyBondUpdater as GPU_DyBondUpdater

class DyBond(hoomd.operation.Updater):
    def __init__(self, trigger: hoomd.trigger.Trigger, nlist, thermo, force_cpu=False):
        super().__init__(trigger)
        self.nlist = nlist
        self.thermo = thermo
        self.force_cpu = force_cpu
        self._param_sets = []
        self._updater = GPU_DyBondUpdater(trigger, nlist, thermo)

    def set_params(self, bond_type, A, B, nondybonds, r0,
                   rmin, rmax, kspring, flag_temperature, flag_force,
                   flag_metropolis, Tmelt, alpha, kon_init, kon_melt,
                   koff_init, koff_melt, checksteps, dt, particle_clusterids,
                   self_avoiding, userseed):
        self._param_sets.append({
            'bond_type': bond_type, 'A': A, 'B': B, 'r0': r0,
            'rmin': rmin, 'rmax': rmax, 'flag_temperature': flag_temperature,
            'flag_force': flag_force, 'flag_metropolis': flag_metropolis,
            'Tmelt': Tmelt, 'alpha': alpha, 'kon_init': kon_init,
            'kon_melt': kon_melt, 'koff_init': koff_init, 'koff_melt': koff_melt,
            'self_avoiding': self_avoiding, 'kspring': kspring, 'userseed': userseed,
            'dt': dt
        })

    def _attach_hook(self):
        self._updater._simulation = self._simulation
        self._updater.nlist._attach(self._simulation)
        self._updater.thermo._attach(self._simulation)

        bond_force = None
        for force in self._simulation.operations.integrator.forces:
            if isinstance(force, hoomd.md.bond.Bond):
                bond_force = force
                break
        if bond_force is None:
            bond_force = hoomd.md.bond.Harmonic()
            bond_force.params[self._param_sets[0]['bond_type']] = dict(k=200.0, r0=1.0)
            self._simulation.operations.integrator.forces.append(bond_force)

        for p in self._param_sets:
            sim_dt = self._simulation.operations.integrator.dt
            dt_ratio = p['dt'] / sim_dt if sim_dt > 0 else 1.0
            
            spec = GPU_DyBond(
                name=p['bond_type'],
                bond=bond_force,
                kon=p['kon_init'] * dt_ratio,
                koff=p['koff_init'] * dt_ratio,
                rmin=p['rmin'],
                rmax=p['rmax'],
                nameA=p['A'],
                nameB=p['B'],
                metropolisFlag=bool(p['flag_metropolis']),
                forceFlag=bool(p['flag_force']),
                selfAvoidFlag=bool(p['self_avoiding']),
                r0=p['r0']
            )
            if p['flag_force']:
                spec.setForceParams(p['flag_force'])
            if p['flag_temperature']:
                spec.setTemperatureParams(p['Tmelt'], p['alpha'], p['kon_melt'] * dt_ratio, p['koff_melt'] * dt_ratio)
            
            self._updater.append(spec)

        self._updater._attach_hook()
        self._cpp_obj = self._updater._cpp_obj

    def get_bonds_made(self, bond_type):
        for db in self._updater.dybonds:
            if db.name == bond_type:
                return db.bonds
        return 0

    def get_num_dynamic_bonds(self):
        return sum(db.bonds for db in self._updater.dybonds)

def _get_device():
    try:
        return hoomd.device.GPU()
    except Exception:
        return hoomd.device.CPU()


def _available_devices():
    """Every device the host can actually build, as (id, factory) pairs.

    Tests parameterized over this run once per device, so a CPU/GPU divergence in
    the bonding kernels fails the suite on any host that has both.
    """
    devices = [("cpu", hoomd.device.CPU)]
    try:
        hoomd.device.GPU()
    except Exception:
        return devices
    return devices + [("gpu", hoomd.device.GPU)]


def _make_sim(positions, typeids, types, bond_types, box_L, kT=0.1, seed=1,
              sorter_period=None, frozen=False, static_bonds=None, device=None,
              bond_force=None):
    """Build a Simulation. If ``frozen`` is True no integration method is added,
    so particles never move -- useful for holding an exact contact geometry while
    only the dynamic-bond kinetics evolve.

    ``static_bonds`` is a list of ``(bond_typeid, (tag_i, tag_j))`` placed in the
    initial snapshot. The updater traces these at startup to assign molecule tags,
    so they are what makes two reactive sites belong to the same molecule.
    ``device`` overrides the autodetected device. ``bond_force`` replaces the
    default harmonic bond potential, so a test can drive DyBond with FENE or
    Tether bonds instead; its params must already be set for every bond type.
    """
    n = len(positions)
    snap = hoomd.Snapshot()
    snap.particles.N = n
    snap.particles.types = list(types)
    snap.particles.typeid[:] = np.asarray(typeids, dtype=int)
    snap.particles.position[:] = positions
    if frozen:
        mass = 1e20
        snap.particles.mass[:] = mass
        rng = np.random.default_rng(seed)
        snap.particles.velocity[:] = rng.normal(0.0, np.sqrt(kT / mass), size=(n, 3))
    else:
        snap.particles.mass[:] = 1.0
    snap.configuration.box = [box_L, box_L, box_L, 0, 0, 0]
    snap.bonds.types = list(bond_types)
    if static_bonds:
        snap.bonds.N = len(static_bonds)
        snap.bonds.typeid[:] = np.asarray([bt for bt, _ in static_bonds], dtype=int)
        snap.bonds.group[:] = np.asarray([pair for _, pair in static_bonds], dtype=int)

    sim = hoomd.Simulation(device=_get_device() if device is None else device(),
                           seed=seed)
    sim.create_state_from_snapshot(snap)

    sim.operations.tuners.clear()
    if sorter_period is not None:
        sim.operations.tuners.append(
            hoomd.tune.ParticleSorter(trigger=hoomd.trigger.Periodic(sorter_period)))

    dt = 0.001
    integ = hoomd.md.Integrator(dt=dt)
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=[])
    wca = hoomd.md.pair.LJ(nlist=cell, mode="shift")
    for a, b in itertools.combinations_with_replacement(types, 2):
        wca.params[(a, b)] = dict(epsilon=1.0, sigma=1.0)
        wca.r_cut[(a, b)] = 2.0 ** (1.0 / 6.0)
    integ.forces.append(wca)
    if bond_force is None:
        bond_force = hoomd.md.bond.Harmonic()
        for bt in bond_types:
            bond_force.params[bt] = dict(k=200.0, r0=1.0)
    integ.forces.append(bond_force)
    integ.methods.append(
        hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT))
    sim.operations.integrator = integ
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    return sim, cell, thermo, n, dt


def _max_valence(sim, n):
    g = np.asarray(sim.state.get_snapshot().bonds.group)
    if len(g) == 0:
        return 0, 0
    counts = np.bincount(g.reshape(-1), minlength=n)
    return len(g), int(counts.max())


def test_small_dimers_form_with_valence_one():
    """Two isolated pairs -> exactly two bonds, valence 1."""
    pos = np.array([[0, 0, 0], [1.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]])
    sim, cell, thermo, n, dt = _make_sim(pos, [0, 0, 0, 0], ["P"], ["P-P"], 20.0)
    cs = 1
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    bonder.set_params(bond_type="P-P", A="P", B="P", nondybonds=0, r0=1.0,
                      rmin=0.8, rmax=1.3, kspring=200.0, flag_temperature=0,
                      flag_force=0, flag_metropolis=1, Tmelt=1.6, alpha=200.0,
                      kon_init=40.0, kon_melt=0.0, koff_init=0.001, koff_melt=0.0,
                      checksteps=cs, dt=dt, particle_clusterids="0 1 2 3",
                      self_avoiding=0, userseed=7)
    sim.operations.updaters.append(bonder)
    worst = 0
    for _ in range(20):
        sim.run(5)
        nb, mv = _max_valence(sim, n)
        worst = max(worst, mv)
    nb, _ = _max_valence(sim, n)
    assert nb == 2
    assert worst == 1


def test_valence_stays_one_under_particle_sorting():
    """Regression for the getNthTag bug: with a frequent ParticleSorter (so bond
    storage index != tag-set position and tag != index), valence must never
    exceed 1."""
    npair = 32
    pos = []
    for k in range(npair):
        x = k * 2.5
        pos.append([x, 0, 0]); pos.append([x + 1.0, 0, 0])
    pos = np.array(pos, float)
    pos[:, 0] -= pos[:, 0].mean()
    n = len(pos)
    L = npair * 2.5 + 5
    sim, cell, thermo, n, dt = _make_sim(pos, [0] * n, ["P"], ["P-P"], L,
                                         kT=0.5, sorter_period=3)
    cs = 1
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    bonder.set_params(bond_type="P-P", A="P", B="P", nondybonds=0, r0=1.0,
                      rmin=0.8, rmax=1.3, kspring=200.0, flag_temperature=0,
                      flag_force=0, flag_metropolis=1, Tmelt=1.6, alpha=200.0,
                      kon_init=40.0, kon_melt=0.0, koff_init=0.05, koff_melt=0.0,
                      checksteps=cs, dt=dt,
                      particle_clusterids=" ".join(str(i) for i in range(n)),
                      self_avoiding=0, userseed=7)
    sim.operations.updaters.append(bonder)
    worst = 0
    for _ in range(40):
        sim.run(25)
        _, mv = _max_valence(sim, n)
        worst = max(worst, mv)
    assert worst == 1


def test_dense_mixture_forms_dimers():
    """A dense P gas self-assembles into dimers (majority paired, valence 1)."""
    n_side, spacing = 5, 1.25
    coords = np.array(list(itertools.product(range(n_side), repeat=3)), float)
    L = n_side * spacing
    coords = (coords + 0.5) * spacing - L / 2
    n = len(coords)
    sim, cell, thermo, n, dt = _make_sim(coords, [0] * n, ["P"], ["P-P"], L,
                                         kT=1.0, seed=3)
    sim.operations.integrator.methods[0].gamma.default = 1.0
    cs = 20
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    bonder.set_params(bond_type="P-P", A="P", B="P", nondybonds=0, r0=1.0,
                      rmin=0.8, rmax=1.4, kspring=200.0, flag_temperature=0,
                      flag_force=0, flag_metropolis=1, Tmelt=1.6, alpha=200.0,
                      kon_init=40.0, kon_melt=0.0, koff_init=0.2, koff_melt=0.0,
                      checksteps=cs, dt=0.005,
                      particle_clusterids=" ".join(str(i) for i in range(n)),
                      self_avoiding=0, userseed=7)
    sim.operations.updaters.append(bonder)
    worst = 0
    for _ in range(12):
        sim.run(1000)
        nb, mv = _max_valence(sim, n)
        worst = max(worst, mv)
    nb, _ = _max_valence(sim, n)
    assert worst == 1
    assert 2 * nb / n > 0.4  # a substantial fraction paired into dimers


def test_two_bond_types_coexist_valence_one():
    """A-B and A-C dynamic bond types on one updater both populate; no A binder
    is ever bonded twice; get_bonds_made matches the snapshot."""
    n_side, spacing = 5, 1.3
    coords = np.array(list(itertools.product(range(n_side), repeat=3)), float)
    L = n_side * spacing
    coords = (coords + 0.5) * spacing - L / 2
    n = len(coords)
    typeids = [i % 3 for i in range(n)]
    sim, cell, thermo, n, dt = _make_sim(coords, typeids, ["A", "B", "C"],
                                         ["A-B", "A-C"], L, kT=1.0, seed=5)
    sim.operations.integrator.methods[0].gamma.default = 1.0
    cs = 20
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    common = dict(nondybonds=0, r0=1.0, rmin=0.8, rmax=1.4, kspring=200.0,
                  flag_temperature=0, flag_force=0, flag_metropolis=1,
                  Tmelt=1.6, alpha=200.0, kon_melt=0.0, koff_melt=0.0,
                  checksteps=cs, dt=0.005,
                  particle_clusterids=" ".join(str(i) for i in range(n)),
                  self_avoiding=0, userseed=7)
    bonder.set_params(bond_type="A-B", A="A", B="B", kon_init=40.0,
                      koff_init=0.5, **common)
    bonder.set_params(bond_type="A-C", A="A", B="C", kon_init=40.0,
                      koff_init=0.5, **common)
    sim.operations.updaters.append(bonder)

    worst = 0
    n_ab = n_ac = 0
    for _ in range(12):
        sim.run(1000)
        _, mv = _max_valence(sim, n)
        worst = max(worst, mv)
        snap = sim.state.get_snapshot()
        tid = np.asarray(snap.bonds.typeid)
        n_ab = int((tid == 0).sum())
        n_ac = int((tid == 1).sum())
    assert worst == 1
    assert n_ab > 0 and n_ac > 0
    # updater's per-type counter agrees with the actual bond table
    assert bonder.get_bonds_made("A-B") == n_ab
    assert bonder.get_bonds_made("A-C") == n_ac
    assert bonder.get_num_dynamic_bonds() == n_ab + n_ac


def test_ABCD_competition_matches_ideal_ratio():
    """Compiled-plugin version of the A/B/C/D competition.

    Each unit holds one A, one D, one B and one C at the exact rest length
    (A and D each 1.0 from both B and C). Four dynamic bond types A-B, A-C, D-B,
    D-C compete for the shared B and C binders. Particles are frozen so the
    contact geometry is fixed and the equilibrium is the ideal detailed-balance
    result: P(bond present) from the 6-state partition function

        Z = 1 + K_AB + K_AC + K_DB + K_DC + K_AB*K_DC + K_AC*K_DB,
        P(A-B) = K_AB (1 + K_DC)/Z, etc.,   K_XY = kon_XY/koff_XY.

    The measured populations must match these to within a few percent, and no
    binder is ever in two bonds.
    """
    Aid, Bid, Cid, Did = 0, 1, 2, 3
    xA = np.sqrt(1.0 - 0.7 ** 2)
    local = {Aid: [-xA, 0.7, 0.0], Did: [xA, 0.7, 0.0],
             Bid: [0.0, 0.0, 0.0], Cid: [0.0, 1.4, 0.0]}
    n_units, dx = 40, 12.0
    pos, typeids = [], []
    for u in range(n_units):
        for t in (Aid, Did, Bid, Cid):
            p = list(local[t]); p[0] += u * dx
            pos.append(p); typeids.append(t)
    pos = np.array(pos)
    pos[:, 0] -= pos[:, 0].mean()  # center the units in the box
    L = n_units * dx + 10.0

    sim, cell, thermo, n, dt = _make_sim(
        pos, typeids, ["A", "B", "C", "D"], ["A-B", "A-C", "D-B", "D-C"],
        L, seed=17, frozen=True)

    koff = 0.05
    K = {"A-B": 3.0, "A-C": 0.5, "D-B": 1.5, "D-C": 2.0}
    cs = 1
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    common = dict(nondybonds=0, r0=1.0, rmin=0.5, rmax=1.3, kspring=100.0,
                  flag_temperature=0, flag_force=0, flag_metropolis=0,  # exact contact
                  Tmelt=1.6, alpha=200.0, kon_melt=0.0, koff_melt=0.0,
                  checksteps=cs, dt=1.0,
                  particle_clusterids=" ".join(str(i) for i in range(n)),
                  self_avoiding=0, userseed=7)
    bonder.set_params(bond_type="A-B", A="A", B="B", kon_init=K["A-B"] * koff,
                      koff_init=koff, **common)
    bonder.set_params(bond_type="A-C", A="A", B="C", kon_init=K["A-C"] * koff,
                      koff_init=koff, **common)
    bonder.set_params(bond_type="D-B", A="D", B="B", kon_init=K["D-B"] * koff,
                      koff_init=koff, **common)
    bonder.set_params(bond_type="D-C", A="D", B="C", kon_init=K["D-C"] * koff,
                      koff_init=koff, **common)
    sim.operations.updaters.append(bonder)

    kab, kac, kdb, kdc = K["A-B"], K["A-C"], K["D-B"], K["D-C"]
    Z = 1 + kab + kac + kdb + kdc + kab * kdc + kac * kdb
    ideal = [kab * (1 + kdc) / Z, kac * (1 + kdb) / Z,
             kdb * (1 + kac) / Z, kdc * (1 + kab) / Z]

    sim.run(20000)  # burn in
    names = ["A-B", "A-C", "D-B", "D-C"]
    acc = np.zeros(4)
    n_samples, interval = 6000, 10
    worst = 0
    for i in range(n_samples):
        sim.run(interval)
        for ti, nm in enumerate(names):
            acc[ti] += bonder.get_bonds_made(nm)
        if i % 500 == 0:
            _, mv = _max_valence(sim, n)
            worst = max(worst, mv)
    measured = acc / n_samples / n_units

    assert worst == 1
    # With the per-type just-unbound exclusion there is no ordering bias, so the
    # frozen occupancies match the ideal partition function tightly.
    for ti in range(4):
        assert abs(measured[ti] - ideal[ti]) < 0.02, (names[ti], measured[ti], ideal[ti])
    # A's B:C preference and D's C:B preference follow the ideal ratios
    assert abs((measured[0] / measured[1]) / (ideal[0] / ideal[1]) - 1) < 0.12
    assert abs((measured[3] / measured[2]) / (ideal[3] / ideal[2]) - 1) < 0.12


def test_ABCD_dynamic_competition_rearranges():
    """Moving A/B/C/D soup at low epsilon: bonds form, break and swap partners
    constantly. This is the dynamic counterpart of the frozen test -- here the
    particles diffuse, so the closest-partner rule and the Metropolis distance
    factor are both live and partners genuinely compete geometrically.

    At low binding affinity (epsilon = ln(kon/koff) ~ 0-1) with a fast off-rate,
    bonds turn over many times over the run. We check the bookkeeping survives
    heavy rearrangement (valence never exceeds 1 -- the invariant the getNthTag
    bug used to break), that all four bond types stay populated, that partners
    actually swap (many more distinct bonded pairs than are ever present at
    once), and that the competition still favors the higher-rate partner
    (A prefers B, D prefers C), tracking the kon/koff ratio."""
    n_side, spacing = 6, 1.25
    coords = np.array(list(itertools.product(range(n_side), repeat=3)), float)
    L = n_side * spacing
    coords = (coords + 0.5) * spacing - L / 2
    n = len(coords)
    typeids = [i % 4 for i in range(n)]  # equimolar A,B,C,D
    sim, cell, thermo, n, dt = _make_sim(coords, typeids, ["A", "B", "C", "D"],
                                         ["A-B", "A-C", "D-B", "D-C"], L,
                                         kT=1.0, seed=23)
    sim.operations.integrator.methods[0].gamma.default = 1.0
    cs = 20
    # low epsilon (Keq ~ 1-3) with a fast off-rate -> frequent rearrangement
    koff = 2.0                      # Poff = koff*cs*dt = 0.2  -> bond life ~5 checks
    Keq = {"A-B": 3.0, "A-C": 1.0, "D-B": 1.0, "D-C": 3.0}
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    common = dict(nondybonds=0, r0=1.0, rmin=0.8, rmax=1.4, kspring=200.0,
                  flag_temperature=0, flag_force=0, flag_metropolis=1,
                  Tmelt=1.6, alpha=200.0, kon_melt=0.0, koff_melt=0.0,
                  checksteps=cs, dt=0.005,
                  particle_clusterids=" ".join(str(i) for i in range(n)),
                  self_avoiding=0, userseed=7)
    for name, (a, b) in {"A-B": ("A", "B"), "A-C": ("A", "C"),
                         "D-B": ("D", "B"), "D-C": ("D", "C")}.items():
        bonder.set_params(bond_type=name, A=a, B=b,
                          kon_init=Keq[name] * koff, koff_init=koff, **common)
    sim.operations.updaters.append(bonder)

    sim.run(5000)  # burn in
    names = ["A-B", "A-C", "D-B", "D-C"]
    totals = np.zeros(4)
    worst_valence = 0
    peak_simultaneous = 0
    seen_pairs = set()
    n_samples, interval = 150, 100
    for _ in range(n_samples):
        sim.run(interval)
        snap = sim.state.get_snapshot()
        g = np.asarray(snap.bonds.group)
        tid = np.asarray(snap.bonds.typeid)
        for ti in range(4):
            totals[ti] += int((tid == ti).sum())
        if len(g):
            counts = np.bincount(g.reshape(-1), minlength=n)
            worst_valence = max(worst_valence, int(counts.max()))
            peak_simultaneous = max(peak_simultaneous, len(g))
            for pair in g:
                seen_pairs.add((int(min(pair)), int(max(pair))))

    # 1) correctness invariant survives heavy turnover
    assert worst_valence == 1
    # 2) all four bond types stay populated
    assert all(totals[ti] > 0 for ti in range(4)), totals
    # 3) genuine rearrangement: far more distinct pairs than ever coexist
    assert peak_simultaneous > 10
    assert len(seen_pairs) > 3 * peak_simultaneous, (len(seen_pairs), peak_simultaneous)
    # 4) competition still favors the higher-rate partner (A->B, D->C)
    n_ab, n_ac, n_db, n_dc = totals
    assert n_ab > n_ac and n_dc > n_db, totals
    assert (n_ab / n_ac) > 1.5 and (n_dc / n_db) > 1.5, totals


def test_same_endpoint_bond_types_unbind_by_exact_bond_type():
    """Regression for unbinding by endpoint type instead of bond type.

    Two dynamic bond types share the same A/B particle types but have different
    HOOMD bond types and different off-rates. A pre-existing bond of the "slow"
    type must not be removed when the "fast" type updater runs."""
    pos = np.array([[0, 0, 0], [1.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]], float)
    snap = hoomd.Snapshot()
    snap.particles.N = 4
    snap.particles.types = ["A", "B"]
    snap.particles.typeid[:] = np.array([0, 1, 0, 1], dtype=int)
    snap.particles.position[:] = pos
    snap.particles.mass[:] = 1.0
    snap.configuration.box = [20.0, 20.0, 20.0, 0, 0, 0]
    snap.bonds.types = ["A-B-fast", "A-B-slow"]
    snap.bonds.N = 1
    snap.bonds.group[:] = np.array([[2, 3]], dtype=int)
    snap.bonds.typeid[:] = np.array([1], dtype=int)

    sim = hoomd.Simulation(device=_get_device(), seed=11)
    sim.create_state_from_snapshot(snap)

    dt = 0.001
    integ = hoomd.md.Integrator(dt=dt)
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=[])
    wca = hoomd.md.pair.LJ(nlist=cell, mode="shift")
    for a, b in itertools.combinations_with_replacement(["A", "B"], 2):
        wca.params[(a, b)] = dict(epsilon=1.0, sigma=1.0)
        wca.r_cut[(a, b)] = 2.0 ** (1.0 / 6.0)
    integ.forces.append(wca)
    harm = hoomd.md.bond.Harmonic()
    harm.params["A-B-fast"] = dict(k=200.0, r0=1.0)
    harm.params["A-B-slow"] = dict(k=200.0, r0=1.0)
    integ.forces.append(harm)
    sim.operations.integrator = integ
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    bonder = DyBond(hoomd.trigger.Periodic(1), cell, thermo)
    common = dict(nondybonds=0, r0=1.0, rmin=0.8, rmax=1.3, kspring=200.0,
                  flag_temperature=0, flag_force=0, flag_metropolis=0,
                  Tmelt=1.6, alpha=200.0, kon_init=0.0, kon_melt=0.0,
                  koff_melt=0.0, checksteps=1, dt=dt,
                  particle_clusterids="0 1 2 3", self_avoiding=0, userseed=7)
    bonder.set_params(bond_type="A-B-fast", A="A", B="B", koff_init=1000.0, **common)
    bonder.set_params(bond_type="A-B-slow", A="A", B="B", koff_init=0.0, **common)
    sim.operations.updaters.append(bonder)

    sim.run(2)
    snap = sim.state.get_snapshot()
    assert snap.bonds.N == 1
    assert int(np.asarray(snap.bonds.typeid)[0]) == 1
    assert bonder.get_bonds_made("A-B-fast") == 0
    assert bonder.get_bonds_made("A-B-slow") == 1


def test_zero_temperature_guards_bind_and_unbind():
    """At T=0 exact-contact binding and stretched slip-bond unbinding should be
    well-defined, not NaN-driven."""
    cs = 1

    # exact-contact binding with Metropolis enabled: factor should reduce to 1
    pos = np.array([[0, 0, 0], [1.0, 0, 0]], float)
    sim, cell, thermo, _, dt = _make_sim(pos, [0, 0], ["P"], ["P-P"], 20.0, frozen=True)
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    bonder.set_params(bond_type="P-P", A="P", B="P", nondybonds=0, r0=1.0,
                      rmin=0.8, rmax=1.3, kspring=200.0, flag_temperature=0,
                      flag_force=0, flag_metropolis=1, Tmelt=1.6, alpha=200.0,
                      kon_init=1000.0, kon_melt=0.0, koff_init=0.0, koff_melt=0.0,
                      checksteps=cs, dt=dt, particle_clusterids="0 1",
                      self_avoiding=0, userseed=7)
    sim.operations.updaters.append(bonder)
    sim.run(2)
    assert bonder.get_num_dynamic_bonds() == 1

    # stretched slip-bond at T=0: off-rate should become effectively certain
    snap = hoomd.Snapshot()
    snap.particles.N = 2
    snap.particles.types = ["P"]
    snap.particles.typeid[:] = np.array([0, 0], dtype=int)
    snap.particles.position[:] = np.array([[0, 0, 0], [1.1, 0, 0]], float)
    mass = 1e20
    snap.particles.mass[:] = mass
    rng = np.random.default_rng(13)
    snap.particles.velocity[:] = rng.normal(0.0, np.sqrt(1.0 / mass), size=(2, 3))
    snap.configuration.box = [20.0, 20.0, 20.0, 0, 0, 0]
    snap.bonds.types = ["P-P"]
    snap.bonds.N = 1
    snap.bonds.group[:] = np.array([[0, 1]], dtype=int)
    snap.bonds.typeid[:] = np.array([0], dtype=int)
    sim = hoomd.Simulation(device=_get_device(), seed=13)
    sim.create_state_from_snapshot(snap)
    integ = hoomd.md.Integrator(dt=dt)
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    integ.methods.append(langevin)
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=[])
    wca = hoomd.md.pair.LJ(nlist=cell, mode="shift")
    wca.params[("P", "P")] = dict(epsilon=1.0, sigma=1.0)
    wca.r_cut[("P", "P")] = 2.0 ** (1.0 / 6.0)
    integ.forces.append(wca)
    harm = hoomd.md.bond.Harmonic()
    harm.params["P-P"] = dict(k=200.0, r0=1.0)
    integ.forces.append(harm)
    sim.operations.integrator = integ
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    bonder.set_params(bond_type="P-P", A="P", B="P", nondybonds=0, r0=1.0,
                      rmin=0.8, rmax=1.3, kspring=200.0, flag_temperature=0,
                      flag_force=1.0, flag_metropolis=1, Tmelt=1.6, alpha=200.0,
                      kon_init=0.0, kon_melt=0.0, koff_init=1000.0, koff_melt=0.0,
                      checksteps=cs, dt=dt, particle_clusterids="0 1",
                      self_avoiding=0, userseed=7)
    sim.operations.updaters.append(bonder)
    sim.run(1)
    assert bonder.get_num_dynamic_bonds() == 0


def test_bond_resolution_typename_and_typeid():
    """Test that bond_typename and bond_typeid resolve correctly using the real hoomd package."""
    pos = np.array([[0, 0, 0], [1.0, 0, 0]], float)
    sim, cell, thermo, _, dt = _make_sim(pos, [0, 0], ["P"], ["P-P"], 20.0, frozen=True)
    
    # 1. Test using bond_typename directly
    updater1 = GPU_DyBondUpdater(hoomd.trigger.Periodic(1), cell, thermo)
    harm = hoomd.md.bond.Harmonic()
    harm.params["P-P"] = dict(k=200.0, r0=1.0)
    sim.operations.integrator.forces.append(harm)
    
    updater1[('P', 'P')] = {
        'bond': harm,
        'bond_typename': 'P-P',
        'kon': 10.0,
        'koff': 0.1,
    }
    spec1 = updater1[('P', 'P')]
    assert spec1.name == 'P-P'

    # 2. Test using bond_typeid with deferred resolution during simulation attachment
    updater2 = GPU_DyBondUpdater(hoomd.trigger.Periodic(1), cell, thermo)
    updater2[('P', 'P')] = {
        'bond': harm,
        'bond_typeid': 0,
        'kon': 10.0,
        'koff': 0.1,
    }
    spec2 = updater2[('P', 'P')]
    
    # Before simulation run/attachment, the name is the integer index
    assert spec2.name == 0
    
    # Attach updater and run 0 steps to trigger attachment
    sim.operations.updaters.append(updater2)
    sim.run(0)
    
    # After simulation attachment, the name is resolved to the string typename
    assert spec2.name == 'P-P'


# --------------------------------------------------------------------------
# Self-avoidance: same-molecule exclusion must be governed by the flag, and
# must be governed identically on every device.
# --------------------------------------------------------------------------

def _self_avoidance_sim(device, self_avoiding, seed=23):
    """Two reactive R sites tethered into one molecule, plus a free R pair.

    Layout (all distances 1.0, so every candidate sits at the bond rest length):

        R0 --static-- L2 --static-- R1        <- one molecule, R0/R1 in range
        R3           R4                       <- two singleton molecules

    R0 and R1 are 1.0 apart and are therefore eligible partners on distance
    grounds alone; only the molecule test can keep them apart. R3-R4 is the
    control and must bond regardless. Returns (n_intra, n_inter).
    """
    s3 = np.sqrt(3.0) / 2.0
    pos = np.array([
        [0.0, 0.0, 0.0],    # 0  R, molecule A
        [1.0, 0.0, 0.0],    # 1  R, molecule A
        [0.5, s3, 0.0],     # 2  L, molecule A (tether hub)
        [5.0, 0.0, 0.0],    # 3  R, singleton
        [6.0, 0.0, 0.0],    # 4  R, singleton
    ])
    typeids = [0, 0, 1, 0, 0]
    sim, cell, thermo, n, dt = _make_sim(
        pos, typeids, ["R", "L"], ["R-R", "tether"], 24.0,
        seed=seed, frozen=True, device=device,
        static_bonds=[(1, (0, 2)), (1, (2, 1))])

    cs = 1
    bonder = DyBond(hoomd.trigger.Periodic(cs), cell, thermo)
    bonder.set_params(bond_type="R-R", A="R", B="R", nondybonds=0, r0=1.0,
                      rmin=0.8, rmax=1.3, kspring=200.0, flag_temperature=0,
                      flag_force=0, flag_metropolis=0, Tmelt=1.6, alpha=200.0,
                      kon_init=40.0, kon_melt=0.0, koff_init=0.0, koff_melt=0.0,
                      checksteps=cs, dt=dt,
                      particle_clusterids=" ".join(str(i) for i in range(n)),
                      self_avoiding=int(self_avoiding), userseed=7)
    sim.operations.updaters.append(bonder)
    sim.run(400)

    snap = sim.state.get_snapshot()
    groups = np.asarray(snap.bonds.group)
    typeid = np.asarray(snap.bonds.typeid)
    dyn = {frozenset(map(int, g)) for g, t in zip(groups, typeid) if t == 0}
    n_intra = int(frozenset((0, 1)) in dyn)
    n_inter = int(frozenset((3, 4)) in dyn)
    return n_intra, n_inter


@pytest.mark.parametrize("device_id,device", _available_devices())
def test_self_avoidance_blocks_intramolecular_bond(device_id, device):
    """With the flag set, two sites on one molecule must never bond."""
    n_intra, n_inter = _self_avoidance_sim(device, self_avoiding=True)
    assert n_inter == 1, f"[{device_id}] control pair 3-4 failed to bond"
    assert n_intra == 0, (
        f"[{device_id}] sites 0 and 1 share a molecule and bonded anyway")


@pytest.mark.parametrize("device_id,device", _available_devices())
def test_self_avoidance_off_permits_intramolecular_bond(device_id, device):
    """Without the flag, the same two sites are allowed to bond.

    This is the half that the CPU path used to get wrong: it applied the
    molecule test unconditionally, so intramolecular bonding was impossible on
    CPU no matter how the flag was set, while GPU honored it.
    """
    n_intra, n_inter = _self_avoidance_sim(device, self_avoiding=False)
    assert n_inter == 1, f"[{device_id}] control pair 3-4 failed to bond"
    assert n_intra == 1, (
        f"[{device_id}] self_avoiding=False but sites 0 and 1 never bonded")


# --------------------------------------------------------------------------
# The Metropolis acceptance factor must use whichever bond potential the
# integrator uses, not an assumed harmonic form. This is what generalizes the
# scheme of Mitra et al. (Soft Matter 19, 4223, 2023), where only a harmonic
# bond was supported.
# --------------------------------------------------------------------------

BOND_POTENTIALS = {
    # name -> (factory, expected plugin class stem)
    "harmonic": (lambda: _harmonic_force(k=20.0, r0=1.0), "DyBondHarmonic"),
    "fene": (lambda: _fene_force(k=2.0, r0=2.0), "DyBondFENE"),
    "tether": (lambda: _tether_force(), "DyBondTether"),
}


def _harmonic_force(k, r0, bond_types=("R-R",)):
    f = hoomd.md.bond.Harmonic()
    for bt in bond_types:
        f.params[bt] = dict(k=k, r0=r0)
    return f


def _fene_force(k, r0, bond_types=("R-R",)):
    # sigma is chosen so the WCA part is already cut off at the separations
    # used below; the FENE stretching term then sets the whole bond energy.
    f = hoomd.md.bond.FENEWCA()
    for bt in bond_types:
        f.params[bt] = dict(k=k, r0=r0, epsilon=1.0, sigma=0.9, delta=0.0)
    return f


def _tether_force(bond_types=("R-R",)):
    f = hoomd.md.bond.Tether()
    for bt in bond_types:
        f.params[bt] = dict(k_b=10.0, l_min=0.7, l_c1=0.9, l_c0=1.2, l_max=1.4)
    return f


def _bond_energy(make_force, bond_type, separation, device):
    """Energy HOOMD itself assigns to one bond of this type at ``separation``.

    Measured rather than recomputed in the test, so the assertions below compare
    DyBond's acceptance factor against the same evaluator the integrator uses
    instead of against a formula reimplemented here.
    """
    snap = hoomd.Snapshot()
    snap.particles.N = 2
    snap.particles.types = ["R"]
    snap.particles.position[:] = [[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]
    snap.configuration.box = [20.0, 20.0, 20.0, 0, 0, 0]
    snap.bonds.types = [bond_type]
    snap.bonds.N = 1
    snap.bonds.typeid[:] = [0]
    snap.bonds.group[:] = [[0, 1]]

    sim = hoomd.Simulation(device=device(), seed=1)
    sim.create_state_from_snapshot(snap)
    force = make_force()
    integ = hoomd.md.Integrator(dt=1e-6)
    integ.forces.append(force)
    sim.operations.integrator = integ
    sim.run(0)
    return float(force.energy)


def _pair_lattice(separation, reps=4, spacing=5.0):
    """``reps**3`` well-separated pairs, each held at ``separation``."""
    pos = []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                base = np.array([i * spacing, j * spacing, k * spacing])
                pos.append(base)
                pos.append(base + np.array([separation, 0.0, 0.0]))
    pos = np.array(pos)
    pos -= pos.mean(axis=0)
    return pos, reps ** 3, reps * spacing + 2.0 * spacing


@pytest.mark.parametrize("device_id,device", _available_devices())
@pytest.mark.parametrize("potential", ["harmonic", "fene"])
def test_metropolis_uses_attached_bond_potential(device_id, device, potential):
    """Bound fraction must follow the energy of the *attached* potential.

    A pair frozen at fixed separation is a two-state Markov chain with per-sweep
    probabilities p_on = k_on n dt exp(-U_b/kT) and p_off = k_off n dt, so its
    stationary bound fraction is p_on / (p_on + p_off). Here k_off is chosen from
    the independently measured U_b so that the correct answer is exactly 1/2. A
    plugin that evaluated some other bond form would land elsewhere: for these
    parameters the FENE bond costs ~1.4 kT more than the harmonic one at the same
    separation, which moves the bound fraction to ~0.2 or ~0.8.
    """
    make_force, _ = BOND_POTENTIALS[potential]
    sep, kT, cs = 1.25, 1.0, 1

    u_bond = _bond_energy(make_force, "R-R", sep, device)
    pos, n_pairs, box_L = _pair_lattice(sep)
    force = make_force()
    sim, cell, thermo, n, dt = _make_sim(
        pos, [0] * len(pos), ["R"], ["R-R"], box_L, kT=kT, seed=11,
        frozen=True, device=device, bond_force=force)

    # The plugin uses the instantaneous kinetic temperature, so read it rather
    # than assuming it equals the requested kT.
    sim.run(0)
    temp = (2.0 * thermo.translational_kinetic_energy
            / thermo.translational_degrees_of_freedom)

    kon = 300.0
    koff = kon * np.exp(-u_bond / temp)     # -> ideal bound fraction of 1/2
    assert koff * cs * dt < 1.0, "off-probability must stay below one"

    updater = GPU_DyBondUpdater(hoomd.trigger.Periodic(cs), cell, thermo)
    updater["R", "R"] = dict(bond=force, bond_typename="R-R",
                             kon=kon, koff=koff,
                             rmin=0.8, rmax=1.3, r0=1.0,
                             metropolisFlag=True, selfAvoidFlag=True)
    spec = updater["R", "R"]
    sim.operations.updaters.append(updater)

    sim.run(3000)                                   # burn in
    samples = []
    for _ in range(2000):
        sim.run(5)
        samples.append(spec.bonds)
    measured = float(np.mean(samples)) / n_pairs

    assert abs(measured - 0.5) < 0.04, (
        f"[{device_id}/{potential}] bound fraction {measured:.3f} != 0.5 "
        f"predicted from the measured bond energy U_b = {u_bond:.4f} "
        f"at kT = {temp:.4f}")


@pytest.mark.parametrize("device_id,device", _available_devices())
@pytest.mark.parametrize("potential", sorted(BOND_POTENTIALS))
def test_bond_potential_selects_matching_plugin_class(device_id, device, potential):
    """Each supported bond form instantiates its own templated plugin class.

    Guards the dispatch in DyBond._attach_hook, which derives the plugin class
    from the HOOMD potential's _cpp_class_name: a renamed evaluator upstream
    would otherwise fail silently or fall back to the wrong energy.
    """
    make_force, expected = BOND_POTENTIALS[potential]
    pos, n_pairs, box_L = _pair_lattice(1.25, reps=2)
    force = make_force()
    sim, cell, thermo, n, dt = _make_sim(
        pos, [0] * len(pos), ["R"], ["R-R"], box_L, kT=1.0, seed=5,
        frozen=True, device=device, bond_force=force)

    updater = GPU_DyBondUpdater(hoomd.trigger.Periodic(1), cell, thermo)
    updater["R", "R"] = dict(bond=force,
                             bond_typename="R-R", kon=300.0, koff=10.0,
                             rmin=0.8, rmax=1.3, r0=1.0)
    spec = updater["R", "R"]
    sim.operations.updaters.append(updater)
    sim.run(1)

    suffix = "GPU" if device_id == "gpu" else ""
    assert type(spec._cpp_obj).__name__ == expected + suffix
