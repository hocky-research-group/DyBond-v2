# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd.dybond import _dybond
import hoomd
from hoomd import operation
from hoomd.md import _md
from hoomd.logging import log, Loggable, modify_namespace;

__all__ = ['DyBond', 'DyBondUpdater']

class DyBond(metaclass = Loggable):
    def __init__(self,name,bond,kon,koff,rmin,rmax,nameA,nameB,selfAvoidFlag,
                 metropolisFlag = True,forceFlag = False):
        self.name = name
        self.bond = bond
        self.kon, self.koff = float(kon), float(koff)
        self.rmin, self.rmax = float(rmin), float(rmax)
        self.names = []
        self[nameA,nameB]
        self.selfAvoidFlag = bool(selfAvoidFlag)
        self.metropolisFlag = metropolisFlag
        self.forceFlag = forceFlag
        self.temperatureFlag = False
        self.r0 = self.get_r0()

    def get_r0(self):
        """Determine equilibrium/target distance r0 from the HOOMD bond object.

        Extracts 'r0' directly for Harmonic / FENE bonds, or the average of
        l_c1 and l_c0 for Tether bonds. Raises ValueError if r0 cannot be resolved.
        """
        params = None
        bond_name = self.name
        if hasattr(self, '_simulation') and self._simulation is not None and isinstance(bond_name, int):
            bond_name = self._simulation.state.bond_types[bond_name]

        if hasattr(self.bond, 'params'):
            try:
                params = self.bond.params[bond_name]
            except (KeyError, IndexError, TypeError):
                if isinstance(bond_name, int) and hasattr(self.bond.params, 'keys'):
                    keys = list(self.bond.params.keys())
                    if 0 <= bond_name < len(keys):
                        params = self.bond.params[keys[bond_name]]
                elif hasattr(self.bond.params, 'keys') and len(self.bond.params.keys()) == 1:
                    params = self.bond.params[list(self.bond.params.keys())[0]]

        if params is None and hasattr(self.bond, '_cpp_obj') and hasattr(self.bond._cpp_obj, 'getParams'):
            try:
                cpp_params = self.bond._cpp_obj.getParams(bond_name)
                if hasattr(cpp_params, 'asDict'):
                    params = cpp_params.asDict()
                elif hasattr(cpp_params, 'r_0'):
                    return float(cpp_params.r_0)
                elif hasattr(cpp_params, 'l_c0') and hasattr(cpp_params, 'l_c1'):
                    return 0.5 * float(cpp_params.l_c1 + cpp_params.l_c0)
            except Exception:
                params = None

        if params is not None:
            if 'r0' in params:
                return float(params['r0'])
            if 'l_c0' in params and 'l_c1' in params:
                return 0.5 * float(params['l_c1'] + params['l_c0'])

        raise ValueError(
            f"Cannot determine r0 for bond type '{bond_name}' from bond object {self.bond}. "
            "Bond parameters must specify 'r0' (e.g. Harmonic, FENEWCA) or 'l_c0' and 'l_c1' (Tether)."
        )

    def add_particles(self, A, B):
        A = str(A) if isinstance(A,int) else A
        B = str(B) if isinstance(B,int) else B
        if isinstance(A,str) and isinstance(B,str):
            self.names.append((A,B))
            return
        if (not isinstance(A,str)) and (not isinstance(B,str)):
            for a in A:
                for b in B:
                    self.names.append((a,b))
            return
        if isinstance(A,str):
            A,B = B,A
        for a in A:
            self.names.append((a,B))

    def setForceParams(self,a):
        self.forceFlag = True 
        self.a = a
    def setTemperatureParams(self,Tmelt, alpha, kon_melt, koff_melt):
        self.temperatureFlag = True
        self.Tmelt = Tmelt
        self.alpha = alpha
        self.kon_melt = kon_melt
        self.koff_melt = koff_melt
    def setSelfAvoidingFlag(self,flag):
        self.selfAvoidFlag = flag
    def setMetropolisFlag(self,flag):
        self.metropolisFlag = flag

    @log
    def bonds(self):
        if hasattr(self, '_cpp_obj'):
            return self._cpp_obj.getNumBonds()
        else:
            return -1

    def __getitem__(self,item):
        A,B = item
        self.add_particles(A,B)

    def _attach(self,sim):
        self._simulation = sim
        self._attach_hook()

    def _attach_hook(self):
        if self.forceFlag and self.temperatureFlag:
            raise Exception("Temperature flag and force flag cannot be set together")
        self.bond._attach(self._simulation)
        if isinstance(self.name, int):
            self.name = self._simulation.state.bond_types[self.name]
        bond_class = self.bond._cpp_class_name.replace("PotentialBond","")
        bond_params = self.bond._cpp_obj.getParams(self.name)
        bond_type = self._simulation.state.bond_types.index(self.name)
        self._convert_names(self._simulation.state.particle_types)
        if isinstance(self._simulation.device,hoomd.device.CPU):
            cls = getattr(_dybond,"DyBond"+bond_class)
        elif isinstance(self._simulation.device,hoomd.device.GPU):
            cls = getattr(_dybond,"DyBond"+bond_class+"GPU")
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            bond_type, self.dt, self.period, bond_params,
                            float(self.kon), float(self.koff), float(self.rmin), float(self.rmax))
        self._cpp_obj.setMetropolisFlag(self.metropolisFlag)
        self._cpp_obj.setSelfAvoidingFlag(self.selfAvoidFlag)
        self._cpp_obj.setNearestNeighborFlag(self.nearestNeighbor)
        self._cpp_obj.setR0(self.r0)
        if self.forceFlag:
            self._cpp_obj.setForceParams(self.a)
        if self.temperatureFlag:
            self._cpp_obj.setTempParams(self.Tmelt, self.alpha, 
                                        self.kon_melt, self.koff_melt)
        for (typeA, typeB) in self.types:
            self._cpp_obj.addTypePair(typeA, typeB)
    
    def _convert_names(self, particle_names):
        self.types = []
        for (A,B) in self.names:
            typeA = particle_names.index(A)
            typeB = particle_names.index(B)
            self.types.append((typeA,typeB))
    def _set_time(self,dt,period,nearestNeighbor=True):
        self.dt = dt
        self.period = period
        self.nearestNeighbor = nearestNeighbor

class DyBondUpdater(operation.Updater):
    def __init__(self, trigger: hoomd.trigger.Trigger, nlist, thermo, nearestNeighbor=True):
        '''
        Creates an instance of the dybond updater.
        NOTE: Not tested for running using multiple GPU's using MPI
        '''
        # initialize base class
        if isinstance(trigger, hoomd.trigger.Periodic):
            self.period = trigger.period
        elif isinstance(trigger, hoomd.trigger.And) or isinstance(trigger, hoomd.trigger.And): 
            for t in trigger.triggers:
                if hasattr(t,"period"):
                    self.period = t.period
                    break
        else:
            self.period = trigger
        super().__init__(trigger)
        self.nlist = nlist
        self.thermo = thermo
        self.nearestNeighbor = nearestNeighbor
        self.specs = {}

    @property
    def dybonds(self):
        return list(dict.fromkeys(self.specs.values()))

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Key must be a tuple of two particle types, e.g. ('A', 'B')")
        if isinstance(value, dict):
            required = ['bond', 'kon', 'koff', 'rmin', 'rmax']
            missing = [k for k in required if k not in value]
            if 'selfAvoidFlag' not in value and 'selfAvoidingFlag' not in value:
                missing.append('selfAvoidFlag')
            bond_name = value.get('bond_typename', value.get('name'))
            if bond_name is None:
                bond_typeid = value.get('bond_typeid')
                if bond_typeid is not None:
                    if hasattr(self, '_simulation') and self._simulation is not None:
                        bond_name = self._simulation.state.bond_types[int(bond_typeid)]
                    else:
                        bond_name = int(bond_typeid)
            if bond_name is None:
                missing.append('bond_typename')
            if missing:
                raise KeyError(f"Missing required parameter(s) for dynamic bond: {', '.join(missing)}")
            bond = value['bond']
            kon = float(value['kon'])
            koff = float(value['koff'])
            rmin = float(value['rmin'])
            rmax = float(value['rmax'])
            metropolisFlag = value.get('metropolisFlag', True)
            forceFlag = 'force_a' in value
            selfAvoidFlag = bool(value['selfAvoidFlag'] if 'selfAvoidFlag' in value else value['selfAvoidingFlag'])
            spec = DyBond(
                name=bond_name, bond=bond, kon=kon, koff=koff,
                rmin=rmin, rmax=rmax, nameA=key[0], nameB=key[1],
                selfAvoidFlag=selfAvoidFlag,
                metropolisFlag=metropolisFlag, forceFlag=forceFlag
            )
            if 'force_a' in value:
                spec.setForceParams(value['force_a'])
            if 'Tmelt' in value:
                spec.setTemperatureParams(
                    value['Tmelt'], value.get('alpha', 200.0),
                    value.get('kon_melt', 0.0), value.get('koff_melt', 0.0)
                )
            self.specs[key] = spec
        else:
            self.specs[key] = value

    def __getitem__(self, key):
        return self.specs[key]

    def append(self, spec):
        for (a, b) in spec.names:
            self.specs[(a, b)] = spec

    def _attach_hook(self):
        # initialize the reflected c++ class
        self.nlist._attach(self._simulation)
        self.thermo._attach(self._simulation)
        if isinstance(self._simulation.device,hoomd.device.CPU):
            cls = _dybond.DyBondUpdaterGPU
            self.nlist._cpp_obj.setStorageMode(
                                _md.NeighborList.storageMode.half)
        elif isinstance(self._simulation.device,hoomd.device.GPU):
            cls = _dybond.DyBondUpdaterGPU
            self.nlist._cpp_obj.setStorageMode(
                                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                                                self.trigger,
                                                self.nlist._cpp_obj, 
                                                self.thermo._cpp_obj);
        for bond in self.dybonds:
            bond._set_time(period = self.period,dt = self._simulation.operations.integrator.dt,nearestNeighbor=self.nearestNeighbor)
            bond._attach(self._simulation)
            self._cpp_obj.addDyBond(bond._cpp_obj)
            for (typeA, typeB) in bond.names:
                try:
                    current_rcut = self.nlist.r_cut.get((typeA, typeB), 0.0)
                    if current_rcut is None or current_rcut is False:
                        current_rcut = 0.0
                except Exception:
                    current_rcut = 0.0
                self.nlist.r_cut[typeA, typeB] = max(float(current_rcut), float(bond.rmax))
        self._cpp_obj.configure()
