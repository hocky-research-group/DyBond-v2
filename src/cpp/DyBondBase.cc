#include "DyBondBase.h"


#include "hoomd/HOOMDMath.h"

using std::cout;
using std::endl;

namespace hoomd
    {
std::pair<idx, idx> orderedPair(idx typeA, idx typeB)
    {
    if (typeA < typeB)
        {
        return std::pair<idx, idx>(typeA, typeB);
        }
    else
        {
        return std::pair<idx, idx>(typeB, typeA);
        }
    }

long unsigned int DyBondBase::getNumBonds()
    {
    return m_bondTags.size();
    }

void DyBondBase::printContainers(bool verbose)
    {
    cout << "Bond Tags(" << m_bondTags.size() << "):" << endl;
    if (verbose)
        {
        for (idx i = 0; i < m_bondTags.size(); i++)
            {
            cout << m_bondTags[i] << ", ";
            }
        cout << endl;
        cout << endl;
        }    }

DyBondBase::DyBondBase(std::shared_ptr<SystemDefinition> sysdef,
                       idx type,
                       Scalar dt,
                       idx period,
                       Scalar kon,
                       Scalar koff,
                       Scalar rmin,
                       Scalar rmax)
    : m_sysdef(sysdef), m_bond_type(type), m_dt(dt), m_period(period), m_kon(kon), m_koff(koff),
      m_rmin(rmin), m_rmax(rmax), m_instance_number(instance_counter++),
      m_exec_conf(sysdef->getParticleData()->getExecConf()), m_seed(sysdef->getSeed())
    {
    m_bondTags = GPUVector<idx>(m_exec_conf);
    m_dybondParticleTags = GPUVector<idx>(m_exec_conf);
    }

void DyBondBase::addTypePair(idx typeA, idx typeB)
    {
    m_particleType_pairs.insert(orderedPair(typeA, typeB));
    m_particleTypes.insert(typeA);
    m_particleTypes.insert(typeB);
m_gpu_params_dirty = true;
    /// m_particle_types.push_back(orderedPair(typeA, typeB));
    }
void DyBondBase::setTempParams(Scalar Tmelt, Scalar alpha, Scalar kon_melt, Scalar koff_melt)
    {
    m_Tmelt = Tmelt;
    m_alpha = alpha;
    m_kon_melt = kon_melt;
    m_koff_melt = koff_melt;
    m_flag_temperature = true;
    m_gpu_params_dirty = true;
    }
idx DyBondBase::getBondType()
    {
    return m_bond_type;
    }
void DyBondBase::setMetropolisFlag(bool flag)
    {
    m_flag_metropolis = flag;
    m_gpu_params_dirty = true;
    }

void DyBondBase::setSelfAvoidingFlag(bool flag)
    {
    m_flag_self_avoiding = flag;
    m_gpu_params_dirty = true;
    }
void DyBondBase::setForceParams(Scalar a)
    {
    m_flag_force = true;
    m_force_a = a;
    m_gpu_params_dirty = true;
    }

template<class T> set<T> DyBondBase::eraseFromGPUVector(set<T> toErase, GPUVector<T>& fromVector)
    {
    set<T> erased;
    if (toErase.empty())
        {
        return erased;
        }
    GPUVector<T> newArray(fromVector.size(), m_sysdef->getParticleData()->getExecConf());
    size_t new_size = 0;

    {
    ArrayHandle<T> h_fromVector(fromVector, access_location::host, access_mode::read);
    ArrayHandle<T> h_newArray(newArray, access_location::host, access_mode::overwrite);
    for (size_t n = 0; n < fromVector.size(); ++n)
        {
        T item = h_fromVector.data[n];
        if ((bool)toErase.count(item))
            {
            erased.insert(item);
            }
        else
            {
            h_newArray.data[new_size++] = item;
            }
        }
    }
    newArray.resize(new_size);
    fromVector.swap(newArray);
    return erased;
    }
template<class T> void DyBondBase::addToGPUVector(set<T> toAdd, GPUVector<T>& toVector)
    {
    if (toAdd.empty()) return;
    size_t old_size = toVector.size();
    toVector.resize(old_size + toAdd.size());
    {
    ArrayHandle<T> h_toVector(toVector, access_location::host, access_mode::readwrite);
    size_t n = old_size;
    for (T add : toAdd)
        {
        h_toVector.data[n++] = add;
        }
    }
    }

bool DyBondBase::isMatch(const idx& bond_type, const idx& particleA_type, const idx& particleB_type)
    {
    if (bond_type != m_bond_type)
        {
        return false;
        }

    std::pair<idx, idx> pair = orderedPair(particleA_type, particleB_type);
    /// cout << m_particleType_pairs.count(pair) << endl;
    return (bool)m_particleType_pairs.count(pair);
    }
bool DyBondBase::isMatch(const idx& particleA_type, const idx& particleB_type)
    {
    std::pair<idx, idx> pair = orderedPair(particleA_type, particleB_type);
    return (bool)m_particleType_pairs.count(pair);
    }
bool DyBondBase::isMatch(const idx& particleA_type)
    {
    return (bool)m_particleTypes.count(particleA_type);
    }
void DyBondBase::addParticles(const set<idx>& particleTags)
    {
    addToGPUVector<idx>(particleTags, m_dybondParticleTags);
    }

void DyBondBase::makeBonds(const set<idx>& bondTagsToMake)
    {
    addToGPUVector(bondTagsToMake, m_bondTags);
    }

void DyBondBase::breakBonds(const set<idx>& bondTagsToBreak)
    {
    eraseFromGPUVector(bondTagsToBreak, m_bondTags);
    }
void DyBondBase::setNearestNeighborFlag(bool flag)
    {
    m_flag_nearest_neighbor = flag;
    m_gpu_params_dirty = true;
    }

void DyBondBase::setR0(Scalar r0)
    {
    m_r0 = r0;
    m_gpu_params_dirty = true;
    }

namespace detail
    {
void export_DyBondBase(pybind11::module& m)
    {
    pybind11::class_<DyBondBase, std::shared_ptr<DyBondBase>>(m, "DyBondBase")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>())
        .def("addTypePair", &DyBondBase::addTypePair)
        .def("setTempParams", &DyBondBase::setTempParams)
        .def("setMetropolisFlag", &DyBondBase::setMetropolisFlag)
        .def("setForceParams", &DyBondBase::setForceParams)
        .def("getNumBonds", &DyBondBase::getNumBonds)
        .def("setSelfAvoidingFlag", &DyBondBase::setSelfAvoidingFlag)
        .def("setNearestNeighborFlag", &DyBondBase::setNearestNeighborFlag)
        .def("setR0", &DyBondBase::setR0);
    }
    } // namespace detail

idx DyBondBase::instance_counter = 0;

    } // namespace hoomd

