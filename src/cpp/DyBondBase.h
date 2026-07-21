#ifndef _DYBONDBASE_H_
#define _DYBONDBASE_H_

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#else
typedef void* hipStream_t;
#endif
#include "hoomd/GPUArray.h"
#include "hoomd/GPUVector.h"
#include <hoomd/Updater.h>
#include "hoomd/md/NeighborList.h"
#include <map>
#include <random>
#include <vector>

#ifndef __HPICC__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace hoomd
    {
typedef unsigned int idx;
template<class T, class Hash = std::hash<T>> using set = std::unordered_set<T, Hash>;
class DyBondBase
    {
    public:
    DyBondBase(std::shared_ptr<SystemDefinition> sysdef,
               idx type,
               Scalar dt,
               idx period,
               Scalar kon,
               Scalar koff,
               Scalar rmin,
               Scalar rmax);
    void addTypePair(idx typeA, idx typeB);
    void setTempParams(Scalar Tmelt, Scalar alpha, Scalar kon_melt, Scalar koff_melt);
    void setMetropolisFlag(bool flag);
    void setSelfAvoidingFlag(bool flag);
    void setForceParams(Scalar a);
    void setNearestNeighborFlag(bool flag);
    void setR0(Scalar r0);
    long unsigned int getNumBonds();
    virtual void setGPUStruct() {};

    void printContainers(bool verbose = false);
    idx getBondType();
    bool isMatch(const idx& particleA_type, const idx& particleB_type, const idx& bond_type);
    bool isMatch(const idx& particleA_type, const idx& particleB_type);
    bool isMatch(const idx& particleA_type);
    void addParticles(const set<idx>& particleTags);
    void makeBonds(const set<idx>& bondTagsToMake);
    void breakBonds(const set<idx>& bondTagsToBreak);
    virtual void findBondsToBreak(set<idx>& toRemoveTag,
                                  std::vector<int4>& brokenEvents,
                                  const std::shared_ptr<hoomd::ParticleData>& pdata,
                                  const std::shared_ptr<hoomd::BondData>& bond_data,
                                  GPUVector<idx>& m_bond_ranks,
                                  const GPUArray<Scalar>& properties,
                                  Scalar translational_dof,
                                  const uint64_t& timestep) {};
    virtual void findBondsToMake(std::vector<std::pair<idx, idx>>& toAddPairs,
                                 const std::shared_ptr<hoomd::ParticleData>& pdata,
                                 const std::shared_ptr<hoomd::md::NeighborList>& nlist,
                                 const GPUVector<idx>& mol_tags,
                                 GPUVector<idx>& m_bond_ranks,
                                 const GPUArray<Scalar>& properties,
                                 Scalar translational_dof,
                                 const uint64_t& timestep) {};
    virtual hipStream_t getStream() const { return 0; }

    protected:
    struct particle_type_pair_hash
        {
        template<class T1, class T2> std::size_t operator()(const std::pair<T1, T2>& pair) const
            {
            return 10 * pair.first + pair.second;
            }
        };

    template<class T> set<T> eraseFromGPUVector(set<T> toErase, GPUVector<T>& fromVector);
    template<class T> void addToGPUVector(set<T> toAdd, GPUVector<T>& fromVector);

    static idx instance_counter;
    const idx BONDING_RNG_IDENTIFIER = 201;
    const idx UNBONDING_RNG_IDENTIFIER = 200;
    const idx NOT_BONDABLE = 0;
    const idx m_instance_number;

    std::shared_ptr<SystemDefinition> m_sysdef;
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;

    idx m_bond_type;

    Scalar m_dt;
    Scalar m_seed;
    idx m_period;

    Scalar m_kon, m_koff;
    Scalar m_rmin, m_rmax;
    bool m_flag_metropolis = true;

    bool m_flag_force = false;
    Scalar m_force_a;

    bool m_flag_self_avoiding = false;
    bool m_flag_nearest_neighbor = true;
    Scalar m_r0 = 1.0;

    bool m_flag_temperature = false;
    Scalar m_Tmelt;
    Scalar m_alpha;
    Scalar m_kon_melt, m_koff_melt;

    set<std::pair<idx, idx>, particle_type_pair_hash> m_particleType_pairs;
    set<idx> m_particleTypes;
    GPUVector<idx> m_bondTags;
    GPUVector<idx> m_dybondParticleTags;
    bool m_gpu_params_dirty = true;
    };



namespace detail
    {
void export_DyBondBase(pybind11::module& m);
    }
    } // namespace hoomd
#endif
