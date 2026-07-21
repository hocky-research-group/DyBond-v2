#ifndef _DYBOND_H_
#define _DYBOND_H_

#include "DyBondBase.h"
#include "hoomd/GPUVector.h"
#include "hoomd/md/EvaluatorBondFENE.h"
#include "hoomd/md/EvaluatorBondHarmonic.h"
#include "hoomd/md/EvaluatorBondTether.h"
#include <hoomd/Updater.h>
#include <hoomd/md/NeighborList.h>
#ifdef ENABLE_HIP
#include "DyBond.cuh"
#endif
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

template<class bond_class> class DyBond : public DyBondBase
    {
    public:
    typedef typename bond_class::param_type Params;
    DyBond(std::shared_ptr<SystemDefinition> sysdef,
           idx type,
           Scalar dt,
           idx period,
           pybind11::dict bond_params,
           Scalar kon,
           Scalar koff,
           Scalar rmin,
           Scalar rmax)
        : DyBondBase(sysdef, type, dt, period, kon, koff, rmin, rmax), m_bond_params(bond_params)
        {
        }
    void findBondsToBreak(set<idx>& toRemoveTag,
                          std::vector<int4>& brokenEvents,
                          const std::shared_ptr<hoomd::ParticleData>& pdata,
                          const std::shared_ptr<hoomd::BondData>& bond_data,
                          GPUVector<idx>& m_bond_ranks,
                          const GPUArray<Scalar>& properties,
                          Scalar translational_dof,
                          const uint64_t& timestep);
    void findBondsToMake(std::vector<std::pair<idx, idx>>& toAddPairs,
                         const std::shared_ptr<hoomd::ParticleData>& pdata,
                         const std::shared_ptr<hoomd::md::NeighborList>& nlist,
                         const GPUVector<idx>& mol_tags,
                         GPUVector<idx>& m_bond_ranks,
                         const GPUArray<Scalar>& properties,
                         Scalar translational_dof,
                         const uint64_t& timestep);

    protected:
    Params m_bond_params;
    };

typedef DyBond<md::EvaluatorBondFENE> DyBondFENE;
typedef DyBond<md::EvaluatorBondHarmonic> DyBondHarmonic;
typedef DyBond<md::EvaluatorBondTether> DyBondTether;

namespace detail
    {
void export_DyBond(pybind11::module& m);
    }

#ifdef ENABLE_HIP
#include "DyBond.cuh"

template<class bond_class> class DyBondGPU : public DyBondBase
    {
    public:
    typedef typename bond_class::param_type BondParams;
    DyBondGPU(std::shared_ptr<SystemDefinition> sysdef,
              idx type,
              Scalar dt,
              idx period,
              pybind11::dict bond_params,
              Scalar kon,
              Scalar koff,
              Scalar rmin,
              Scalar rmax)
        : DyBondBase(sysdef, type, dt, period, kon, koff, rmin, rmax), m_bond_params(bond_params)
        {
        m_breaks = GPUArray<bool>(m_exec_conf);
        m_makeTag1 = GPUArray<int>(m_exec_conf);
        m_makeTag2 = GPUArray<int>(m_exec_conf);
        m_bondsToBreak = GPUArray<int4>(m_exec_conf);
        m_numBondsToBreak = GPUArray<idx>(m_exec_conf);
        m_bondsToMake = GPUArray<int2>(m_exec_conf);
        m_numBondsToMake = GPUArray<idx>(m_exec_conf);
        m_cand_partner = GPUArray<idx>(m_exec_conf);
        m_cand_dist = GPUArray<Scalar>(m_exec_conf);
        m_prop_from = GPUArray<idx>(m_exec_conf);
        m_prop_to = GPUArray<idx>(m_exec_conf);
        m_n_prop = GPUArray<idx>(m_exec_conf);
        m_dybond_params.d_typeAs = nullptr;
        m_dybond_params.d_typeBs = nullptr;
        d_dybond_params = nullptr;
        hipStreamCreate(&m_stream);
        hipHostMalloc(&m_numBondsToBreak_host, sizeof(idx), hipHostMallocDefault);
        hipHostMalloc(&m_numBondsToMake_host, sizeof(idx), hipHostMallocDefault);
        }
    virtual ~DyBondGPU()
        {
        hipHostFree(m_numBondsToBreak_host);
        hipHostFree(m_numBondsToMake_host);
        hipStreamDestroy(m_stream);
        }
    void findBondsToBreak(set<idx>& toRemoveTag,
                          std::vector<int4>& brokenEvents,
                          const std::shared_ptr<hoomd::ParticleData>& pdata,
                          const std::shared_ptr<hoomd::BondData>& bond_data,
                          GPUVector<idx>& m_bond_ranks,
                          const GPUArray<Scalar>& properties,
                          Scalar translational_dof,
                          const uint64_t& timestep);
    void findBondsToMake(std::vector<std::pair<idx, idx>>& toAddPairs,
                         const std::shared_ptr<hoomd::ParticleData>& pdata,
                         const std::shared_ptr<hoomd::md::NeighborList>& nlist,
                         const GPUVector<idx>& mol_tags,
                         GPUVector<idx>& m_bond_ranks,
                         const GPUArray<Scalar>& properties,
                         Scalar translational_dof,
                         const uint64_t& timestep);
    void setGPUStruct();
    virtual hipStream_t getStream() const override { return m_stream; }

    protected:
    BondParams m_bond_params;
    GPUArray<bool> m_breaks;
    GPUArray<int> m_makeTag1;
    GPUArray<int> m_makeTag2;
    GPUArray<int4> m_bondsToBreak;
    GPUArray<idx> m_numBondsToBreak;
    GPUArray<int2> m_bondsToMake;
    GPUArray<idx> m_numBondsToMake;

    GPUArray<idx> m_cand_partner;
    GPUArray<Scalar> m_cand_dist;
    GPUArray<idx> m_prop_from;
    GPUArray<idx> m_prop_to;
    GPUArray<idx> m_n_prop;

    DyBondParams<bond_class> m_dybond_params;
    DyBondParams<bond_class>* d_dybond_params;
    hipStream_t m_stream;
    idx* m_numBondsToBreak_host;
    idx* m_numBondsToMake_host;
    };

typedef DyBondGPU<md::EvaluatorBondFENE> DyBondFENEGPU;
typedef DyBondGPU<md::EvaluatorBondHarmonic> DyBondHarmonicGPU;
typedef DyBondGPU<md::EvaluatorBondTether> DyBondTetherGPU;

namespace detail
    {
void export_DyBondGPU(pybind11::module& m);
    }

#endif

    } // namespace hoomd
#endif
