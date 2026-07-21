// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _DYBOND_UPDATER_H_
#define _DYBOND_UPDATER_H_

/*! \file DyBondUpdater.h
 */
#include "DyBond.h"
#include "DyBondBase.h"
#include "hoomd/md/ComputeThermo.h"
#include "hoomd/md/NeighborList.h"
#include <hoomd/Updater.h>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#ifndef __HPICC__
#include <pybind11/pybind11.h>
#endif

//! An updater for dynamically creating and removing bonds between pairs of eligible particle types
namespace hoomd
    {
namespace gpu
    {
typedef unsigned int idx;
template<class T, class Hash = std::hash<T>> using set = std::unordered_set<T, Hash>;

class DyBondUpdater : public Updater
    {
    struct particle_type_pair_hash
        {
        template<class T1, class T2> std::size_t operator()(const std::pair<T1, T2>& pair) const
            {
            return 10 * pair.first + pair.second;
            }
        };

    public:
    //! Constructor
    DyBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<Trigger> trigger,
                  std::shared_ptr<md::NeighborList> nlist,
                  std::shared_ptr<md::ComputeThermo> thermo);

    //! Take one timestep forward
    void update(uint64_t timestep);
    void addDyBond(std::shared_ptr<DyBondBase> dybond);
    void configure();
    //! Returns a list of log quantities this compute calculates

    private:
    std::shared_ptr<md::NeighborList> m_nlist;
    std::shared_ptr<md::ComputeThermo> m_thermo;
    std::vector<std::shared_ptr<DyBondBase>> m_DyBonds;
    std::shared_ptr<BondData> m_bond_data;
    std::unordered_map<std::pair<idx, idx>, idx, particle_type_pair_hash> m_typePair_to_dybond;
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif;
    GPUVector<idx> m_mol_tags;
    GPUVector<idx> m_bond_ranks;
    GPUArray<idx> m_actually_bonded;
    GPUArray<idx> m_actually_unbonded;
    std::vector<idx> m_host_bonded_tags;
    std::vector<idx> m_host_unbonded_tags;
    int DYBOND_NOT_FOUND = -1;

    ///int getDyBondIndex(const idx& particleAType, const idx& particleBType);
    std::vector<idx> getShuffledIndices(idx size, uint64_t timestep);
    int getDyBondIndex(const BondData::members_t& bond_pair, const idx& bond_type);
    void bondMolecules(std::vector<std::vector<idx>>& molMembers,
                       const BondData::members_t& bond_pair);
    idx getMolTag(std::vector<std::vector<idx>>& molMembers, idx particle_tag);
    void findBondsToBreak(const std::shared_ptr<DyBondBase>& dybond,
                          set<idx>& bondTags_toBreak,
                          std::vector<int4>& brokenEvents,
                          const GPUArray<Scalar>& properties,
                          Scalar translational_dof,
                          const uint64_t& timestep);
    void breakBonds(const std::vector<std::vector<int4>>& all_brokenEvents, uint64_t timestep);
    void findBondsToMake(const std::shared_ptr<DyBondBase>& dybond,
                          std::vector<std::pair<idx, idx>>& bondTags_toMake,
                          const GPUArray<Scalar>& properties,
                          Scalar translational_dof,
                          const uint64_t& timestep);
    void makeBonds(const std::vector<std::vector<std::pair<idx, idx>>> all_bondPairs_toMake, uint64_t timestep);
    ///idx getMolTag(idx particle_tag);
    };
} // end gpu namespace
 
//! Export the DyBondUpdater class to python
namespace detail
    {
void export_DyBondUpdaterGPU(pybind11::module& m);
    }
    

    }  // namespace hoomd
#endif // _DYBOND_UPDATER_H_
