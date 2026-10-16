#define USE_DIST_SORTED_NLIST
#define USE_MERSENNE_TWISTER_ENGINE

#include "DyBondUpdater.h"
#ifdef ENABLE_HIP
#include "DyBond.cuh"
#endif
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include <hoomd/RandomNumbers.h>
#include <bits/stdc++.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <random>
#include <string>

#define random() 1.0 * rand() / RAND_MAX

// windows defines a macro min and max
#undef min
#undef max
using std::cout;
using std::endl;

namespace hoomd
    {
namespace gpu
    {
// start the random number generator

/*
    Dynamic Bond Updater creates bonds between two types of particles.
*/

/* \param sysdef: System to dynamically create bonds
    \param nlist : neighbour list to use for finding particles to bond to
    \param thermo: ComputeThermo object to get temperature of the system
    \param group : The group of particles this updater method is to work on
*/
DyBondUpdater::DyBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<Trigger> trigger,
                             std::shared_ptr<md::NeighborList> nlist,
                             std::shared_ptr<md::ComputeThermo> thermo)

    : Updater(sysdef, trigger), m_nlist(nlist), m_thermo(thermo),
      m_mol_tags(m_pdata->getExecConf()), m_bond_ranks(m_pdata->getExecConf()),
      m_actually_bonded(m_pdata->getExecConf()), m_actually_unbonded(m_pdata->getExecConf())
    {
    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();
    rng = std::mt19937_64(sysdef->getSeed());
    unif = std::uniform_real_distribution<double>(0, 1);
    m_set_temperature_properties = GPUArray<Scalar>(1, m_exec_conf);
    assert(m_nlist);
    assert(m_thermo);
    }

/* Rates and the Metropolis factor are evaluated at temp = 2 properties[0] / translational_dof.
   With a temperature callback set, properties[0] is replaced by the kinetic energy for which
   that expression equals callback(timestep), so the CPU and GPU bonding code read that
   temperature unchanged. The Python side supplies a callback that looks up the active
   integrator's thermostat kT at each call, so every change made outside DyBond takes effect at
   the next sweep; exceptions it raises propagate to sim.run.
*/
void DyBondUpdater::setTemperatureCallback(pybind11::object callback)
    {
    m_temperature_callback = callback;
    }

bool min(std::vector<idx> A, std::vector<idx> B)
    {
    auto a = std::min_element(A.begin(), A.end());
    auto b = std::min_element(B.begin(), B.end());
    if (*a < *b)
        {
        return true;
        }
    else
        {
        return false;
        }
    }

void DyBondUpdater::configure()
    {
    const idx num_bonds = m_bond_data->getN();
    const idx num_particles = m_pdata->getN();
    ArrayHandle<idx> h_bond_tags(m_bond_data->getTags(), access_location::host, access_mode::read);
    std::vector<std::vector<idx>> mol_members;
    std::vector<set<idx>> particleTagsToAdd;
    set<idx> allDybondParticleTags;
    set<idx> particleTagsToBond;
    std::vector<set<idx>> bondTagsToBond;
    for (idx i = 0; i < m_DyBonds.size(); i++)
        {
        bondTagsToBond.push_back(set<idx>());
        particleTagsToAdd.push_back(set<idx>());
        }

    for (idx tag = 0; tag < num_particles; tag++)
        {
        m_mol_tags.push_back(-1);
        mol_members.push_back(std::vector<idx>());
        mol_members[tag].push_back(tag);
        idx particleType = m_pdata->getType(tag);
        for (idx i = 0; i < m_DyBonds.size(); i++)
            {
            if (m_DyBonds[i]->isMatch(particleType))
                {
                particleTagsToAdd[i].insert(tag);
                allDybondParticleTags.insert(tag);
                }
            }
        }
    /// Loop over initial bonds to determine molecules (from static bonds)
    /// and dybonds in initial configuration
    for (idx index = 0; index < num_bonds; index++)
        {
        const BondData::members_t bond_pair = m_bond_data->getMembersByIndex(index);
        const idx bond_type = m_bond_data->getTypeByIndex(index);
        int DyBond_index = getDyBondIndex(bond_pair, bond_type);
        if (DyBond_index == DYBOND_NOT_FOUND) /// ie. isStaticBond
            {
            bondMolecules(mol_members, bond_pair);
            }
        else /// ie. isDynamicBond
            {
            particleTagsToBond.insert(bond_pair.tag[0]);
            particleTagsToBond.insert(bond_pair.tag[1]);
            bondTagsToBond[DyBond_index].insert(h_bond_tags.data[index]);
            }
        }

    m_bond_ranks.resize(num_particles);
    {
    ArrayHandle<idx> h_bond_ranks(m_bond_ranks, access_location::host, access_mode::overwrite);
    for (idx i = 0; i < num_particles; i++)
        {
        h_bond_ranks.data[i] = particleTagsToBond.count(i) ? 1 : 0;
        }
    }

    for (idx i = 0; i < m_DyBonds.size(); i++)
        {
        m_DyBonds[i]->addParticles(particleTagsToAdd[i]);
        m_DyBonds[i]->makeBonds(bondTagsToBond[i]);
        m_DyBonds[i]->setGPUStruct();
        }
    /// Sort and save molecule tags,
    ArrayHandle<idx> h_molTags(m_mol_tags, access_location::host, access_mode::read);
    std::sort(mol_members.begin(), mol_members.end(), min);
    for (idx molTag = 0; molTag < mol_members.size(); molTag++)
        {
        for (const auto particleTag : mol_members.at(molTag))
            {
            h_molTags.data[particleTag] = molTag;
            }
        }
    assert(m_mol_tags.size() == num_particles);
    /// for (const auto &db : m_DyBonds)
    ///     {
    ///      db->printContainers();
    ///     }
    }
void DyBondUpdater::update(uint64_t timestep)
    {
    Updater::update(timestep);
    /*
    if (timestep % 1000 == 0)
        {
        for (const auto& db : m_DyBonds)
            {
            db->printContainers();
            }
        }
    */
    m_nlist->compute(timestep);
    m_thermo->compute(timestep);
    Scalar translational_dof = m_thermo->getTranslationalDOF();
    const GPUArray<Scalar>* properties_ptr = &m_thermo->getProperties();
    if (m_temperature_callback && !m_temperature_callback.is_none())
        {
        Scalar temp;
            {
            pybind11::gil_scoped_acquire gil;
            temp = m_temperature_callback(timestep).cast<Scalar>();
            }
        if (temp <= 1e-6 || translational_dof <= 0)
            {
            return;
            }
            {
            ArrayHandle<Scalar> h_set(m_set_temperature_properties,
                                      access_location::host,
                                      access_mode::overwrite);
            h_set.data[0] = temp * translational_dof / Scalar(2.0);
            }
        properties_ptr = &m_set_temperature_properties;
        }
    else
        {
        Scalar temp = m_thermo->getTranslationalTemperature();
        if (temp <= 1e-6)
            {
            return;
            }
        }
    const GPUArray<Scalar>& properties = *properties_ptr;

    std::vector<set<idx>> allBondTagsToBreak(m_DyBonds.size());
    std::vector<std::vector<int4>> allBrokenEvents(m_DyBonds.size());
    std::vector<std::vector<std::pair<idx, idx>>> allBondPairsToMake(m_DyBonds.size());

    std::vector<idx> shuffled_indices = getShuffledIndices(m_DyBonds.size(), timestep);
    for (const auto& i : shuffled_indices)
        {
        findBondsToMake(m_DyBonds[i], allBondPairsToMake[i], properties, translational_dof, timestep);
        findBondsToBreak(m_DyBonds[i], allBondTagsToBreak[i], allBrokenEvents[i], properties, translational_dof, timestep);
        }

    breakBonds(allBrokenEvents, timestep);
    makeBonds(allBondPairsToMake, timestep);
    }

void DyBondUpdater::addDyBond(std::shared_ptr<DyBondBase> dybond)
    {
    m_DyBonds.push_back(dybond);
    }

/// int DyBondUpdater::getDyBondIndex(const idx& particleA_type, const idx& particleB_type)
///     {
///     bool foundDyBond = false;
///     int DyBond_index = -1;
///     for (idx i = 0; i < m_DyBonds.size(); i++)
///         {
///         if (m_DyBonds.at(i)->isMatch(particleA_type, particleB_type))
///             {
///             if (foundDyBond)
///                 {
///                 throw "DyBond " + std::to_string(i) + " and " + std::to_string(DyBond_index)
///                     + " both match bond for particle type pair " +
///                     std::to_string(particleA_type)
///                     + " and " + std::to_string(particleB_type);
///                 }
///             foundDyBond = true;
///             DyBond_index = i;
///             }
///         }
///     /// std::cout << DyBond_index << std::endl;
///     return DyBond_index;
///     }

int DyBondUpdater::getDyBondIndex(const BondData::members_t& bond_pair, const idx& bond_type)
    {
    idx particleA_tag = bond_pair.tag[0];
    idx particleB_tag = bond_pair.tag[1];
    idx particleA_type = m_pdata->getType(particleA_tag);
    idx particleB_type = m_pdata->getType(particleB_tag);
    bool foundDyBond = false;
    int DyBond_index = -1;
    for (idx i = 0; i < m_DyBonds.size(); i++)
        {
        if (m_DyBonds.at(i)->isMatch(bond_type, particleA_type, particleB_type))
            {
            if (foundDyBond)
                {
                throw "DyBond " + std::to_string(i) + " and " + std::to_string(DyBond_index)
                    + " both match bond " + std::to_string(bond_type) + " for particle pair "
                    + std::to_string(particleA_type) + " and " + std::to_string(particleB_type);
                }
            foundDyBond = true;
            DyBond_index = i;
            }
        /// cout << particleA_type << " " << particleB_type << " " << bond_type << " " <<
        /// DyBond_index << endl;
        }
    return DyBond_index;
    }

void DyBondUpdater::bondMolecules(std::vector<std::vector<idx>>& mol_members,
                                  const BondData::members_t& bond_pair)
    {
    idx particleA_tag = bond_pair.tag[0];
    idx particleB_tag = bond_pair.tag[1];
    /// Find molecule associated with partcleI
    idx particleA_moltag = getMolTag(mol_members, particleA_tag);
    idx particleB_moltag = getMolTag(mol_members, particleB_tag);
    /// If bound return
    if (particleA_moltag == particleB_moltag)
        {
        return;
        }
    /// Else let A be the bigger molecule
    if (mol_members.at(particleA_moltag).size() < mol_members.at(particleB_moltag).size())
        {
        auto temp = particleA_moltag;
        particleA_moltag = particleB_moltag;
        particleB_moltag = temp;
        }
    /// Add B to A, move B toBased,
    mol_members[particleA_moltag].insert(mol_members.at(particleA_moltag).end(),
                                         mol_members.at(particleB_moltag).begin(),
                                         mol_members.at(particleB_moltag).end());
    std::swap(mol_members[particleB_moltag], mol_members[mol_members.size() - 1]);
    mol_members.pop_back();
    return;
    }

void DyBondUpdater::findBondsToBreak(const std::shared_ptr<DyBondBase>& dybond,
                                     set<idx>& bondTags_toBreak,
                                     std::vector<int4>& brokenEvents,
                                     const GPUArray<Scalar>& properties,
                                     Scalar translational_dof,
                                     const uint64_t& timestep)
    {
    dybond->findBondsToBreak(bondTags_toBreak, brokenEvents, m_pdata, m_bond_data, m_bond_ranks, properties, translational_dof, timestep);
    }

void DyBondUpdater::breakBonds(const std::vector<std::vector<int4>>& all_brokenEvents, uint64_t timestep)
    {
    std::vector<std::pair<idx, std::pair<idx, idx>>> tags_and_pairs;
    for (idx i = 0; i < m_DyBonds.size(); i++)
        {
        const auto& brokenEvents = all_brokenEvents[i];
        for (const auto& event : brokenEvents)
            {
            tags_and_pairs.push_back({(idx)event.x, {(idx)event.y, (idx)event.z}});
            }
        }

    set<idx> unbondedParticleTags;
    m_host_unbonded_tags.clear();
    ArrayHandle<idx> h_bond_ranks(m_bond_ranks, access_location::host, access_mode::readwrite);
    for (const auto& item : tags_and_pairs)
        {
        idx tag = item.first;
        idx tagA = item.second.first;
        idx tagB = item.second.second;
        unbondedParticleTags.insert(tagA);
        unbondedParticleTags.insert(tagB);
        h_bond_ranks.data[tagA] = 0;
        h_bond_ranks.data[tagB] = 0;
        m_bond_data->removeBondedGroup(tag);
        idx tagA_sorted = std::min(tagA, tagB);
        idx tagB_sorted = std::max(tagA, tagB);
        m_exec_conf->msg->notice(3) << "[DyBond] Bond broken between " << tagA_sorted << " and " << tagB_sorted
                                    << " at timestep " << timestep << "\n";
        }

    for (idx i = 0; i < m_DyBonds.size(); i++)
        {
        const auto& brokenEvents = all_brokenEvents[i];
        set<idx> bondTagsToBreak;
        for (const auto& event : brokenEvents)
            {
            bondTagsToBreak.insert(event.x);
            }
        m_DyBonds[i]->breakBonds(bondTagsToBreak);
        }
    for (const auto& tag : unbondedParticleTags)
        {
        m_host_unbonded_tags.push_back(tag);
        }
    }

void DyBondUpdater::findBondsToMake(const std::shared_ptr<DyBondBase>& dybond,
                                    std::vector<std::pair<idx, idx>>& bondTags_toMake,
                                    const GPUArray<Scalar>& properties,
                                    Scalar translational_dof,
                                    const uint64_t& timestep)
    {
    dybond->findBondsToMake(bondTags_toMake, m_pdata, m_nlist, m_mol_tags, m_bond_ranks, properties, translational_dof, timestep);
    }

std::vector<idx> DyBondUpdater::getShuffledIndices(idx size, uint64_t timestep)
    {
    std::vector<idx> ret;
    for (idx i = 0; i < size; i++)
        {
        ret.push_back(i);
        }
    
    RandomGenerator rng_shuffle(Seed(202, timestep, m_sysdef->getSeed()), Counter(0xffffffff));
    UniformDistribution<Scalar> uniform(0, 1);
    
    for (idx i = ret.size() - 1; i > 0; --i)
        {
        idx j = (idx)(uniform(rng_shuffle) * (i + 1));
        if (j > i) j = i;
        std::swap(ret[i], ret[j]);
        }
    return ret;
    }

void DyBondUpdater::makeBonds(
    const std::vector<std::vector<std::pair<idx, idx>>> all_bondPairs_toMake, uint64_t timestep)
    {
    ArrayHandle<idx> h_bond_ranks(m_bond_ranks, access_location::host, access_mode::readwrite);
    for (idx i = 0; i < m_DyBonds.size(); i++)
        {
        set<idx> bondTagsToAdd;
        const auto& bondPairs_toMake = all_bondPairs_toMake[i];
        for (const auto& pair : bondPairs_toMake)
            {
            idx tag1 = std::min(pair.first, pair.second);
            idx tag2 = std::max(pair.first, pair.second);
            h_bond_ranks.data[tag1] = 1;
            h_bond_ranks.data[tag2] = 1;
            idx bondTag = m_bond_data->addBondedGroup(
                Bond(m_DyBonds[i]->getBondType(), tag1, tag2));
            bondTagsToAdd.insert(bondTag);
            m_exec_conf->msg->notice(3) << "[DyBond] Bond formed between " << tag1 << " and " << tag2
                                        << " at timestep " << timestep << "\n";
            }
        m_DyBonds[i]->makeBonds(bondTagsToAdd);
        }
    }

idx DyBondUpdater::getMolTag(std::vector<std::vector<idx>>& mol_members, idx particle_tag)
    {
    size_t numMols = mol_members.size();
    for (idx i = 0; i < numMols; i++)
        {
        for (const idx& j : mol_members.at(i))
            {
            if (j == particle_tag)
                {
                return i;
                }
            }
        }
    throw "Didn't find " + std::to_string(particle_tag) + "in any molecules";
    }

} // end gpu namespace
 
namespace detail
    {
void export_DyBondUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<gpu::DyBondUpdater, Updater, std::shared_ptr<gpu::DyBondUpdater>>(m, "DyBondUpdaterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<md::NeighborList>,
                            std::shared_ptr<md::ComputeThermo>>())
        .def("addDyBond", &gpu::DyBondUpdater::addDyBond)
        .def("configure", &gpu::DyBondUpdater::configure)
        .def("setTemperatureCallback", &gpu::DyBondUpdater::setTemperatureCallback);
    }
    } // namespace detail
    } // namespace hoomd
