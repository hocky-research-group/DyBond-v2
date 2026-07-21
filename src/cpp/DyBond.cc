#include "DyBond.h"
#include <hoomd/HOOMDMath.h>
#include <hoomd/RandomNumbers.h>
#ifdef ENABLE_HIP
#include "DyBond.cuh"
#include "hip/hip_runtime.h"
using namespace hoomd::kernel;
#endif

using std::cout;
using std::endl;

namespace hoomd
    {
template<class bond_class>
void DyBond<bond_class>::findBondsToBreak(set<idx>& toRemoveTag,
                                          std::vector<int4>& brokenEvents,
                                          const std::shared_ptr<hoomd::ParticleData>& pdata,
                                          const std::shared_ptr<hoomd::BondData>& bond_data,
                                          GPUVector<idx>& m_bond_ranks,
                                          const GPUArray<Scalar>& properties,
                                          Scalar translational_dof,
                                          const uint64_t& timestep)
    {
    ArrayHandle<Scalar> h_properties(properties, access_location::host, access_mode::read);
    Scalar temp = 0.0;
    if (translational_dof > 0.0)
        {
        temp = (h_properties.data[0] * 2.0) / translational_dof;
        }
    if (temp <= 1e-6)
        {
        return;
        }
    ArrayHandle<idx> h_bondIndices(bond_data->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<BondData::members_t> h_bondPairs(bond_data->getMembersArray(),
                                                 access_location::host,
                                                 access_mode::read);
    ArrayHandle<idx> h_particleIndices(pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_particlePositions(pdata->getPositions(),
                                             access_location::host,
                                             access_mode::read);
    ArrayHandle<idx> h_bond_tags(m_bondTags, access_location::host, access_mode::read);
    BoxDim box = pdata->getBox();
    ArrayHandle<idx> h_bondRanks(m_bond_ranks, access_location::host, access_mode::readwrite);

    Scalar koff = m_koff;
    if (m_flag_temperature)
        {
        Scalar g = 0.5 * (tanh(m_alpha * (temp - m_Tmelt)) + 1.0);
        koff = koff * (1 - g) + m_koff_melt * g;
        }
    Scalar Poff = koff * m_period * m_dt;
    for (idx i = 0; i < m_bondTags.size(); i++)
        {
        idx bond_tag = h_bond_tags.data[i];
        Scalar P = Poff;
        idx bond_index = h_bondIndices.data[bond_tag];
        BondData::members_t bond_pair = h_bondPairs.data[bond_index];
        idx tagA = bond_pair.tag[0];
        idx tagB = bond_pair.tag[1];
        if (m_flag_force)
            {
            idx a_idx = h_particleIndices.data[tagA];
            idx b_idx = h_particleIndices.data[tagB];

            Scalar4 a_pos = h_particlePositions.data[a_idx];
            Scalar4 b_pos = h_particlePositions.data[b_idx];
            Scalar3 delta_r = box.minImage(
                make_scalar3(a_pos.x - b_pos.x, a_pos.y - b_pos.y, a_pos.z - b_pos.z));
            Scalar dist_sq = delta_r.x * delta_r.x + delta_r.y * delta_r.y + delta_r.z * delta_r.z;

            Scalar force_divr = Scalar(0.0);
            Scalar bond_eng = Scalar(0.0);
            bond_class eval(dist_sq, m_bond_params);
            eval.evalForceAndEnergy(force_divr, bond_eng);
            /// Scalar force = force_divr * sqrt(dist_sq);
            P = P * exp(m_force_a * sqrt(bond_eng / temp));
            }
        RandomGenerator rng(Seed(UNBONDING_RNG_IDENTIFIER + m_instance_number, timestep, m_seed),
                            Counter(tagA < tagB ? tagA : tagB, tagA < tagB ? tagB : tagA));
        UniformDistribution<Scalar> uniform(0, 1);
        Scalar p = uniform(rng);
        if (p < P)
            {
            toRemoveTag.insert(bond_tag);
            h_bondRanks.data[tagA] = 0;
            h_bondRanks.data[tagB] = 0;
            brokenEvents.push_back(make_int4(bond_tag, tagA, tagB, 0));
            }
        }
    }

template<class bond_class>
void DyBond<bond_class>::findBondsToMake(std::vector<std::pair<idx, idx>>& toAddPairs,
                                         const std::shared_ptr<hoomd::ParticleData>& pdata,
                                         const std::shared_ptr<hoomd::md::NeighborList>& nlist,
                                         const GPUVector<idx>& mol_tags,
                                         GPUVector<idx>& bond_ranks,
                                         const GPUArray<Scalar>& properties,
                                         Scalar translational_dof,
                                         const uint64_t& timestep)
    {
    ArrayHandle<Scalar> h_properties(properties, access_location::host, access_mode::read);
    Scalar temp = 0.0;
    if (translational_dof > 0.0)
        {
        temp = (h_properties.data[0] * 2.0) / translational_dof;
        }
    if (temp <= 1e-6)
        {
        return;
        }
    ArrayHandle<idx> h_particleIndices(pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<idx> h_particleTags(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_particlePositions(pdata->getPositions(),
                                             access_location::host,
                                             access_mode::read);
    ArrayHandle<idx> h_numNeighbors(nlist->getNNeighArray(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<idx> h_neighbors(nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<size_t> h_neighborOffset(nlist->getHeadList(),
                                         access_location::host,
                                         access_mode::read);
    bool full_storage = nlist->getStorageMode() == md::NeighborList::storageMode::full;
    BoxDim box = pdata->getBox();

    ArrayHandle<idx> h_molTags(mol_tags, access_location::host, access_mode::read);
    ArrayHandle<idx> h_bondRanks(bond_ranks, access_location::host, access_mode::readwrite);
    ArrayHandle<idx> h_dybondParticleTags(m_dybondParticleTags,
                                          access_location::host,
                                          access_mode::read);
    Scalar kon = m_kon;
    if (m_flag_temperature)
        {
        Scalar g = 0.5 * (tanh(m_alpha * (temp - m_Tmelt)) + 1.0);
        kon = kon * (1 - g) + m_kon_melt * g;
        }
    Scalar Pon = kon * m_dt * m_period;

    // Shuffle the particles list using HOOMD's RandomGenerator (Philox) to prevent tag/ordering bias
    std::vector<idx> shuffled_particles(m_dybondParticleTags.size());
    for (idx i = 0; i < m_dybondParticleTags.size(); i++)
        {
        shuffled_particles[i] = h_dybondParticleTags.data[i];
        }
    RandomGenerator rng_shuffle(Seed(BONDING_RNG_IDENTIFIER + m_instance_number, timestep, m_seed), Counter(0xffffffff));
    UniformDistribution<Scalar> uniform_shuffle(0, 1);
    for (idx i = shuffled_particles.size() - 1; i > 0; --i)
        {
        idx j = (idx)(uniform_shuffle(rng_shuffle) * (i + 1));
        if (j > i) j = i;
        std::swap(shuffled_particles[i], shuffled_particles[j]);
        }

    idx N = pdata->getN();
    std::vector<idx> cand_partner(N, 0xffffffff);
    std::vector<Scalar> cand_dist(N, 0.0);
    std::vector<Scalar> cand_rsq(N, 0.0);

    // Phase 1: Candidate Search
    for (idx i = 0; i < shuffled_particles.size(); i++)
        {
        idx tagA = shuffled_particles[i];
        if (h_bondRanks.data[tagA] > 0)
            {
            continue;
            }
        idx a_idx = h_particleIndices.data[tagA];
        Scalar4 a_pos = h_particlePositions.data[a_idx];
        idx n_neigh = h_numNeighbors.data[a_idx];
        size_t my_offset = h_neighborOffset.data[a_idx];
        idx best_tagB = 0xffffffff;
        Scalar best_rsq = 0.0;
        Scalar best_dist = 0.0;

        if (m_flag_nearest_neighbor)
            {
            Scalar best_dd = 1e30;
            for (idx j = 0; j < n_neigh; j++)
                {
                idx b_idx = h_neighbors.data[my_offset + j];
                idx tagB = h_particleTags.data[b_idx];
                if (h_bondRanks.data[tagB] > 0)
                    {
                    continue;
                    }
                if (h_molTags.data[tagA] == h_molTags.data[tagB])
                    {
                    continue;
                    }
                Scalar4 b_pos = h_particlePositions.data[b_idx];
                unsigned int typeB = __scalar_as_int(b_pos.w);
                if (m_particleTypes.find(typeB) == m_particleTypes.end())
                    {
                    continue;
                    }
                unsigned int typeA = __scalar_as_int(a_pos.w);
                std::pair<idx, idx> sorted_type_pair(typeA < typeB ? typeA : typeB, typeA < typeB ? typeB : typeA);
                if (m_particleType_pairs.find(sorted_type_pair)
                    == m_particleType_pairs.end())
                    {
                    continue;
                    }
                Scalar3 delta_r = box.minImage(
                    make_scalar3(a_pos.x - b_pos.x, a_pos.y - b_pos.y, a_pos.z - b_pos.z));
                Scalar rsq = delta_r.x * delta_r.x + delta_r.y * delta_r.y + delta_r.z * delta_r.z;
                Scalar dist = sqrt(rsq);
                if (dist > m_rmin && dist < m_rmax)
                    {
                    Scalar dd = fabs(dist - m_r0);
                    if (dd < best_dd)
                        {
                        best_dd = dd;
                        best_tagB = tagB;
                        best_dist = dist;
                        best_rsq = rsq;
                        }
                    }
                }
            }
        else
            {
            for (idx j = 0; j < n_neigh; j++)
                {
                idx b_idx = h_neighbors.data[my_offset + j];
                idx tagB = h_particleTags.data[b_idx];
                if (!full_storage && tagA >= tagB)
                    {
                    continue;
                    }
                if (h_bondRanks.data[tagB] > 0)
                    {
                    continue;
                    }
                if (h_molTags.data[tagA] == h_molTags.data[tagB])
                    {
                    continue;
                    }
                Scalar4 b_pos = h_particlePositions.data[b_idx];
                unsigned int typeB = __scalar_as_int(b_pos.w);
                if (m_particleTypes.find(typeB) == m_particleTypes.end())
                    {
                    continue;
                    }
                unsigned int typeA = __scalar_as_int(a_pos.w);
                std::pair<idx, idx> sorted_type_pair(typeA < typeB ? typeA : typeB, typeA < typeB ? typeB : typeA);
                if (m_particleType_pairs.find(sorted_type_pair)
                    == m_particleType_pairs.end())
                    {
                    continue;
                    }
                Scalar3 delta_r = box.minImage(
                    make_scalar3(a_pos.x - b_pos.x, a_pos.y - b_pos.y, a_pos.z - b_pos.z));
                Scalar rsq = delta_r.x * delta_r.x + delta_r.y * delta_r.y + delta_r.z * delta_r.z;
                Scalar dist = sqrt(rsq);
                if (dist > m_rmin && dist < m_rmax)
                    {
                    best_tagB = tagB;
                    best_dist = dist;
                    best_rsq = rsq;
                    break;
                    }
                }
            }

        if (best_tagB != 0xffffffff)
            {
            cand_partner[tagA] = best_tagB;
            cand_dist[tagA] = best_dist;
            cand_rsq[tagA] = best_rsq;
            }
        }

    // Phase 2: Lock Matching
    std::vector<std::pair<idx, idx>> tentative_matches;
    for (idx i = 0; i < shuffled_particles.size(); i++)
        {
        idx tagA = shuffled_particles[i];
        if (h_bondRanks.data[tagA] > 0)
            {
            continue;
            }
        idx tagB = cand_partner[tagA];
        if (tagB == 0xffffffff)
            {
            continue;
            }
        if (h_bondRanks.data[tagB] == 0)
            {
            // Lock claims
            h_bondRanks.data[tagA] = 1;
            h_bondRanks.data[tagB] = 1;
            tentative_matches.push_back(std::pair<idx, idx>(tagA, tagB));
            }
        }

    // Phase 3: Kinetics Evaluation (Accept/Reject)
    for (const auto& pair : tentative_matches)
        {
        idx tagA = pair.first;
        idx tagB = pair.second;
        Scalar rsq = cand_rsq[tagA];
        Scalar P = Pon;
        if (m_flag_metropolis)
            {
            Scalar force_divr = Scalar(0.0);
            Scalar bond_eng = Scalar(0.0);
            bond_class eval(rsq, m_bond_params);
            eval.evalForceAndEnergy(force_divr, bond_eng);
            P *= exp(-bond_eng / temp);
            }
        RandomGenerator rng(
            Seed(BONDING_RNG_IDENTIFIER + m_instance_number, timestep, m_seed),
            Counter(tagA < tagB ? tagA : tagB, tagA < tagB ? tagB : tagA));
        UniformDistribution<Scalar> uniform(0, 1);
        Scalar p = uniform(rng);
        if (p < P)
            {
            toAddPairs.push_back(pair);
            }
        else
            {
            // Reject: Release locks
            h_bondRanks.data[tagA] = 0;
            h_bondRanks.data[tagB] = 0;
            }
        }
    }

namespace detail
    {
void export_DyBond(pybind11::module& m)
    {
    pybind11::class_<DyBondFENE, DyBondBase, std::shared_ptr<DyBondFENE>>(m, "DyBondFENE")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            pybind11::dict,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>());
    pybind11::class_<DyBondHarmonic, DyBondBase, std::shared_ptr<DyBondHarmonic>>(m,
                                                                                  "DyBondHarmonic")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            pybind11::dict,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>());
    pybind11::class_<DyBondTether, DyBondBase, std::shared_ptr<DyBondTether>>(m, "DyBondTether")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            pybind11::dict,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>());
    ///.def("addTypePair", &DyBondTether::addTypePair)
    ///.def("setTempParams", &DyBondTether::setTempParams)
    ///.def("setMetropolisFlag", &DyBond<bond_class2>::setMetropolisFlag)
    ///.def("setForceParams", &DyBond<bond_class2>::setForceParams)
    ///.def("setSelfAvoidingFlag", &DyBond<bond_class2>::setSelfAvoidingFlag);
    }
    } // namespace detail

#ifdef ENABLE_HIP

template<class bond_type> void DyBondGPU<bond_type>::setGPUStruct()
    {

    m_dybond_params.NOT_BONDABLE = NOT_BONDABLE;
    m_dybond_params.bond_type = m_bond_type;

    m_dybond_params.dt = m_dt;
    m_dybond_params.period = m_period;

    m_dybond_params.seed = m_seed;
    m_dybond_params.unbonding_rng_identifier = UNBONDING_RNG_IDENTIFIER + m_instance_number;
    m_dybond_params.bonding_rng_identifier = BONDING_RNG_IDENTIFIER + m_instance_number;

    m_dybond_params.kon = m_kon;
    m_dybond_params.koff = m_koff;
    m_dybond_params.rmin = m_rmin;
    m_dybond_params.rmax = m_rmax;
    m_dybond_params.metropolisFlag = m_flag_metropolis;
    m_dybond_params.nearestNeighborFlag = m_flag_nearest_neighbor;
    m_dybond_params.r0 = m_r0;

    m_dybond_params.forceFlag = m_flag_force;
    m_dybond_params.force_a = m_force_a;

    m_dybond_params.selfAvoidingFlag = m_flag_self_avoiding;

    m_dybond_params.temperatureFlag = m_flag_temperature;
    m_dybond_params.Tmelt = m_Tmelt;
    m_dybond_params.alpha = m_alpha;
    m_dybond_params.kon_melt = m_kon_melt;
    m_dybond_params.koff_melt = m_koff_melt;

    m_dybond_params.bond_params = m_bond_params;
    m_dybond_params.numDyBondParticles = m_dybondParticleTags.size();

    idx numPairs = m_particleType_pairs.size();
    m_dybond_params.numTypePairs = numPairs;
    idx typeAs[numPairs];
    idx typeBs[numPairs];
    idx i = 0;
    for (std::pair<idx, idx> pair : m_particleType_pairs)
        {
        typeAs[i] = pair.first;
        typeBs[i] = pair.second;
        i++;
        }

    if (m_gpu_params_dirty)
        {
        if (m_dybond_params.d_typeAs == nullptr)
            {
            hipMalloc(&(m_dybond_params.d_typeAs), numPairs * sizeof(idx));
            }
        hipMemcpy(m_dybond_params.d_typeAs, typeAs, numPairs * sizeof(idx), hipMemcpyHostToDevice);

        if (m_dybond_params.d_typeBs == nullptr)
            {
            hipMalloc(&(m_dybond_params.d_typeBs), numPairs * sizeof(idx));
            }
        hipMemcpy(m_dybond_params.d_typeBs, typeBs, numPairs * sizeof(idx), hipMemcpyHostToDevice);

        if (d_dybond_params == nullptr)
            {
            hipMalloc(&d_dybond_params, sizeof(DyBondParams<bond_type>));
            }
        hipMemcpy(d_dybond_params, &m_dybond_params, sizeof(DyBondParams<bond_type>), hipMemcpyHostToDevice);
        m_gpu_params_dirty = false;
        }
    }

template<class bond_type>
void DyBondGPU<bond_type>::findBondsToBreak(set<idx>& toRemoveTag,
                                            std::vector<int4>& brokenEvents,
                                            const std::shared_ptr<hoomd::ParticleData>& pdata,
                                            const std::shared_ptr<hoomd::BondData>& bond_data,
                                            GPUVector<idx>& bond_ranks,
                                            const GPUArray<Scalar>& properties,
                                            Scalar translational_dof,
                                            const uint64_t& timestep)
    {
    setGPUStruct();
    idx numBonds = m_bondTags.size();
    if (m_bondsToBreak.getNumElements() < numBonds)
        {
        GPUArray<int4> new_bondsToBreak(numBonds, m_exec_conf);
        m_bondsToBreak.swap(new_bondsToBreak);
        }
    if (m_numBondsToBreak.getNumElements() < 1)
        {
        GPUArray<idx> new_numBondsToBreak(1, m_exec_conf);
        m_numBondsToBreak.swap(new_numBondsToBreak);
        }

    {
    ArrayHandle<int4> d_bondsToBreak(m_bondsToBreak, access_location::device, access_mode::overwrite);
    ArrayHandle<idx> d_numBondsToBreak(m_numBondsToBreak, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_properties(properties, access_location::device, access_mode::read);
    ArrayHandle<idx> d_bond_ranks(bond_ranks, access_location::device, access_mode::readwrite);

    *m_numBondsToBreak_host = 0;
    hipMemcpyAsync(d_numBondsToBreak.data, m_numBondsToBreak_host, sizeof(idx), hipMemcpyHostToDevice, m_stream);

    if (m_flag_force)
        {
        ArrayHandle<idx> d_bondIndices(bond_data->getRTags(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<BondData::members_t> d_bondPairs(bond_data->getMembersArray(),
                                                     access_location::device,
                                                     access_mode::read);
        ArrayHandle<idx> d_particleIndices(pdata->getRTags(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_particlePositions(pdata->getPositions(),
                                                 access_location::device,
                                                 access_mode::read);
        BoxDim box = pdata->getBox();

        ArrayHandle<idx> d_bondTags(m_bondTags, access_location::device, access_mode::read);

        gpu_try_breaks_with_force<bond_type, 2>(numBonds,
                                                d_bondTags.data,
                                                d_bondIndices.data,
                                                d_bondPairs.data,
                                                d_particleIndices.data,
                                                d_particlePositions.data,
                                                d_bond_ranks.data,
                                                d_bondsToBreak.data,
                                                d_numBondsToBreak.data,
                                                timestep,
                                                d_properties.data,
                                                translational_dof,
                                                d_dybond_params,
                                                box,
                                                m_stream);
        }
    else
        {
        ArrayHandle<idx> d_bondTags(m_bondTags, access_location::device, access_mode::read);
        ArrayHandle<idx> d_bondIndices(bond_data->getRTags(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<BondData::members_t> d_bondPairs(bond_data->getMembersArray(),
                                                     access_location::device,
                                                     access_mode::read);
        gpu_try_breaks<bond_type>(numBonds,
                                  d_bondTags.data,
                                  d_bondIndices.data,
                                  d_bondPairs.data,
                                  d_bond_ranks.data,
                                  d_bondsToBreak.data,
                                  d_numBondsToBreak.data,
                                  timestep,
                                  d_properties.data,
                                  translational_dof,
                                  d_dybond_params,
                                  m_stream);
        }
    hipMemcpyAsync(m_numBondsToBreak_host, d_numBondsToBreak.data, sizeof(idx), hipMemcpyDeviceToHost, m_stream);
    }
    {
    hipStreamSynchronize(m_stream);
    idx count = *m_numBondsToBreak_host;
    if (count > 0)
        {
        ArrayHandle<int4> h_bondsToBreak(m_bondsToBreak, access_location::host, access_mode::read);
        for (idx i = 0; i < count; i++)
            {
            int4 event = h_bondsToBreak.data[i];
            toRemoveTag.insert(event.x);
            brokenEvents.push_back(event);
            }
        }
    }
    }

template<class bond_type>
void DyBondGPU<bond_type>::findBondsToMake(std::vector<std::pair<idx, idx>>& toAddPairs,
                                           const std::shared_ptr<hoomd::ParticleData>& pdata,
                                           const std::shared_ptr<hoomd::md::NeighborList>& nlist,
                                           const GPUVector<idx>& mol_tags,
                                           GPUVector<idx>& bond_ranks,
                                           const GPUArray<Scalar>& properties,
                                           Scalar translational_dof,
                                           const uint64_t& timestep)
    {
    setGPUStruct();
    idx N = pdata->getN();

    if (m_cand_partner.getNumElements() < N)
        {
        GPUArray<idx> new_cand_partner(N, m_exec_conf);
        m_cand_partner.swap(new_cand_partner);
        GPUArray<Scalar> new_cand_dist(N, m_exec_conf);
        m_cand_dist.swap(new_cand_dist);
        GPUArray<idx> new_prop_from(N, m_exec_conf);
        m_prop_from.swap(new_prop_from);
        GPUArray<idx> new_prop_to(N, m_exec_conf);
        m_prop_to.swap(new_prop_to);
        }
    if (m_n_prop.getNumElements() < 1)
        {
        GPUArray<idx> new_n_prop(1, m_exec_conf);
        m_n_prop.swap(new_n_prop);
        }
    if (m_bondsToMake.getNumElements() < N)
        {
        GPUArray<int2> new_bondsToMake(N, m_exec_conf);
        m_bondsToMake.swap(new_bondsToMake);
        }
    if (m_numBondsToMake.getNumElements() < 1)
        {
        GPUArray<idx> new_numBondsToMake(1, m_exec_conf);
        m_numBondsToMake.swap(new_numBondsToMake);
        }

    {
    ArrayHandle<idx> d_cand_partner(m_cand_partner, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_cand_dist(m_cand_dist, access_location::device, access_mode::overwrite);
    ArrayHandle<idx> d_prop_from(m_prop_from, access_location::device, access_mode::overwrite);
    ArrayHandle<idx> d_prop_to(m_prop_to, access_location::device, access_mode::overwrite);
    ArrayHandle<idx> d_n_prop(m_n_prop, access_location::device, access_mode::readwrite);

    ArrayHandle<int2> d_bondsToMake(m_bondsToMake, access_location::device, access_mode::overwrite);
    ArrayHandle<idx> d_numBondsToMake(m_numBondsToMake, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_properties(properties, access_location::device, access_mode::read);

    ArrayHandle<idx> d_dybondParticleTags(m_dybondParticleTags, access_location::device, access_mode::read);
    ArrayHandle<idx> d_particleTags(pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<idx> d_particleIndices(pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_particlePositions(pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = pdata->getBox();

    ArrayHandle<idx> d_numNeighbors(nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<idx> d_neighbors(nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<size_t> d_neighborOffset(nlist->getHeadList(), access_location::device, access_mode::read);
    ArrayHandle<idx> d_molTags(mol_tags, access_location::device, access_mode::read);
    ArrayHandle<idx> d_bondRanks(bond_ranks, access_location::device, access_mode::readwrite);

    // Reset device counters
    idx zero = 0;
    hipMemcpyAsync(d_n_prop.data, &zero, sizeof(idx), hipMemcpyHostToDevice, m_stream);
    hipMemcpyAsync(d_numBondsToMake.data, &zero, sizeof(idx), hipMemcpyHostToDevice, m_stream);

    // 1. Candidate search
    gpu_dybond_candidates<bond_type>(N, m_dybondParticleTags.size(), d_particlePositions.data, d_particleIndices.data, d_particleTags.data,
                                     d_numNeighbors.data, d_neighbors.data, d_neighborOffset.data,
                                     d_bondRanks.data, d_molTags.data, d_dybondParticleTags.data, d_dybond_params, box,
                                     d_cand_partner.data, d_cand_dist.data, 256, m_stream);

    // 2. Lock-free mutual exclusion matching
    gpu_dybond_bind_match(m_dybondParticleTags.size(), d_dybondParticleTags.data, d_cand_partner.data,
                          d_bondRanks.data, d_prop_from.data, d_prop_to.data, d_n_prop.data, 256, m_stream);

    // Copy n_prop back to host to see if we need to call accept kernel
    idx n_prop = 0;
    hipMemcpyAsync(&n_prop, d_n_prop.data, sizeof(idx), hipMemcpyDeviceToHost, m_stream);
    hipStreamSynchronize(m_stream);

    if (n_prop > 0)
        {
        if (n_prop > N) n_prop = N;
        // 3. Probability & kinetics evaluation (accept/reject)
        gpu_dybond_bind_accept<bond_type>(n_prop, m_dybondParticleTags.size(), d_prop_from.data, d_prop_to.data, d_cand_dist.data,
                                          d_bondRanks.data, d_properties.data, translational_dof,
                                          d_dybond_params, timestep, d_bondsToMake.data, d_numBondsToMake.data,
                                          256, m_stream);
        }

    hipMemcpyAsync(m_numBondsToMake_host, d_numBondsToMake.data, sizeof(idx), hipMemcpyDeviceToHost, m_stream);
    }
    {
    hipStreamSynchronize(m_stream);
    idx count = *m_numBondsToMake_host;
    if (count > 0)
        {
        ArrayHandle<int2> h_bondsToMake(m_bondsToMake, access_location::host, access_mode::read);
        for (idx i = 0; i < count; i++)
            {
            toAddPairs.push_back(std::pair<idx, idx>((idx)h_bondsToMake.data[i].x, (idx)h_bondsToMake.data[i].y));
            }
        }
    }
    }

namespace detail
    {
void export_DyBondGPU(pybind11::module& m)
    {
    pybind11::class_<DyBondFENEGPU, DyBondBase, std::shared_ptr<DyBondFENEGPU>>(m, "DyBondFENEGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            pybind11::dict,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>());
    pybind11::class_<DyBondHarmonicGPU, DyBondBase, std::shared_ptr<DyBondHarmonicGPU>>(
        m,
        "DyBondHarmonicGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            pybind11::dict,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>());
    pybind11::class_<DyBondTetherGPU, DyBondBase, std::shared_ptr<DyBondTetherGPU>>(
        m,
        "DyBondTetherGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            idx,
                            Scalar,
                            idx,
                            pybind11::dict,
                            Scalar,
                            Scalar,
                            Scalar,
                            Scalar>());
    }
    } // namespace detail

#endif

    } // namespace hoomd
