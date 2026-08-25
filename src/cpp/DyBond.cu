#include "DyBond.cuh"
#include <iostream>
#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/md/EvaluatorBondFENE.h"
#include "hoomd/md/EvaluatorBondHarmonic.h"
#include "hoomd/md/EvaluatorBondTether.h"

namespace hoomd
    {
namespace kernel
    {

template<class bond_class, int group_size>
__global__ void gpu_try_breaks_with_force_kernel(const idx numBonds,
                                                 const idx* d_bondTags,
                                                 const idx* d_bondIndices,
                                                 const group_storage<group_size>* d_bondPairs,
                                                 const idx* d_particleIndices,
                                                 const Scalar4* d_particlePositions,
                                                 idx* d_bond_ranks,
                                                 int4* d_bondsToBreak,
                                                 idx* d_numBondsToBreak,
                                                 const uint64_t timestep,
                                                 const Scalar* d_properties,
                                                 Scalar translational_dof,
                                                 const DyBondParams<bond_class>* d_params,
                                                 const BoxDim box)
    {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numBonds)
        {
        Scalar temp = Scalar(0.0);
        if (translational_dof > Scalar(0.0))
            {
            temp = (d_properties[0] * Scalar(2.0)) / translational_dof;
            }
        if (temp <= Scalar(1e-6)) return;

        Scalar koff = d_params->koff;
        if (d_params->temperatureFlag)
            {
            Scalar g = 0.5 * (tanh(d_params->alpha * (temp - d_params->Tmelt)) + 1.0);
            koff = koff * (1 - g) + d_params->koff_melt * g;
            }
        Scalar Poff = koff * (Scalar)d_params->period * d_params->dt;

        idx bond_tag = d_bondTags[i];
        idx bond_index = d_bondIndices[bond_tag];
        group_storage<group_size> bond_pair = d_bondPairs[bond_index];

        idx a_idx = d_particleIndices[bond_pair.tag[0]];
        idx b_idx = d_particleIndices[bond_pair.tag[1]];

        Scalar4 a_pos = d_particlePositions[a_idx];
        Scalar4 b_pos = d_particlePositions[b_idx];

        Scalar3 delta_r
            = box.minImage(make_scalar3(a_pos.x - b_pos.x, a_pos.y - b_pos.y, a_pos.z - b_pos.z));
        Scalar dist_sq = delta_r.x * delta_r.x + delta_r.y * delta_r.y + delta_r.z * delta_r.z;

        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);
        bond_class eval(dist_sq, d_params->bond_params);
        eval.evalForceAndEnergy(force_divr, bond_eng);
        Scalar P = Poff * exp(d_params->force_a * sqrt(bond_eng / temp));

        idx tagA = bond_pair.tag[0];
        idx tagB = bond_pair.tag[1];
        RandomGenerator rng(Seed(d_params->unbonding_rng_identifier, timestep, d_params->seed),
                            Counter(tagA < tagB ? tagA : tagB, tagA < tagB ? tagB : tagA));
        UniformDistribution<Scalar> uniform(0, 1);
        Scalar p = uniform(rng);

        if (p < P)
            {
            idx slot = atomicAdd(d_numBondsToBreak, 1);
            d_bondsToBreak[slot] = make_int4(bond_tag, tagA, tagB, 0);
            d_bond_ranks[tagA] = 0;
            d_bond_ranks[tagB] = 0;
            }
        }
    }

template<class bond_class, int group_size>
hipError_t gpu_try_breaks_with_force(const idx& numBonds,
                                     const idx* d_bondTags,
                                     const idx* d_bondIndices,
                                     const group_storage<group_size>* d_bondPairs,
                                     const idx* d_particleIndices,
                                     const Scalar4* d_particlePositions,
                                     idx* d_bond_ranks,
                                     int4* d_bondsToBreak,
                                     idx* d_numBondsToBreak,
                                     const uint64_t timestep,
                                     const Scalar* d_properties,
                                     Scalar translational_dof,
                                     const DyBondParams<bond_class>* d_params,
                                     BoxDim box,
                                     hipStream_t stream)
    {
    int block_size = 256;
    dim3 grid((int)ceil((double)numBonds / (double)block_size), 1, 1);
    if (grid.x == 0)
        {
        grid.x = 1;
        }
    dim3 threads(block_size, 1, 1);
    // run the kernel
    hipLaunchKernelGGL((gpu_try_breaks_with_force_kernel<bond_class, group_size>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       stream,
                       numBonds,
                       d_bondTags,
                       d_bondIndices,
                       d_bondPairs,
                       d_particleIndices,
                       d_particlePositions,
                       d_bond_ranks,
                       d_bondsToBreak,
                       d_numBondsToBreak,
                       timestep,
                       d_properties,
                       translational_dof,
                       d_params,
                       box);
    return hipSuccess;
    }

template<class bond_class>
__global__ void gpu_try_breaks_kernel(idx numBonds,
                                       const idx* d_bondTags,
                                       const idx* d_bondIndices,
                                       const group_storage<2>* d_bondPairs,
                                       idx* d_bond_ranks,
                                       int4* d_bondsToBreak,
                                       idx* d_numBondsToBreak,
                                       uint64_t timestep,
                                       const Scalar* d_properties,
                                       Scalar translational_dof,
                                       const DyBondParams<bond_class>* d_params)
    {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numBonds)
        {
        Scalar temp = Scalar(0.0);
        if (translational_dof > Scalar(0.0))
            {
            temp = (d_properties[0] * Scalar(2.0)) / translational_dof;
            }
        if (temp <= Scalar(1e-6)) return;

        Scalar koff = d_params->koff;
        if (d_params->temperatureFlag)
            {
            Scalar g = 0.5 * (tanh(d_params->alpha * (temp - d_params->Tmelt)) + 1.0);
            koff = koff * (1 - g) + d_params->koff_melt * g;
            }
        Scalar Poff = koff * (Scalar)d_params->period * d_params->dt;

        idx bond_tag = d_bondTags[i];
        idx bond_index = d_bondIndices[bond_tag];
        group_storage<2> bond_pair = d_bondPairs[bond_index];
        idx tagA = bond_pair.tag[0];
        idx tagB = bond_pair.tag[1];

        RandomGenerator rng(Seed(d_params->unbonding_rng_identifier, timestep, d_params->seed),
                            Counter(tagA < tagB ? tagA : tagB, tagA < tagB ? tagB : tagA));
        UniformDistribution<Scalar> uniform(0, 1);
        Scalar p = uniform(rng);

        if (p < Poff)
            {
            idx slot = atomicAdd(d_numBondsToBreak, 1);
            d_bondsToBreak[slot] = make_int4(bond_tag, tagA, tagB, 0);
            d_bond_ranks[tagA] = 0;
            d_bond_ranks[tagB] = 0;
            }
        }
    }

template<class bond_class>
hipError_t gpu_try_breaks(const idx& numBonds,
                          const idx* d_bondTags,
                          const idx* d_bondIndices,
                          const group_storage<2>* d_bondPairs,
                          idx* d_bond_ranks,
                          int4* d_bondsToBreak,
                          idx* d_numBondsToBreak,
                          const uint64_t timestep,
                          const Scalar* d_properties,
                          Scalar translational_dof,
                          const DyBondParams<bond_class>* d_params,
                          hipStream_t stream)
    {
    int block_size = 256;
    dim3 grid((int)ceil((double)numBonds / (double)block_size), 1, 1);
    if (grid.x == 0)
        {
        grid.x = 1;
        }
    dim3 threads(block_size, 1, 1);
    // run the kernel
    hipLaunchKernelGGL(gpu_try_breaks_kernel<bond_class>,
                       dim3(grid),
                       dim3(threads),
                       0,
                       stream,
                       numBonds,
                       d_bondTags,
                       d_bondIndices,
                       d_bondPairs,
                       d_bond_ranks,
                       d_bondsToBreak,
                       d_numBondsToBreak,
                       timestep,
                       d_properties,
                       translational_dof,
                       d_params);
    return hipSuccess;
    }

__device__ bool isMatch(idx typeA, idx typeB, idx* d_typeAs, idx* d_typeBs, idx num_pairs)
    {
    if (typeA > typeB)
        {
        idx temp = typeA;
        typeA = typeB;
        typeB = temp;
        }
    for (idx j = 0; j < num_pairs; j++)
        {
        if (typeA == d_typeAs[j] && typeB == d_typeBs[j])
            return true;
        }
    return false;
    }

template<class bond_class>
__global__ void gpu_try_makes_pair_kernel(int2* d_bondsToMake,
                                          idx* d_numBondsToMake,
                                          const uint32_t numPairs,
                                          const idx* d_dybondParticleTags,
                                          const idx* d_particleIndices,
                                          const idx* d_particleTags,
                                          const Scalar4* d_particlePositions,
                                          const idx* d_molTags,
                                          const idx* d_neighborIndices,
                                          const idx* d_numNeighbors,
                                          const size_t* d_neighborOffset,
                                          const idx* d_bondRanks,
                                          const uint64_t timestep,
                                          const Scalar* d_properties,
                                          Scalar translational_dof,
                                          const bool fullStorageFlag,
                                          const DyBondParams<bond_class>* d_params,
                                          const BoxDim box)
    {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPairs)
        {
        Scalar temp = Scalar(0.0);
        if (translational_dof > Scalar(0.0))
            {
            temp = (d_properties[0] * Scalar(2.0)) / translational_dof;
            }
        if (temp <= Scalar(1e-6)) return;

        Scalar kon = d_params->kon;
        if (d_params->temperatureFlag)
            {
            Scalar g = 0.5 * (tanh(d_params->alpha * (temp - d_params->Tmelt)) + 1.0);
            kon = kon * (1 - g) + d_params->kon_melt * g;
            }
        Scalar Pon = kon * d_params->dt * (Scalar)d_params->period;
        idx dybondIdx = 0;
        idx tagA = 0;
        uint64_t sumTotal = 0;
        for (dybondIdx; dybondIdx < d_params->numDyBondParticles; dybondIdx++)
            {
            tagA = d_dybondParticleTags[dybondIdx];
            if (d_bondRanks[tagA] != 0)
                continue;
            idx indexA = d_particleIndices[tagA];
            if (i < sumTotal + d_numNeighbors[indexA])
                break;
            sumTotal += d_numNeighbors[indexA];
            }

        if (dybondIdx >= d_params->numDyBondParticles)
            return;

        idx indexA = d_particleIndices[tagA];
        size_t offset = d_neighborOffset[indexA];
        idx typeA = __scalar_as_int(d_particlePositions[indexA].w);

        idx pairIndex = offset + (i - sumTotal);
        idx indexB = d_neighborIndices[pairIndex];

        idx typeB = __scalar_as_int(d_particlePositions[indexB].w);
        Scalar P = Pon;
        if (!isMatch(typeA, typeB, d_params->d_typeAs, d_params->d_typeBs, d_params->numTypePairs))
            {
            P = -1;
            }
        idx tagB = d_particleTags[indexB];
        if (d_bondRanks[tagB])
            {
            P = -1;
            }
        if (fullStorageFlag && tagA > tagB)
            {
            P = -1;
            }
        if (d_params->selfAvoidingFlag && d_molTags[tagA] == d_molTags[tagB])
            {
            P = -1;
            }
        Scalar4 a_pos = d_particlePositions[indexA];
        Scalar4 b_pos = d_particlePositions[indexB];
        Scalar3 delta = make_scalar3(a_pos.x - b_pos.x, a_pos.y - b_pos.y, a_pos.z - b_pos.z);
        delta = box.minImage(delta);
        Scalar rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        Scalar r = sqrt(rsq);
        if (r < d_params->rmin || r > d_params->rmax)
            {
            P = -1;
            }
        if (d_params->metropolisFlag)
            {
            Scalar force_divr = Scalar(0.0);
            Scalar bond_eng = Scalar(0.0);
            bond_class eval(rsq, d_params->bond_params);
            eval.evalForceAndEnergy(force_divr, bond_eng);
            P *= exp(-bond_eng / temp);
            }

        RandomGenerator rng(Seed(d_params->bonding_rng_identifier, timestep, d_params->seed),
                            Counter(tagA < tagB ? tagA : tagB, tagA < tagB ? tagB : tagA));
        UniformDistribution<Scalar> uniform(0, 1);
        Scalar p = uniform(rng);

        if (p < P)
            {
            idx slot = atomicAdd(d_numBondsToMake, 1);
            d_bondsToMake[slot] = make_int2(tagA, tagB);
            }
        }
    }

template<class bond_class>
__global__ void gpu_try_makes_particle_kernel(int2* d_bondsToMake,
                                              idx* d_numBondsToMake,
                                              const idx* d_dybondParticleTags,
                                              const idx* d_particleIndices,
                                              const idx* d_particleTags,
                                              const Scalar4* d_particlePositions,
                                              const idx* d_molTags,
                                              const idx* d_neighborIndices,
                                              const idx* d_numNeighbors,
                                              const size_t* d_neighborOffset,
                                              const idx* d_bondRanks,
                                              const uint64_t timestep,
                                              const Scalar* d_properties,
                                              Scalar translational_dof,
                                              const bool fullStorageFlag,
                                              const DyBondParams<bond_class>* d_params,
                                              const BoxDim box)
    {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_params->numDyBondParticles)
        {
        Scalar temp = Scalar(0.0);
        if (translational_dof > Scalar(0.0))
            {
            temp = (d_properties[0] * Scalar(2.0)) / translational_dof;
            }
        if (temp <= Scalar(1e-6)) return;

        Scalar kon = d_params->kon;
        if (d_params->temperatureFlag)
            {
            Scalar g = 0.5 * (tanh(d_params->alpha * (temp - d_params->Tmelt)) + 1.0);
            kon = kon * (1 - g) + d_params->kon_melt * g;
            }
        Scalar Pon = kon * d_params->dt * (Scalar)d_params->period;
        idx tagA = d_dybondParticleTags[i];
        if (d_bondRanks[tagA] != 0)
            return;

        idx indexA = d_particleIndices[tagA];
        idx typeA = __scalar_as_int(d_particlePositions[indexA].w);
        size_t offset = d_neighborOffset[indexA];
        idx numNeighs = d_numNeighbors[indexA];

        idx best_tagB = 0;
        idx best_indexB = 0;
        Scalar best_rsq = 0.0;
        Scalar min_diff = 1e9;
        bool found = false;

        for (idx j = 0; j < numNeighs; j++)
            {
            idx indexB = d_neighborIndices[offset + j];
            idx typeB = __scalar_as_int(d_particlePositions[indexB].w);
            idx tagB = d_particleTags[indexB];

            if (d_bondRanks[tagB])
                continue;
            if (fullStorageFlag && tagA > tagB)
                continue;
            if (d_params->selfAvoidingFlag && d_molTags[tagA] == d_molTags[tagB])
                continue;

            if (!isMatch(typeA, typeB, d_params->d_typeAs, d_params->d_typeBs, d_params->numTypePairs))
                continue;

            Scalar4 a_pos = d_particlePositions[indexA];
            Scalar4 b_pos = d_particlePositions[indexB];
            Scalar3 delta = make_scalar3(a_pos.x - b_pos.x, a_pos.y - b_pos.y, a_pos.z - b_pos.z);
            delta = box.minImage(delta);
            Scalar rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            Scalar r = sqrt(rsq);

            if (r < d_params->rmin || r > d_params->rmax)
                continue;

            Scalar diff = abs(r - d_params->r0);
            if (diff < min_diff)
                {
                min_diff = diff;
                best_tagB = tagB;
                best_indexB = indexB;
                best_rsq = rsq;
                found = true;
                }
            }

        if (found)
            {
            Scalar P = Pon;
            if (d_params->metropolisFlag)
                {
                Scalar force_divr = Scalar(0.0);
                Scalar bond_eng = Scalar(0.0);
                bond_class eval(best_rsq, d_params->bond_params);
                eval.evalForceAndEnergy(force_divr, bond_eng);
                P *= exp(-bond_eng / temp);
                }

            RandomGenerator rng(Seed(d_params->bonding_rng_identifier, timestep, d_params->seed),
                                Counter(tagA < best_tagB ? tagA : best_tagB, tagA < best_tagB ? best_tagB : tagA));
            UniformDistribution<Scalar> uniform(0, 1);
            Scalar p = uniform(rng);

            if (p < P)
                {
                idx slot = atomicAdd(d_numBondsToMake, 1);
                d_bondsToMake[slot] = make_int2(tagA, best_tagB);
                }
            }
        }
    }

template<class bond_class>
hipError_t gpu_try_makes(int2* d_bondsToMake,
                          idx* d_numBondsToMake,
                          const uint32_t& numPairs,
                          const idx* d_dybondParticleTags,
                          const idx* d_particleIndices,
                          const idx* d_particleTags,
                          const Scalar4* d_particlePositions,
                          const idx* d_molTags,
                          const idx* d_neighborIndices,
                          const idx* d_numNeighbors,
                          const size_t* d_neighborOffset,
                          const idx* d_bondRanks,
                          const uint64_t& timestep,
                          const Scalar* d_properties,
                          Scalar translational_dof,
                          const bool& fullStorageFlag,
                          const bool& nearestNeighbor,
                          const DyBondParams<bond_class>* d_params,
                          BoxDim box,
                          hipStream_t stream)
    {
    int block_size = 256;
    if (nearestNeighbor)
        {
        dim3 grid((int)ceil((double)numPairs / (double)block_size), 1, 1);
        if (grid.x == 0)
            {
            grid.x = 1;
            }
        dim3 threads(block_size, 1, 1);
        hipLaunchKernelGGL((gpu_try_makes_particle_kernel<bond_class>),
                           dim3(grid),
                           dim3(threads),
                           0,
                           stream,
                           d_bondsToMake,
                           d_numBondsToMake,
                           d_dybondParticleTags,
                           d_particleIndices,
                           d_particleTags,
                           d_particlePositions,
                           d_molTags,
                           d_neighborIndices,
                           d_numNeighbors,
                           d_neighborOffset,
                           d_bondRanks,
                           timestep,
                           d_properties,
                           translational_dof,
                           fullStorageFlag,
                           d_params,
                           box);
        }
    else
        {
        dim3 grid((int)ceil((double)numPairs / (double)block_size), 1, 1);
        if (grid.x == 0)
            {
            grid.x = 1;
            }
        dim3 threads(block_size, 1, 1);
        hipLaunchKernelGGL((gpu_try_makes_pair_kernel<bond_class>),
                           dim3(grid),
                           dim3(threads),
                           0,
                           stream,
                           d_bondsToMake,
                           d_numBondsToMake,
                           numPairs,
                           d_dybondParticleTags,
                           d_particleIndices,
                           d_particleTags,
                           d_particlePositions,
                           d_molTags,
                           d_neighborIndices,
                           d_numNeighbors,
                           d_neighborOffset,
                           d_bondRanks,
                           timestep,
                           d_properties,
                           translational_dof,
                           fullStorageFlag,
                           d_params,
                           box);
        }
    return hipSuccess;
    }

template hipError_t gpu_try_breaks<md::EvaluatorBondFENE>(const idx& numBonds,
                                                         const idx* d_bondTags,
                                                         const idx* d_bondIndices,
                                                         const group_storage<2>* d_bondPairs,
                                                         idx* d_bond_ranks,
                                                         int4* d_bondsToBreak,
                                                         idx* d_numBondsToBreak,
                                                         const uint64_t timestep,
                                                         const Scalar* d_properties,
                                                         Scalar translational_dof,
                                                         const DyBondParams<md::EvaluatorBondFENE>* d_params,
                                                         hipStream_t stream);
template hipError_t gpu_try_breaks<md::EvaluatorBondHarmonic>(const idx& numBonds,
                                                             const idx* d_bondTags,
                                                             const idx* d_bondIndices,
                                                             const group_storage<2>* d_bondPairs,
                                                             idx* d_bond_ranks,
                                                             int4* d_bondsToBreak,
                                                             idx* d_numBondsToBreak,
                                                             const uint64_t timestep,
                                                             const Scalar* d_properties,
                                                             Scalar translational_dof,
                                                             const DyBondParams<md::EvaluatorBondHarmonic>* d_params,
                                                             hipStream_t stream);
template hipError_t gpu_try_breaks<md::EvaluatorBondTether>(const idx& numBonds,
                                                           const idx* d_bondTags,
                                                           const idx* d_bondIndices,
                                                           const group_storage<2>* d_bondPairs,
                                                           idx* d_bond_ranks,
                                                           int4* d_bondsToBreak,
                                                           idx* d_numBondsToBreak,
                                                           const uint64_t timestep,
                                                           const Scalar* d_properties,
                                                           Scalar translational_dof,
                                                           const DyBondParams<md::EvaluatorBondTether>* d_params,
                                                           hipStream_t stream);

template hipError_t gpu_try_breaks_with_force<md::EvaluatorBondFENE, 2>(const idx& numBonds,
                                                                       const idx* d_bondTags,
                                                                       const idx* d_bondIndices,
                                                                       const group_storage<2>* d_bondPairs,
                                                                       const idx* d_particleIndices,
                                                                       const Scalar4* d_particlePositions,
                                                                       idx* d_bond_ranks,
                                                                       int4* d_bondsToBreak,
                                                                       idx* d_numBondsToBreak,
                                                                       const uint64_t timestep,
                                                                       const Scalar* d_properties,
                                                                       Scalar translational_dof,
                                                                       const DyBondParams<md::EvaluatorBondFENE>* d_params,
                                                                       BoxDim box,
                                                                       hipStream_t stream);
template hipError_t gpu_try_breaks_with_force<md::EvaluatorBondHarmonic, 2>(const idx& numBonds,
                                                                           const idx* d_bondTags,
                                                                           const idx* d_bondIndices,
                                                                           const group_storage<2>* d_bondPairs,
                                                                           const idx* d_particleIndices,
                                                                           const Scalar4* d_particlePositions,
                                                                           idx* d_bond_ranks,
                                                                           int4* d_bondsToBreak,
                                                                           idx* d_numBondsToBreak,
                                                                           const uint64_t timestep,
                                                                           const Scalar* d_properties,
                                                                           Scalar translational_dof,
                                                                           const DyBondParams<md::EvaluatorBondHarmonic>* d_params,
                                                                           BoxDim box,
                                                                           hipStream_t stream);
template hipError_t gpu_try_breaks_with_force<md::EvaluatorBondTether, 2>(const idx& numBonds,
                                                                         const idx* d_bondTags,
                                                                         const idx* d_bondIndices,
                                                                         const group_storage<2>* d_bondPairs,
                                                                         const idx* d_particleIndices,
                                                                         const Scalar4* d_particlePositions,
                                                                         idx* d_bond_ranks,
                                                                         int4* d_bondsToBreak,
                                                                         idx* d_numBondsToBreak,
                                                                         const uint64_t timestep,
                                                                         const Scalar* d_properties,
                                                                         Scalar translational_dof,
                                                                         const DyBondParams<md::EvaluatorBondTether>* d_params,
                                                                         BoxDim box,
                                                                         hipStream_t stream);

template hipError_t gpu_try_makes<md::EvaluatorBondFENE>(int2* d_bondsToMake,
                                                         idx* d_numBondsToMake,
                                                         const uint32_t& numPairs,
                                                         const idx* d_dybondParticleTags,
                                                         const idx* d_particleIndices,
                                                         const idx* d_particleTags,
                                                         const Scalar4* d_particlePositions,
                                                         const idx* d_molTags,
                                                         const idx* d_neighborIndices,
                                                         const idx* d_numNeighbors,
                                                         const size_t* d_neighborOffset,
                                                         const idx* d_bondRanks,
                                                         const uint64_t& timestep,
                                                         const Scalar* d_properties,
                                                         Scalar translational_dof,
                                                         const bool& fullStorageFlag,
                                                         const bool& nearestNeighbor,
                                                         const DyBondParams<md::EvaluatorBondFENE>* d_params,
                                                         BoxDim box,
                                                         hipStream_t stream);
template hipError_t gpu_try_makes<md::EvaluatorBondHarmonic>(int2* d_bondsToMake,
                                                             idx* d_numBondsToMake,
                                                             const uint32_t& numPairs,
                                                             const idx* d_dybondParticleTags,
                                                             const idx* d_particleIndices,
                                                             const idx* d_particleTags,
                                                             const Scalar4* d_particlePositions,
                                                             const idx* d_molTags,
                                                             const idx* d_neighborIndices,
                                                             const idx* d_numNeighbors,
                                                             const size_t* d_neighborOffset,
                                                             const idx* d_bondRanks,
                                                             const uint64_t& timestep,
                                                             const Scalar* d_properties,
                                                             Scalar translational_dof,
                                                             const bool& fullStorageFlag,
                                                             const bool& nearestNeighbor,
                                                             const DyBondParams<md::EvaluatorBondHarmonic>* d_params,
                                                             BoxDim box,
                                                             hipStream_t stream);
template hipError_t gpu_try_makes<md::EvaluatorBondTether>(int2* d_bondsToMake,
                                                           idx* d_numBondsToMake,
                                                           const uint32_t& numPairs,
                                                           const idx* d_dybondParticleTags,
                                                           const idx* d_particleIndices,
                                                           const idx* d_particleTags,
                                                           const Scalar4* d_particlePositions,
                                                           const idx* d_molTags,
                                                           const idx* d_neighborIndices,
                                                           const idx* d_numNeighbors,
                                                           const size_t* d_neighborOffset,
                                                           const idx* d_bondRanks,
                                                           const uint64_t& timestep,
                                                           const Scalar* d_properties,
                                                           Scalar translational_dof,
                                                           const bool& fullStorageFlag,
                                                           const bool& nearestNeighbor,
                                                           const DyBondParams<md::EvaluatorBondTether>* d_params,
                                                           BoxDim box,
                                                           hipStream_t stream);

template<class bond_class>
__global__ void gpu_dybond_candidates_kernel(unsigned int num_dybond_particles,
                                             const Scalar4* d_pos,
                                             const idx* d_particleIndices,
                                             const idx* d_particleTags,
                                             const idx* d_n_neigh,
                                             const idx* d_nl,
                                             const size_t* d_head_list,
                                             const idx* d_rank,
                                             const idx* d_molTags,
                                             const idx* d_dybondParticleTags,
                                             const DyBondParams<bond_class>* d_params,
                                             const BoxDim box,
                                             idx* d_cand_partner,
                                             Scalar* d_cand_dist)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_dybond_particles)
        return;

    idx tagA = d_dybondParticleTags[i];
    d_cand_partner[tagA] = 0xffffffff; // NOT_FOUND
    d_cand_dist[tagA] = Scalar(0.0);

    if (d_rank[tagA] != 0)
        return;

    idx indexA = d_particleIndices[tagA];
    Scalar4 pf = d_pos[indexA];
    idx typeA = __scalar_as_int(pf.w);

    unsigned int nn = d_n_neigh[indexA];
    size_t head = d_head_list[indexA];
    Scalar best_dd = Scalar(1e30);
    idx best_tag = 0xffffffff;
    Scalar best_dist = Scalar(0.0);

    for (unsigned int jj = 0; jj < nn; ++jj)
        {
        idx j_idx = d_nl[head + jj];
        idx j_tag = d_particleTags[j_idx];
        if (d_rank[j_tag] != 0)
            continue;

        if (d_params->selfAvoidingFlag && d_molTags[tagA] == d_molTags[j_tag])
            continue;

        Scalar4 pt = d_pos[j_idx];
        idx typeB = __scalar_as_int(pt.w);
        if (!isMatch(typeA, typeB, d_params->d_typeAs, d_params->d_typeBs, d_params->numTypePairs))
            continue;

        Scalar3 delta = make_scalar3(pf.x - pt.x, pf.y - pt.y, pf.z - pt.z);
        delta = box.minImage(delta);
        Scalar rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        Scalar dist = sqrt(rsq);

        if (dist > d_params->rmin && dist < d_params->rmax)
            {
            Scalar dd = fabs(dist - d_params->r0);
            if (dd < best_dd)
                {
                best_dd = dd;
                best_tag = j_tag;
                best_dist = dist;
                }
            }
        }
    d_cand_partner[tagA] = best_tag;
    d_cand_dist[tagA] = best_dist;
    }

template<class bond_class>
hipError_t gpu_dybond_candidates(unsigned int N,
                                 unsigned int num_dybond_particles,
                                 const Scalar4* d_pos,
                                 const idx* d_rtag,
                                 const idx* d_tag,
                                 const idx* d_n_neigh,
                                 const idx* d_nl,
                                 const size_t* d_head_list,
                                 const idx* d_rank,
                                 const idx* d_cluster,
                                 const idx* d_dybondParticleTags,
                                 const DyBondParams<bond_class>* d_params,
                                 const BoxDim box,
                                 idx* d_cand_partner,
                                 Scalar* d_cand_dist,
                                 unsigned int block_size,
                                 hipStream_t stream)
    {
    if (num_dybond_particles == 0) return hipSuccess;
    dim3 grid((num_dybond_particles + block_size - 1) / block_size), threads(block_size);
    hipLaunchKernelGGL((gpu_dybond_candidates_kernel<bond_class>), grid, threads, 0, stream,
                       num_dybond_particles, d_pos, d_rtag, d_tag, d_n_neigh, d_nl,
                       d_head_list, d_rank, d_cluster, d_dybondParticleTags, d_params,
                       box, d_cand_partner, d_cand_dist);
    return hipSuccess;
    }

__global__ void gpu_dybond_bind_match_kernel(unsigned int num_dybond_particles,
                                             const idx* d_dybondParticleTags,
                                             const idx* d_cand_partner,
                                             idx* d_rank,
                                             idx* d_pf,
                                             idx* d_pt,
                                             idx* d_nprop)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_dybond_particles)
        return;

    idx tagA = d_dybondParticleTags[i];
    if (d_rank[tagA] != 0)
        return;

    idx tagB = d_cand_partner[tagA];
    if (tagB == 0xffffffff)
        return;

    // atomic mutual-exclusion claim (tentative rank=1), lower tag first
    idx a = (tagA < tagB) ? tagA : tagB;
    idx c = (tagA < tagB) ? tagB : tagA;

    if (atomicCAS(&d_rank[a], 0u, 1u) != 0u)
        return;
    if (atomicCAS(&d_rank[c], 0u, 1u) != 0u)
        {
        // rollback
        d_rank[a] = 0u;
        return;
        }

    idx idx_prop = atomicAdd(d_nprop, 1u);
    d_pf[idx_prop] = tagA;
    d_pt[idx_prop] = tagB;
    }

hipError_t gpu_dybond_bind_match(unsigned int num_dybond_particles,
                                 const idx* d_dybondParticleTags,
                                 const idx* d_cand_partner,
                                 idx* d_rank,
                                 idx* d_pf,
                                 idx* d_pt,
                                 idx* d_nprop,
                                 unsigned int block_size,
                                 hipStream_t stream)
    {
    if (num_dybond_particles == 0) return hipSuccess;
    dim3 grid((num_dybond_particles + block_size - 1) / block_size), threads(block_size);
    hipLaunchKernelGGL(gpu_dybond_bind_match_kernel, grid, threads, 0, stream,
                       num_dybond_particles, d_dybondParticleTags, d_cand_partner, d_rank,
                       d_pf, d_pt, d_nprop);
    return hipSuccess;
    }

template<class bond_class>
__global__ void gpu_dybond_bind_accept_kernel(unsigned int n_prop,
                                              const idx* d_pf,
                                              const idx* d_pt,
                                              const Scalar* d_cand_dist,
                                              idx* d_rank,
                                              const Scalar* d_properties,
                                              Scalar translational_dof,
                                              const DyBondParams<bond_class>* d_params,
                                              uint64_t timestep,
                                              int2* d_bondsToMake,
                                              idx* d_numBondsToMake)
    {
    unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_prop)
        return;

    idx tagA = d_pf[p];
    idx tagB = d_pt[p];
    Scalar dist = d_cand_dist[tagA];

    Scalar temp = Scalar(0.0);
    if (translational_dof > Scalar(0.0))
        {
        temp = (d_properties[0] * Scalar(2.0)) / translational_dof;
        }

    Scalar kon = d_params->kon;
    if (d_params->temperatureFlag)
        {
        Scalar g = 0.5 * (tanh(d_params->alpha * (temp - d_params->Tmelt)) + 1.0);
        kon = kon * (1 - g) + d_params->kon_melt * g;
        }

    Scalar Pon = kon * d_params->dt * (Scalar)d_params->period;

    if (d_params->metropolisFlag)
        {
        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);
        bond_class eval(dist * dist, d_params->bond_params);
        eval.evalForceAndEnergy(force_divr, bond_eng);
        if (temp <= Scalar(1e-6))
            {
            Pon = (bond_eng <= Scalar(0.0)) ? Pon : Scalar(0.0);
            }
        else
            {
            Pon *= exp(-bond_eng / temp);
            }
        }

    RandomGenerator rng(Seed(d_params->bonding_rng_identifier, timestep, d_params->seed),
                        Counter(tagA < tagB ? tagA : tagB, tagA < tagB ? tagB : tagA));
    UniformDistribution<Scalar> uniform(0, 1);
    Scalar r = uniform(rng);

    if (r < Pon)
        {
        idx idx_add = atomicAdd(d_numBondsToMake, 1u);
        d_bondsToMake[idx_add] = make_int2(tagA, tagB);
        }
    else
        {
        // reject: release locks
        d_rank[tagA] = 0u;
        d_rank[tagB] = 0u;
        }
    }

template<class bond_class>
hipError_t gpu_dybond_bind_accept(unsigned int n_prop,
                                  unsigned int num_dybond_particles,
                                  const idx* d_pf,
                                  const idx* d_pt,
                                  const Scalar* d_cand_dist,
                                  idx* d_rank,
                                  const Scalar* d_properties,
                                  Scalar translational_dof,
                                  const DyBondParams<bond_class>* d_params,
                                  uint64_t timestep,
                                  int2* d_bondsToMake,
                                  idx* d_numBondsToMake,
                                  unsigned int block_size,
                                  hipStream_t stream)
    {
    if (n_prop == 0) return hipSuccess;
    dim3 grid((n_prop + block_size - 1) / block_size), threads(block_size);
    hipLaunchKernelGGL((gpu_dybond_bind_accept_kernel<bond_class>), grid, threads, 0, stream,
                       n_prop, d_pf, d_pt, d_cand_dist, d_rank, d_properties, translational_dof,
                       d_params, timestep, d_bondsToMake, d_numBondsToMake);
    return hipSuccess;
    }

template hipError_t gpu_dybond_candidates<md::EvaluatorBondFENE>(unsigned int N,
                                 unsigned int num_dybond_particles,
                                 const Scalar4* d_pos,
                                 const idx* d_rtag,
                                 const idx* d_tag,
                                 const idx* d_n_neigh,
                                 const idx* d_nl,
                                 const size_t* d_head_list,
                                 const idx* d_rank,
                                 const idx* d_cluster,
                                 const idx* d_dybondParticleTags,
                                 const DyBondParams<md::EvaluatorBondFENE>* d_params,
                                 const BoxDim box,
                                 idx* d_cand_partner,
                                 Scalar* d_cand_dist,
                                 unsigned int block_size,
                                 hipStream_t stream);

template hipError_t gpu_dybond_candidates<md::EvaluatorBondHarmonic>(unsigned int N,
                                 unsigned int num_dybond_particles,
                                 const Scalar4* d_pos,
                                 const idx* d_rtag,
                                 const idx* d_tag,
                                 const idx* d_n_neigh,
                                 const idx* d_nl,
                                 const size_t* d_head_list,
                                 const idx* d_rank,
                                 const idx* d_cluster,
                                 const idx* d_dybondParticleTags,
                                 const DyBondParams<md::EvaluatorBondHarmonic>* d_params,
                                 const BoxDim box,
                                 idx* d_cand_partner,
                                 Scalar* d_cand_dist,
                                 unsigned int block_size,
                                 hipStream_t stream);

template hipError_t gpu_dybond_candidates<md::EvaluatorBondTether>(unsigned int N,
                                 unsigned int num_dybond_particles,
                                 const Scalar4* d_pos,
                                 const idx* d_rtag,
                                 const idx* d_tag,
                                 const idx* d_n_neigh,
                                 const idx* d_nl,
                                 const size_t* d_head_list,
                                 const idx* d_rank,
                                 const idx* d_cluster,
                                 const idx* d_dybondParticleTags,
                                 const DyBondParams<md::EvaluatorBondTether>* d_params,
                                 const BoxDim box,
                                 idx* d_cand_partner,
                                 Scalar* d_cand_dist,
                                 unsigned int block_size,
                                 hipStream_t stream);

template hipError_t gpu_dybond_bind_accept<md::EvaluatorBondFENE>(unsigned int n_prop,
                                  unsigned int num_dybond_particles,
                                  const idx* d_pf,
                                  const idx* d_pt,
                                  const Scalar* d_cand_dist,
                                  idx* d_rank,
                                  const Scalar* d_properties,
                                  Scalar translational_dof,
                                  const DyBondParams<md::EvaluatorBondFENE>* d_params,
                                  uint64_t timestep,
                                  int2* d_bondsToMake,
                                  idx* d_numBondsToMake,
                                  unsigned int block_size,
                                  hipStream_t stream);

template hipError_t gpu_dybond_bind_accept<md::EvaluatorBondHarmonic>(unsigned int n_prop,
                                  unsigned int num_dybond_particles,
                                  const idx* d_pf,
                                  const idx* d_pt,
                                  const Scalar* d_cand_dist,
                                  idx* d_rank,
                                  const Scalar* d_properties,
                                  Scalar translational_dof,
                                  const DyBondParams<md::EvaluatorBondHarmonic>* d_params,
                                  uint64_t timestep,
                                  int2* d_bondsToMake,
                                  idx* d_numBondsToMake,
                                  unsigned int block_size,
                                  hipStream_t stream);

template hipError_t gpu_dybond_bind_accept<md::EvaluatorBondTether>(unsigned int n_prop,
                                  unsigned int num_dybond_particles,
                                  const idx* d_pf,
                                  const idx* d_pt,
                                  const Scalar* d_cand_dist,
                                  idx* d_rank,
                                  const Scalar* d_properties,
                                  Scalar translational_dof,
                                  const DyBondParams<md::EvaluatorBondTether>* d_params,
                                  uint64_t timestep,
                                  int2* d_bondsToMake,
                                  idx* d_numBondsToMake,
                                  unsigned int block_size,
                                  hipStream_t stream);

__global__ void gpu_scatter_ranks_kernel(idx* d_bond_ranks,
                                         const idx* d_actually_bonded, idx num_bonded,
                                         const idx* d_actually_unbonded, idx num_unbonded)
    {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bonded)
        {
        d_bond_ranks[d_actually_bonded[i]] = 1;
        }
    if (i < num_unbonded)
        {
        d_bond_ranks[d_actually_unbonded[i]] = 0;
        }
    }

void gpu_scatter_ranks(idx* d_bond_ranks,
                       const idx* d_actually_bonded, idx num_bonded,
                       const idx* d_actually_unbonded, idx num_unbonded,
                       hipStream_t stream)
    {
    if (num_bonded == 0 && num_unbonded == 0) return;
    idx max_threads = num_bonded > num_unbonded ? num_bonded : num_unbonded;
    int block_size = 256;
    dim3 grid((int)ceil((double)max_threads / (double)block_size), 1, 1);
    if (grid.x == 0) grid.x = 1;
    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL(gpu_scatter_ranks_kernel,
                       dim3(grid),
                       dim3(threads),
                       0,
                       stream,
                       d_bond_ranks,
                       d_actually_bonded, num_bonded,
                       d_actually_unbonded, num_unbonded);
    }

    } // namespace kernel
    } // namespace hoomd
