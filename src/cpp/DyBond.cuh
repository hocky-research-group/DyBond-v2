#ifndef _DYBOND_CUH_
#define _DYBOND_CUH_

#include <hoomd/BondedGroupData.cuh>
#include <hoomd/ParticleData.cuh>

namespace hoomd
    {
typedef unsigned int idx;
template<class bond_class> struct DyBondParams
    {
    typedef typename bond_class::param_type BondParams;
    idx bond_type;

    Scalar dt;
    idx period;

    Scalar seed;
    idx unbonding_rng_identifier;
    idx bonding_rng_identifier;

    idx numDyBondParticles;
    idx NOT_BONDABLE;

    Scalar kon, koff;
    Scalar rmin, rmax;
    bool metropolisFlag;
    bool nearestNeighborFlag;
    Scalar r0;

    bool forceFlag;
    Scalar force_a;

    bool selfAvoidingFlag;

    bool temperatureFlag;
    Scalar Tmelt;
    Scalar alpha;
    Scalar kon_melt, koff_melt;

    idx* d_typeAs;
    idx* d_typeBs;
    idx numTypePairs;

    BondParams bond_params;
    };

namespace kernel
    {
typedef unsigned int idx;

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
                                     hipStream_t stream = 0);
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
                          hipStream_t stream = 0);

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
                          hipStream_t stream = 0);

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
                                 hipStream_t stream = 0);

hipError_t gpu_dybond_bind_match(unsigned int num_dybond_particles,
                                 const idx* d_dybondParticleTags,
                                 const idx* d_cand_partner,
                                 idx* d_rank,
                                 idx* d_pf,
                                 idx* d_pt,
                                 idx* d_nprop,
                                 unsigned int block_size,
                                 hipStream_t stream = 0);

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
                                  hipStream_t stream = 0);

void gpu_scatter_ranks(idx* d_bond_ranks,
                       const idx* d_actually_bonded, idx num_bonded,
                       const idx* d_actually_unbonded, idx num_unbonded,
                       hipStream_t stream = 0);
    } // namespace kernel
    } // namespace hoomd
#endif
