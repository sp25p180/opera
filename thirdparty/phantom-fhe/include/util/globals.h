// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "hestdparms.h"
#include "common.h"
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

namespace phantom {
    class Modulus;

    namespace util {
        class MemoryPoolMT;
        class MemoryPoolST;

        namespace global_variables {
            //            extern const size_t global_rand_gen_GPU_threads;
            //            extern const size_t global_forward_GPU_threads;
            //            extern const size_t global_switch_point;
            //            extern const size_t global_maxthreads;
            //
            //            extern const size_t global_forward_balance;
            //            extern const size_t global_backward_GPU_threads;
            //            extern const size_t global_backward_balance;

            /**
            Default value for the standard deviation of the noise (error) distribution.
            */
            constexpr std::size_t prng_seed_uint64_count = 8;
            constexpr std::size_t prng_seed_byte_count = prng_seed_uint64_count * phantom::util::bytes_per_uint64;

            constexpr double noise_standard_deviation = distributionParameter;

            constexpr double noise_distribution_width_multiplier = 6;

            constexpr double noise_max_deviation = noise_standard_deviation * noise_distribution_width_multiplier;

            /**
            This data structure is a key-value storage that maps degrees of the polynomial modulus
            to vectors of Modulus elements so that when used with the default value for the
            standard deviation of the noise distribution (noise_standard_deviation), the security
            level is at least 128 bits according to https://HomomorphicEncryption.org. This makes
            it easy for non-expert users to select secure parameters.
            */
            const std::map<std::size_t, std::vector<Modulus>> &GetDefaultCoeffModulus128();

            /**
            This data structure is a key-value storage that maps degrees of the polynomial modulus
            to vectors of Modulus elements so that when used with the default value for the
            standard deviation of the noise distribution (noise_standard_deviation), the security
            level is at least 192 bits according to https://HomomorphicEncryption.org. This makes
            it easy for non-expert users to select secure parameters.
            */
            const std::map<std::size_t, std::vector<Modulus>> &GetDefaultCoeffModulus192();

            /**
            This data structure is a key-value storage that maps degrees of the polynomial modulus
            to vectors of Modulus elements so that when used with the default value for the
            standard deviation of the noise distribution (noise_standard_deviation), the security
            level is at least 256 bits according to https://HomomorphicEncryption.org. This makes
            it easy for non-expert users to select secure parameters.
            */
            const std::map<std::size_t, std::vector<Modulus>> &GetDefaultCoeffModulus256();

            // Global memory pool for multi-thread safe case
            extern const std::shared_ptr<MemoryPoolST> global_memory_pool;
        }
    }
}
