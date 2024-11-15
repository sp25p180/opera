#pragma once

#include "modulus.h"
#include "hash.h"
#include "defines.h"
#include "globals.h"
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>

namespace phantom {
    // FHE schemes
    enum class scheme_type : std::uint8_t {
        none = 0,
        // // No scheme set; cannot be used for encryption
        bgv = 1,
        // Brakerski-Gentry-Vaikuntanathan scheme
        bfv = 2,
        // Brakerski/Fan-Vercauteren scheme
        ckks = 3 // Cheon-Kim-Kim-Song scheme
    };

    // Techniques for multiplication
    enum class mul_tech_type : std::uint8_t {
        none = 0,
        // No technique set, if the scheme is not BFV
        behz = 1,
        // BEHZ
        hps = 2,
        // HPS
        hps_overq = 3,
        // HPS over Q
        hps_overq_leveled = 4 // HPS over Q leveled
    };

    /**
    The data type to store unique identifiers of encryption parameters.
    */
    using parms_id_type = util::HashFunction::hash_block_type;

    /**
    A parms_id_type value consisting of zeros.
    */
    constexpr parms_id_type parms_id_zero = util::HashFunction::hash_zero_block;

    class EncryptionParameters {
        friend class PhantomCPUContext;

    public:
        // Creates an empty set of encryption parameters.
        explicit EncryptionParameters(scheme_type scheme = scheme_type::none) : scheme_(scheme) {
            // set default mul tech for BFV
            if (scheme_ == scheme_type::bfv)
                mul_tech_ = mul_tech_type::hps;
            else
                mul_tech_ = mul_tech_type::none;
            compute_parms_id();
        }

        /**
        Creates an empty set of encryption parameters.

        @param[in] scheme The encryption scheme to be used
        @throws std::invalid_argument if scheme is not supported
        */
        explicit EncryptionParameters(std::uint8_t scheme) {
            // Check that a valid scheme is given
            if (!is_valid_scheme(scheme)) {
                throw std::invalid_argument("unsupported scheme");
            }

            scheme_ = static_cast<scheme_type>(scheme);
            // set default mul tech for BFV
            if (scheme_ == scheme_type::bfv)
                mul_tech_ = mul_tech_type::hps;
            compute_parms_id();
        }

        /**
        Creates a copy of a given instance of EncryptionParameters.

        @param[in] copy The EncryptionParameters to copy from
        */
        EncryptionParameters(const EncryptionParameters &copy) = default;

        /**
        Overwrites the EncryptionParameters instance with a copy of a given instance.

        @param[in] assign The EncryptionParameters to copy from
        */
        EncryptionParameters &operator=(const EncryptionParameters &assign) = default;

        /**
        Creates a new EncryptionParameters instance by moving a given instance.

        @param[in] source The EncryptionParameters to move from
        */
        EncryptionParameters(EncryptionParameters &&source) = default;

        /**
        Overwrites the EncryptionParameters instance by moving a given instance.

        @param[in] assign The EncryptionParameters to move from
        */
        EncryptionParameters &operator=(EncryptionParameters &&assign) = default;

        inline void set_mul_tech(mul_tech_type mul_tech) {
            if (scheme_ == scheme_type::bfv) {
                if (is_valid_mul_tech(mul_tech)) {
                    mul_tech_ = mul_tech;
                }
                else {
                    throw std::invalid_argument("unsupported multiplication technique for BFV");
                }
            }
            else {
                throw std::invalid_argument("mul_tech selection is only supported for BFV");
            }
        }

        /**
        Sets the degree of the polynomial modulus parameter.
        The polynomial modulus must be a power of 2 (e.g.  1024, 2048, 4096, 8192, 16384, or 32768).

        @param[in] poly_modulus_degree The new polynomial modulus degree
        @throws std::logic_error if a valid scheme is not set and poly_modulus_degree is non-zero
        */
        inline void set_poly_modulus_degree(std::size_t poly_modulus_degree) {
            if (scheme_ == scheme_type::none && poly_modulus_degree) {
                throw std::logic_error("poly_modulus_degree is not supported for this scheme");
            }

            // Set the degree
            poly_modulus_degree_ = poly_modulus_degree;

            // Re-compute the parms_id
            compute_parms_id();
        }

        /**
         * Sets the size of special modulus
         * @param[in] special_modulus_size size of special modulus, used for hybrid key-switching
         * @throws std::logic_error if a valid scheme is not set and special_modulus_size is non-zero
        */
        inline void set_special_modulus_size(std::size_t special_modulus_size) {
            if (scheme_ == scheme_type::none && special_modulus_size) {
                throw std::logic_error("special_modulus_size is not supported for this scheme");
            }

            special_modulus_size_ = special_modulus_size;

            // Re-compute the parms_id
            compute_parms_id();
        }

        inline void set_galois_elts(const std::vector<uint32_t> &galois_elts) {
            galois_elts_ = galois_elts;
        }

        /**
        Sets the coefficient modulus parameter, consists of a set of primes.
        Each prime must be at most 60 bits,
        and must be congruent to 1 modulo 2*poly_modulus_degree.

        @param[in] coeff_modulus The new coefficient modulus
        @throws std::logic_error if a valid scheme is not set and coeff_modulus is
        is non-empty
        @throws std::invalid_argument if size of coeff_modulus is invalid
        */
        inline void set_coeff_modulus(const std::vector<Modulus> &coeff_modulus) {
            // Check that a scheme is set
            if (scheme_ == scheme_type::none) {
                if (!coeff_modulus.empty()) {
                    throw std::logic_error("coeff_modulus is not supported for this scheme");
                }
            }
            else if (coeff_modulus.size() > COEFF_MOD_COUNT_MAX ||
                     coeff_modulus.size() < COEFF_MOD_COUNT_MIN) {
                throw std::invalid_argument("coeff_modulus is invalid");
            }

            // only copy coeff_modulus to key_modulus_ since it is the first time setting params
            if (coeff_modulus_.empty())
                key_modulus_ = coeff_modulus;

            coeff_modulus_ = coeff_modulus;

            // Re-compute the parms_id
            compute_parms_id();
        }

        /**
        Sets the plaintext modulus parameter.
        The plaintext modulus is an integer
        The plaintext modulus can be at most 60 bits long, but can otherwise be any integer.
        Note, however, batching require plaintext modulus is congruent to 1 modulo 2N.

        @param[in] plain_modulus The new plaintext modulus
        @throws std::logic_error if scheme is not scheme_type::BFV and plain_modulus
        is non-zero
        */
        inline void set_plain_modulus(const Modulus &plain_modulus) {
            // Check that scheme is BFV
            if (scheme_ != scheme_type::bfv && scheme_ != scheme_type::bgv && !plain_modulus.is_zero()) {
                throw std::logic_error("plain_modulus is not supported for this scheme");
            }

            plain_modulus_ = plain_modulus;

            // Re-compute the parms_id
            compute_parms_id();
        }

        /**
        Sets the plaintext modulus parameter.
        The input std::uint64_t will automatically create the Modulus object.
        The plaintext modulus can be at most 60 bits long, but can otherwise be any integer.
        Note, however, batching require plaintext modulus is congruent to 1 modulo 2N.

        @param[in] plain_modulus The new plaintext modulus
        @throws std::invalid_argument if plain_modulus is invalid
        */
        inline void set_plain_modulus(std::uint64_t plain_modulus) {
            set_plain_modulus(Modulus(plain_modulus));
        }

        // Returns the encryption scheme type.
        [[nodiscard]] inline scheme_type scheme() const noexcept {
            return scheme_;
        }

        // Returns the multiplication technique.
        [[nodiscard]] inline mul_tech_type mul_tech() const noexcept {
            return mul_tech_;
        }

        // Returns the degree of the polynomial modulus parameter.
        [[nodiscard]] inline std::size_t poly_modulus_degree() const noexcept {
            return poly_modulus_degree_;
        }

        /**
         * Returns the size of special modulus.
        */
        [[nodiscard]] inline std::size_t special_modulus_size() const noexcept {
            return special_modulus_size_;
        }

        [[nodiscard]] inline auto galois_elts() const noexcept {
            return galois_elts_;
        }

        /**
         * Returns a const reference to key modulus parameter.
         */
        [[nodiscard]] inline const std::vector<Modulus> &key_modulus() const noexcept {
            return key_modulus_;
        }

        // Returns a const reference to the currently set coefficient modulus parameter.
        [[nodiscard]] inline auto coeff_modulus() const noexcept -> const std::vector<Modulus> & {
            return coeff_modulus_;
        }

        // Returns a const reference to the currently set coefficient modulus parameter.
        [[nodiscard]] inline auto coeff_modulus() noexcept -> std::vector<Modulus> & {
            return coeff_modulus_;
        }

        // Returns a const reference to the currently set plaintext modulus parameter.
        [[nodiscard]] inline const Modulus &plain_modulus() const noexcept {
            return plain_modulus_;
        }

        /**
        @parms[in] other The EncryptionParameters to compare against
        */
        [[nodiscard]] inline bool operator==(const EncryptionParameters &other) const noexcept {
            return (parms_id_ == other.parms_id_);
        }

        /**
        Compares a given set of encryption parameters to the current set of
        encryption parameters. The comparison is performed by comparing
        parms_ids of the parameter sets rather than comparing the parameters
        individually.

        @parms[in] other The EncryptionParameters to compare against
        */
        [[nodiscard]] inline bool operator!=(const EncryptionParameters &other) const noexcept {
            return (parms_id_ != other.parms_id_);
        }

        /**
        Enables access to private members of phantom::EncryptionParameters for C
        */
        struct EncryptionParametersPrivateHelper;

        /**
        Returns the parms_id of the current parameters. This function is intended
        for internal use.
        */
        [[nodiscard]] inline auto &parms_id() const noexcept {
            return parms_id_;
        }

        inline void update_parms_id() {
            compute_parms_id();
        }

        void save(std::ostream &stream) {
            save_members(stream);
        }

        void load(std::istream &stream) {
            load_members(stream);
        }

    private:
        [[nodiscard]] static bool is_valid_scheme(std::uint8_t scheme) noexcept {
            switch (scheme) {
                case static_cast<std::uint8_t>(scheme_type::none):
                case static_cast<std::uint8_t>(scheme_type::bfv):
                case static_cast<std::uint8_t>(scheme_type::ckks):
                case static_cast<std::uint8_t>(scheme_type::bgv):
                    return true;

                default:
                    return false;
            }
        }

        [[nodiscard]] static bool is_valid_mul_tech(mul_tech_type mul_tech) noexcept {
            switch (mul_tech) {
                case mul_tech_type::behz:
                case mul_tech_type::hps:
                case mul_tech_type::hps_overq:
                case mul_tech_type::hps_overq_leveled:
                    return true;

                default:
                    return false;
            }
        }

        inline void compute_parms_id() {
            size_t coeff_modulus_size = coeff_modulus_.size();
            size_t key_modulus_size = key_modulus_.size();

            size_t total_uint64_count = size_t(1) + // scheme
                                        size_t(1) + // poly_modulus_degree
                                        size_t(1) + // special_modulus_size
                                        key_modulus_size + // key_modulus
                                        coeff_modulus_size + plain_modulus_.uint64_count();

            std::vector<uint64_t> param_data;
            param_data.resize(total_uint64_count);
            uint64_t *param_data_ptr = param_data.data();

            // Write the scheme identifier
            *param_data_ptr++ = static_cast<uint64_t>(scheme_);

            // Write the poly_modulus_degree. Note that it will always be positive.
            *param_data_ptr++ = static_cast<uint64_t>(poly_modulus_degree_);

            // Write the special_modulus_size
            *param_data_ptr++ = static_cast<uint64_t>(special_modulus_size_);

            for (const auto &mod: key_modulus_) {
                *param_data_ptr++ = mod.value();
            }

            for (const auto &mod: coeff_modulus_) {
                *param_data_ptr++ = mod.value();
            }

            if (plain_modulus_.uint64_count() == 1) {
                *param_data_ptr++ = plain_modulus_.value();
            }

            phantom::util::HashFunction::hash(param_data.data(), total_uint64_count, parms_id_);

            // Did we somehow manage to get a zero block as result? This is reserved for
            // plaintexts to indicate non-NTT-transformed form.
            if (parms_id_ == parms_id_zero) {
                throw std::logic_error("parms_id cannot be zero");
            }
        }

        void save_members(std::ostream &stream) const {
            // Throw exceptions on std::ios_base::badbit and std::ios_base::failbit
            auto old_except_mask = stream.exceptions();
            try {
                stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

                uint64_t poly_modulus_degree64 = static_cast<uint64_t>(poly_modulus_degree_);
                uint64_t coeff_modulus_size64 = static_cast<uint64_t>(coeff_modulus_.size());
                uint8_t scheme = static_cast<uint8_t>(scheme_);

                stream.write(reinterpret_cast<const char *>(&scheme), sizeof(uint8_t));
                stream.write(reinterpret_cast<const char *>(&poly_modulus_degree64), sizeof(uint64_t));
                stream.write(reinterpret_cast<const char *>(&coeff_modulus_size64), sizeof(uint64_t));
                for (const auto &mod: coeff_modulus_) {
                    mod.save(stream);
                }
                // Only BFV uses plain_modulus but save it in any case for simplicity
                plain_modulus_.save(stream);
            }
            catch (const std::ios_base::failure &) {
                stream.exceptions(old_except_mask);
                throw std::runtime_error("I/O error");
            }
            catch (...) {
                stream.exceptions(old_except_mask);
                throw;
            }
            stream.exceptions(old_except_mask);
        }

        void load_members(std::istream &stream) {
            auto old_except_mask = stream.exceptions();
            try {
                stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

                // Read the scheme identifier
                uint8_t scheme;
                stream.read(reinterpret_cast<char *>(&scheme), sizeof(uint8_t));

                // This constructor will throw if scheme is invalid
                EncryptionParameters parms(scheme);

                // Read the poly_modulus_degree
                uint64_t poly_modulus_degree64 = 0;
                stream.read(reinterpret_cast<char *>(&poly_modulus_degree64), sizeof(uint64_t));

                // Only check for upper bound; lower bound is zero for scheme_type::none
                if (poly_modulus_degree64 > POLY_MOD_DEGREE_MAX) {
                    throw std::logic_error("poly_modulus_degree is invalid");
                }

                // Read the coeff_modulus size
                uint64_t coeff_modulus_size64 = 0;
                stream.read(reinterpret_cast<char *>(&coeff_modulus_size64), sizeof(uint64_t));

                // Only check for upper bound; lower bound is zero for scheme_type::none
                if (coeff_modulus_size64 > COEFF_MOD_COUNT_MAX) {
                    throw std::logic_error("coeff_modulus is invalid");
                }

                // Read the coeff_modulus
                std::vector<Modulus> coeff_modulus;
                for (uint64_t i = 0; i < coeff_modulus_size64; i++) {
                    coeff_modulus.emplace_back();
                    coeff_modulus.back().load(stream);
                }

                // Read the plain_modulus
                Modulus plain_modulus;
                plain_modulus.load(stream);

                // Supposedly everything worked so set the values of member variables
                parms.set_poly_modulus_degree(static_cast<size_t>(poly_modulus_degree64));
                parms.set_coeff_modulus(coeff_modulus);

                // Only BFV uses plain_modulus; set_plain_modulus checks that for
                // other schemes it is zero
                parms.set_plain_modulus(plain_modulus);

                // Set the loaded parameters
                std::swap(*this, parms);

                stream.exceptions(old_except_mask);
            }
            catch (const std::ios_base::failure &) {
                stream.exceptions(old_except_mask);
                throw std::runtime_error("I/O error");
            }
            catch (...) {
                stream.exceptions(old_except_mask);
                throw;
            }
            stream.exceptions(old_except_mask);
        }

        scheme_type scheme_;

        mul_tech_type mul_tech_;

        std::size_t poly_modulus_degree_ = 0;

        // used for hybrid key-switching
        // default is 1
        std::size_t special_modulus_size_ = 1;

        // used for hybrid key-switching
        std::vector<Modulus> key_modulus_{};

        std::vector<Modulus> coeff_modulus_{};

        std::vector<uint32_t> galois_elts_{};

        Modulus plain_modulus_{};

        parms_id_type parms_id_ = parms_id_zero;
    };
}
