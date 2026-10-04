// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::conformance {

class Sha256 {
  public:
    void update(const std::uint8_t *data, std::size_t size) {
        bit_count_ += static_cast<std::uint64_t>(size) * 8U;
        while (size != 0U) {
            const std::size_t copy = std::min(size, block_.size() - used_);
            std::copy_n(data, copy, block_.data() + used_);
            used_ += copy;
            data += copy;
            size -= copy;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0U;
            }
        }
    }

    [[nodiscard]] std::string finish() {
        const std::uint64_t bits = bit_count_;
        block_[used_++] = 0x80U;
        if (used_ > 56U) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_),
                      block_.end(), 0U);
            transform(block_.data());
            used_ = 0U;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_),
                  block_.begin() + 56, 0U);
        for (std::size_t index = 0U; index < 8U; ++index)
            block_[63U - index] = static_cast<std::uint8_t>(bits >> (index * 8U));
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t word : state_) output << std::setw(8) << word;
        return output.str();
    }

  private:
    std::array<std::uint32_t, 8U> state_{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<std::uint8_t, 64U> block_{};
    std::uint64_t bit_count_{};
    std::size_t used_{};

    static constexpr std::array<std::uint32_t, 64U> constants_{
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

    static std::uint32_t rotate(std::uint32_t value, unsigned bits) {
        return (value >> bits) | (value << (32U - bits));
    }

    void transform(const std::uint8_t *data) {
        std::array<std::uint32_t, 64U> words{};
        for (std::size_t index = 0U; index < 16U; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(data[index * 4U]) << 24U) |
                (static_cast<std::uint32_t>(data[index * 4U + 1U]) << 16U) |
                (static_cast<std::uint32_t>(data[index * 4U + 2U]) << 8U) |
                static_cast<std::uint32_t>(data[index * 4U + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t a = words[index - 15U];
            const std::uint32_t b = words[index - 2U];
            const std::uint32_t s0 = rotate(a, 7U) ^ rotate(a, 18U) ^ (a >> 3U);
            const std::uint32_t s1 = rotate(b, 17U) ^ rotate(b, 19U) ^ (b >> 10U);
            words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
        }
        auto [a,b,c,d,e,f,g,h] = state_;
        for (std::size_t index = 0U; index < words.size(); ++index) {
            const std::uint32_t s1 = rotate(e, 6U) ^ rotate(e, 11U) ^ rotate(e, 25U);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t first = h + s1 + choice + constants_[index] + words[index];
            const std::uint32_t s0 = rotate(a, 2U) ^ rotate(a, 13U) ^ rotate(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t second = s0 + majority;
            h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
};

inline std::string sha256(std::string_view value) {
    Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t *>(value.data()),
                  value.size());
    return digest.finish();
}

inline std::string sha256_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    Sha256 digest;
    std::array<char, 64U * 1024U> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto size = static_cast<std::size_t>(input.gcount());
        digest.update(reinterpret_cast<const std::uint8_t *>(buffer.data()), size);
    }
    return digest.finish();
}

} // namespace parallel_mater::conformance
