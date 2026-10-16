#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sfm::regional::topology_storage {

class Indices {
public:
    explicit Indices(bool wide = false) : wide_(wide) {}
    struct Reference {
        Indices* owner; size_t index;
        operator uint64_t() const { return owner->get(index); }
        Reference& operator=(uint64_t value) { owner->set(index, value); return *this; }
        Reference& operator=(const Reference& value) { return *this = uint64_t(value); }
    };
    size_t size() const { return wide_ ? values64_.size() : values32_.size(); }
    uint64_t bytes() const { return values32_.capacity() * uint64_t(4) + values64_.capacity() * uint64_t(8); }
    bool wide() const { return wide_; }
    static uint64_t allocationBytes(uint64_t count) {
        const uint64_t width = count > UINT32_MAX ? 8 : 4;
        if (count > UINT64_MAX / width) throw std::length_error("topology index byte count overflow");
        return count * width;
    }
    void resize(size_t count) {
        if (count > UINT32_MAX && !wide_) {
            std::vector<uint64_t> converted(count);
            std::copy(values32_.begin(), values32_.end(), converted.begin());
            values64_ = std::move(converted); std::vector<uint32_t>().swap(values32_); wide_ = true;
        } else if (wide_) {
            if (count > values64_.capacity()) { std::vector<uint64_t> grown(count); std::copy(values64_.begin(), values64_.end(), grown.begin()); values64_.swap(grown); }
            else values64_.resize(count);
        } else {
            if (count > values32_.capacity()) { std::vector<uint32_t> grown(count); std::copy(values32_.begin(), values32_.end(), grown.begin()); values32_.swap(grown); }
            else values32_.resize(count);
        }
    }
    void clear() { values32_.clear(); values64_.clear(); wide_ = false; }
    void shrink_to_fit() { values32_.shrink_to_fit(); values64_.shrink_to_fit(); }
    uint64_t at(size_t index) const { return get(index); }
    Reference at(size_t index) { if (index >= size()) throw std::out_of_range("topology index"); return {this, index}; }
    uint64_t operator[](size_t index) const { return get(index); }
    Reference operator[](size_t index) { return {this, index}; }
private:
    uint64_t get(size_t index) const { return wide_ ? values64_.at(index) : values32_.at(index); }
    void set(size_t index, uint64_t value) {
        if (!wide_ && value > UINT32_MAX) throw std::overflow_error("topology index would truncate a global point identity");
        if (wide_) values64_.at(index) = value; else values32_.at(index) = uint32_t(value);
    }
    bool wide_ = false; std::vector<uint32_t> values32_; std::vector<uint64_t> values64_;
};

class Bits {
public:
    struct Reference {
        Bits* owner; size_t index;
        operator bool() const { return owner->get(index); }
        Reference& operator=(bool value) { owner->set(index, value); return *this; }
        Reference& operator=(const Reference& value) { return *this = bool(value); }
        Reference& operator|=(bool value) { if (value) owner->set(index, true); return *this; }
    };
    explicit Bits(size_t count = 0) { resize(count); }
    size_t size() const { return size_; }
    static uint64_t allocationBytes(uint64_t count) { return (count / 64 + (count % 64 != 0)) * 8; }
    uint64_t bytes() const { return words_.capacity() * uint64_t(8); }
    void resize(size_t count) { size_ = count; words_.assign(count / 64 + (count % 64 != 0), 0); }
    void release() { size_ = 0; std::vector<uint64_t>().swap(words_); }
    bool at(size_t index) const { return get(index); }
    Reference at(size_t index) { if (index >= size_) throw std::out_of_range("topology bitmap"); return {this, index}; }
    bool operator[](size_t index) const { return get(index); }
    Reference operator[](size_t index) { return {this, index}; }
    const std::vector<uint64_t>& words() const { return words_; }
private:
    bool get(size_t index) const { if (index >= size_) throw std::out_of_range("topology bitmap"); return (words_[index / 64] >> (index % 64)) & 1; }
    void set(size_t index, bool value) {
        if (index >= size_) throw std::out_of_range("topology bitmap");
        const auto mask = uint64_t(1) << (index % 64); auto& word = words_[index / 64];
        if (value) word |= mask; else word &= ~mask;
    }
    size_t size_ = 0; std::vector<uint64_t> words_;
};

class RankedBits : public Bits {
public:
    static uint64_t allocationBytes(uint64_t count) { return Bits::allocationBytes(count) + (count / 512 + 1) * 8; }
    uint64_t bytes() const { return Bits::bytes() + ranks_.capacity() * uint64_t(8); }
    void buildRanks() {
        ranks_.assign(words().size() / 8 + 1, 0); uint64_t count = 0;
        for (size_t word = 0; word < words().size(); ++word) { if (word % 8 == 0) ranks_[word / 8] = count; count += population(words()[word]); }
        if (words().size() % 8 == 0) ranks_.back() = count;
    }
    uint64_t rank(size_t index) const {
        if (index >= size()) throw std::out_of_range("topology shared-root rank");
        const size_t word = index / 64, first = word / 8 * 8; uint64_t count = ranks_.at(word / 8);
        for (size_t i = first; i < word; ++i) count += population(words()[i]);
        const uint64_t mask = index % 64 ? (uint64_t(1) << (index % 64)) - 1 : 0;
        return count + population(words()[word] & mask);
    }
    void release() { Bits::release(); std::vector<uint64_t>().swap(ranks_); }
private:
    static uint64_t population(uint64_t value) {
        value -= (value >> 1) & 0x5555555555555555ull;
        value = (value & 0x3333333333333333ull) + ((value >> 2) & 0x3333333333333333ull);
        value = (value + (value >> 4)) & 0x0f0f0f0f0f0f0f0full;
        return (value * 0x0101010101010101ull) >> 56;
    }
    std::vector<uint64_t> ranks_;
};

template<class T> class Chunks {
public:
    static constexpr size_t block = 4096;
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag; using value_type = T; using difference_type = ptrdiff_t; using pointer = T*; using reference = T&;
        Iterator(Chunks* table = nullptr, size_t index = 0) : table_(table), index_(index) {}
        reference operator*() const { return table_->at(index_); }
        pointer operator->() const { return &**this; }
        reference operator[](difference_type step) const { return *(*this + step); }
        Iterator& operator++() { ++index_; return *this; } Iterator operator++(int) { auto old = *this; ++*this; return old; }
        Iterator& operator--() { --index_; return *this; } Iterator operator--(int) { auto old = *this; --*this; return old; }
        Iterator& operator+=(difference_type step) { index_ = size_t(difference_type(index_) + step); return *this; }
        Iterator& operator-=(difference_type step) { return *this += -step; }
        Iterator operator+(difference_type step) const { auto result = *this; return result += step; }
        Iterator operator-(difference_type step) const { auto result = *this; return result -= step; }
        friend Iterator operator+(difference_type step, Iterator value) { return value += step; }
        difference_type operator-(const Iterator& other) const { return difference_type(index_) - difference_type(other.index_); }
        bool operator==(const Iterator& other) const { return index_ == other.index_; } bool operator!=(const Iterator& other) const { return !(*this == other); }
        bool operator<(const Iterator& other) const { return index_ < other.index_; } bool operator>(const Iterator& other) const { return other < *this; }
        bool operator<=(const Iterator& other) const { return !(other < *this); } bool operator>=(const Iterator& other) const { return !(*this < other); }
    private:
        Chunks* table_; size_t index_;
    };
    size_t size() const { return size_; }
    size_t capacity() const { return chunks_.size() * block; }
    uint64_t bytes() const { return capacity() * uint64_t(sizeof(T)) + chunks_.capacity() * uint64_t(sizeof(std::unique_ptr<T[]>)); }
    void reserve(size_t count) { while (capacity() < count) chunks_.push_back(std::make_unique<T[]>(block)); }
    void push_back(const T& value) { if (size_ == capacity()) reserve(size_ + 1); atUnchecked(size_++) = value; }
    void clear() { chunks_.clear(); size_ = 0; }
    void release() { clear(); chunks_.shrink_to_fit(); }
    T& at(size_t index) { if (index >= size_) throw std::out_of_range("topology group"); return atUnchecked(index); }
    const T& at(size_t index) const { if (index >= size_) throw std::out_of_range("topology group"); return atUnchecked(index); }
    T& operator[](size_t index) { return at(index); } const T& operator[](size_t index) const { return at(index); }
    Iterator begin() { return {this, 0}; } Iterator end() { return {this, size_}; }
private:
    T& atUnchecked(size_t index) { return chunks_.at(index / block)[index % block]; }
    const T& atUnchecked(size_t index) const { return chunks_.at(index / block)[index % block]; }
    size_t size_ = 0; std::vector<std::unique_ptr<T[]>> chunks_;
};

} // namespace sfm::regional::topology_storage
