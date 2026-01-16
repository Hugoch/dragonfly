// Copyright 2025, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#pragma once

#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "base/pmr/memory_resource.h"

namespace dfly {

/// Top-K implementation using the HeavyKeeper algorithm.
/// This probabilistic data structure efficiently tracks the K most frequent items
/// in a data stream with high accuracy and low memory overhead.
/// Compatible with Redis TOPK commands from RedisBloom module.
class TOPK {
 public:
  // Create a Top-K sketch with specified parameters.
  // k: number of top items to track
  // width: number of counters per row (hash table buckets)
  // depth: number of rows (hash functions)
  // decay: probability decay constant for exponential decay (0.0-1.0)
  TOPK(uint32_t k, uint32_t width, uint32_t depth, double decay,
       PMR_NS::memory_resource* mr = nullptr);

  TOPK(const TOPK&) = delete;
  TOPK& operator=(const TOPK&) = delete;

  TOPK(TOPK&& other) noexcept;
  TOPK& operator=(TOPK&& other) noexcept;

  ~TOPK() = default;

  // Represents an item in the Top-K list with its estimated count
  struct TopKItem {
    std::string item;
    uint32_t count;
  };

  // Add an item to the sketch.
  // Returns the expelled item if one was removed from Top-K, or empty vector.
  std::vector<std::string> Add(std::string_view item);

  // Add multiple items to the sketch.
  // Returns a vector where each element is either an expelled item or empty string.
  std::vector<std::optional<std::string>> AddMultiple(
      const std::vector<std::string_view>& items);

  // Increment an item's count by the specified amount.
  // Returns the expelled item if one was removed, or empty vector.
  // increment must be between 1 and 100,000
  std::vector<std::string> IncrBy(std::string_view item, uint32_t increment);

  // Increment multiple items by specified amounts.
  // Returns a vector where each element is either an expelled item or empty string.
  std::vector<std::optional<std::string>> IncrByMultiple(
      const std::vector<std::pair<std::string_view, uint32_t>>& items);

  // Query if items are in the Top-K list.
  // Returns 1 if item is in Top-K, 0 otherwise.
  std::vector<int> Query(const std::vector<std::string_view>& items) const;

  // Get estimated counts for items.
  // Note: Counts are probabilistic and may be lower than actual.
  // This command is deprecated in Redis but still functional.
  std::vector<uint32_t> Count(const std::vector<std::string_view>& items) const;

  // Get the Top-K items list, optionally with counts.
  // Items are sorted by estimated frequency (highest first).
  std::vector<TopKItem> List() const;

  // Accessors for Top-K parameters
  uint32_t K() const {
    return k_;
  }

  uint32_t Width() const {
    return width_;
  }

  uint32_t Depth() const {
    return depth_;
  }

  double Decay() const {
    return decay_;
  }

  // Memory usage in bytes
  size_t MallocUsed() const;

  // Serialization support for RDB persistence
  struct SerializedData {
    uint32_t k;
    uint32_t width;
    uint32_t depth;
    double decay;
    std::vector<TopKItem> heap_items;  // Min heap contents
    std::vector<uint32_t> counters;    // Hash table counters
  };
  SerializedData Serialize() const;
  void Deserialize(const SerializedData& data);

 private:
  // Internal representation of a heap item
  struct HeapItem {
    std::string key;
    uint32_t count;
    size_t hash;  // Pre-computed hash for efficiency

    // Min heap comparator (smallest count at top)
    bool operator>(const HeapItem& other) const {
      return count > other.count;
    }
  };

  // Hash function for bucket selection in row
  uint64_t Hash(std::string_view item, uint32_t row) const;

  // Exponential decay logic - returns true with probability decay^count
  bool ShouldDecay(uint32_t current_count) const;

  // Get the minimum count for an item across all hash table rows
  uint32_t GetMinCount(std::string_view item) const;

  // Update the min heap after count changes
  void UpdateHeap(std::string_view item, uint32_t new_count);

  // Try to expel the minimum item from heap if it's full
  // Returns the expelled item or empty string
  std::string TryExpelMin();

  // Check if an item is in the Top-K heap
  bool IsInHeap(std::string_view item) const;

  // Internal increment logic shared by Add and IncrBy
  std::vector<std::string> IncrementInternal(std::string_view item, uint32_t increment);

  // Compute decay probability using lookup table
  // For counts >= kDecayLookupSize, uses extrapolation formula
  double ComputeDecayProbability(uint32_t count) const;

  // Heap maintenance functions for O(log k) operations
  void HeapifyUp(size_t index);
  void HeapifyDown(size_t index);

  // Data members
  uint32_t k_;      // Number of top items to track
  uint32_t width_;  // Hash table width (buckets per row)
  uint32_t depth_;  // Hash table depth (number of rows)
  double decay_;    // Decay constant (0.0-1.0, typically 0.9)

  // Decay lookup table: pre-calculated decay^i for i=0..255
  // Avoids expensive std::pow() calls during decay probability checks
  static constexpr size_t kDecayLookupSize = 256;
  std::array<double, kDecayLookupSize> decay_lookup_;

  // HeavyKeeper data structures
  // Hash table: width × depth matrix of counters
  std::vector<uint32_t, PMR_NS::polymorphic_allocator<uint32_t>> counters_;

  // Min heap: vector of top-K items maintained as a min heap
  std::vector<HeapItem, PMR_NS::polymorphic_allocator<HeapItem>> min_heap_;

  // Fast lookup: item name -> hash for O(1) "is in top-k" queries
  absl::flat_hash_map<std::string, size_t> item_to_hash_;
};

}  // namespace dfly
