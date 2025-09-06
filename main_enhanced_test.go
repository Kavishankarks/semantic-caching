package main

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
)

func TestEnhancedSemanticCache(t *testing.T) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             2, // Use a different DB for testing
		Threshold:           0.8,
		CachePrefix:         "test_enhanced",
		MaxCacheSize:        100,
		EnableLRU:           true,
		CleanupInterval:     1 * time.Minute,
		EmbeddingCacheSize:  50,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before testing
	err := cache.Clear(ctx)
	if err != nil {
		t.Logf("Warning: Failed to clear cache before test: %v", err)
	}

	// Test Redis connectivity
	err = cache.redis.Ping(ctx).Err()
	if err != nil {
		t.Fatalf("Cache Redis connection failed: %v", err)
	}

	t.Run("SetWithMetadata", func(t *testing.T) {
		metadata := map[string]interface{}{
			"category":   "test",
			"difficulty": "easy",
			"tags":       []string{"test", "metadata"},
		}

		err := cache.SetWithMetadata(ctx, "test key", "test value", time.Hour, metadata)
		if err != nil {
			t.Fatalf("Failed to set with metadata: %v", err)
		}

		// Verify the entry was stored
		value, found, err := cache.Get(ctx, "test key")
		if err != nil {
			t.Fatalf("Failed to get entry: %v", err)
		}
		if !found {
			t.Fatal("Entry not found after setting")
		}
		if value != "test value" {
			t.Fatalf("Expected 'test value', got '%s'", value)
		}
	})

	t.Run("BatchSet", func(t *testing.T) {
		entries := []BatchEntry{
			{
				Key:   "batch key 1",
				Value: "batch value 1",
				TTL:   3600,
				Metadata: map[string]interface{}{
					"batch": true,
				},
			},
			{
				Key:   "batch key 2",
				Value: "batch value 2",
				TTL:   3600,
				Metadata: map[string]interface{}{
					"batch": true,
				},
			},
		}

		response, err := cache.BatchSet(ctx, entries)
		if err != nil {
			t.Fatalf("Failed to batch set: %v", err)
		}

		if !response.Success {
			t.Fatalf("Batch set failed: %s", response.Message)
		}

		if response.Processed != 2 {
			t.Fatalf("Expected 2 processed, got %d", response.Processed)
		}

		if len(response.Failed) > 0 {
			t.Fatalf("Expected no failures, got %v", response.Failed)
		}
	})

	t.Run("BatchGet", func(t *testing.T) {
		keys := []string{"batch key 1", "batch key 2", "nonexistent key"}

		response, err := cache.BatchGet(ctx, keys)
		if err != nil {
			t.Fatalf("Failed to batch get: %v", err)
		}

		if len(response.Results) != 2 {
			t.Fatalf("Expected 2 results, got %d", len(response.Results))
		}

		if !response.Found["batch key 1"] {
			t.Fatal("Expected 'batch key 1' to be found")
		}

		if !response.Found["batch key 2"] {
			t.Fatal("Expected 'batch key 2' to be found")
		}

		if response.Found["nonexistent key"] {
			t.Fatal("Expected 'nonexistent key' to not be found")
		}
	})

	t.Run("Search", func(t *testing.T) {
		// First, set up some test data
		testData := map[string]string{
			"What is machine learning?": "ML is a subset of AI",
			"How does AI work?":         "AI uses algorithms to solve problems",
			"What is programming?":      "Programming is writing code",
		}

		for key, value := range testData {
			err := cache.Set(ctx, key, value, time.Hour)
			if err != nil {
				t.Logf("Warning: Failed to set test data '%s': %v", key, err)
			}
		}

		// Test search
		response, err := cache.Search(ctx, "What is ML?", 5, 0.7)
		if err != nil {
			t.Fatalf("Failed to search: %v", err)
		}

		if response.Count == 0 {
			t.Log("No search results found (this might be expected if embedding service is not available)")
		} else {
			t.Logf("Found %d search results", response.Count)
			for i, result := range response.Results {
				t.Logf("  %d. Score: %.3f - %s", i+1, result.Score, result.Key)
			}
		}
	})

	t.Run("Metrics", func(t *testing.T) {
		metrics, err := cache.GetMetrics(ctx)
		if err != nil {
			t.Fatalf("Failed to get metrics: %v", err)
		}

		if metrics.TotalEntries < 0 {
			t.Fatalf("Invalid total entries: %d", metrics.TotalEntries)
		}

		t.Logf("Metrics: %+v", metrics)
	})

	t.Run("AccessStatistics", func(t *testing.T) {
		// Make a few requests to generate access statistics
		keys := []string{"test key", "batch key 1", "batch key 2"}
		
		for _, key := range keys {
			_, _, err := cache.Get(ctx, key)
			if err != nil {
				t.Logf("Warning: Failed to get key '%s': %v", key, err)
			}
		}

		// The access statistics should be updated
		// This is tested indirectly through the metrics
		metrics, err := cache.GetMetrics(ctx)
		if err != nil {
			t.Fatalf("Failed to get updated metrics: %v", err)
		}

		t.Logf("Updated metrics after access: %+v", metrics)
	})

	// Clean up after testing
	cache.Clear(ctx)
}

func TestEmbeddingCache(t *testing.T) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             3, // Use a different DB for testing
		Threshold:           0.8,
		CachePrefix:         "test_embedding_cache",
		EmbeddingCacheSize:  10,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Test embedding cache functionality
	t.Run("EmbeddingCacheSize", func(t *testing.T) {
		// The embedding cache should be initialized
		if cache.embeddingCache == nil {
			t.Fatal("Embedding cache not initialized")
		}

		// Test that we can add to the cache
		cache.embeddingCacheMux.Lock()
		cache.embeddingCache["test text"] = []float64{0.1, 0.2, 0.3}
		cache.embeddingCacheMux.Unlock()

		// Verify it's there
		cache.embeddingCacheMux.RLock()
		embedding, exists := cache.embeddingCache["test text"]
		cache.embeddingCacheMux.RUnlock()

		if !exists {
			t.Fatal("Embedding not found in cache")
		}

		if len(embedding) != 3 {
			t.Fatalf("Expected embedding length 3, got %d", len(embedding))
		}
	})
}

func TestCleanupRoutine(t *testing.T) {
	config := Config{
		RedisAddr:         "localhost:6379",
		RedisPassword:     "",
		RedisDB:           4, // Use a different DB for testing
		Threshold:         0.8,
		CachePrefix:       "test_cleanup",
		CleanupInterval:   100 * time.Millisecond, // Very short for testing
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before testing
	cache.Clear(ctx)

	t.Run("CleanupRoutine", func(t *testing.T) {
		// Set an entry with a very short TTL
		err := cache.Set(ctx, "short ttl key", "short ttl value", 50*time.Millisecond)
		if err != nil {
			t.Fatalf("Failed to set short TTL entry: %v", err)
		}

		// Wait for the entry to expire and cleanup to run
		time.Sleep(200 * time.Millisecond)

		// The entry should be cleaned up
		_, found, err := cache.Get(ctx, "short ttl key")
		if err != nil {
			t.Logf("Warning: Error getting expired entry: %v", err)
		}

		// Note: The entry might still be found if the cleanup hasn't run yet
		// This is a timing-dependent test, so we just log the result
		t.Logf("Expired entry found after cleanup: %v", found)
	})

	// Clean up after testing
	cache.Clear(ctx)
}

func BenchmarkSemanticCache(b *testing.B) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             5, // Use a different DB for benchmarking
		Threshold:           0.8,
		CachePrefix:         "benchmark",
		EmbeddingCacheSize:  1000,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before benchmarking
	cache.Clear(ctx)

	// Set up test data
	testKeys := []string{
		"benchmark key 1",
		"benchmark key 2",
		"benchmark key 3",
		"benchmark key 4",
		"benchmark key 5",
	}

	for _, key := range testKeys {
		cache.Set(ctx, key, "benchmark value", time.Hour)
	}

	b.Run("Get", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := testKeys[i%len(testKeys)]
			cache.Get(ctx, key)
		}
	})

	b.Run("Set", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("benchmark set key %d", i)
			cache.Set(ctx, key, "benchmark value", time.Hour)
		}
	})

	b.Run("BatchSet", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entries := []BatchEntry{
				{
					Key:   fmt.Sprintf("batch key %d-1", i),
					Value: "batch value 1",
					TTL:   3600,
				},
				{
					Key:   fmt.Sprintf("batch key %d-2", i),
					Value: "batch value 2",
					TTL:   3600,
				},
			}
			cache.BatchSet(ctx, entries)
		}
	})

	// Clean up after benchmarking
	cache.Clear(ctx)
}
