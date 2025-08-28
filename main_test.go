package main

import (
	"context"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
)

func TestRedisConnection(t *testing.T) {
	client := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "",
		DB:       0,
	})
	defer client.Close()

	ctx := context.Background()
	
	pong, err := client.Ping(ctx).Result()
	if err != nil {
		t.Fatalf("Failed to connect to Redis: %v", err)
	}
	
	if pong != "PONG" {
		t.Fatalf("Expected PONG, got %s", pong)
	}
	
	t.Log("Redis connection successful")
}

func TestSemanticCacheBasicOperations(t *testing.T) {
	config := Config{
		RedisAddr:     "localhost:6379",
		RedisPassword: "",
		RedisDB:       1, // Use a different DB for testing
		OpenAIToken:   "test-token", // Mock token for testing
		Threshold:     0.8,
		CachePrefix:   "test_cache",
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before testing
	err := cache.Clear(ctx)
	if err != nil {
		t.Logf("Warning: Failed to clear cache before test: %v", err)
	}

	// Test Redis connectivity through the cache
	err = cache.redis.Ping(ctx).Err()
	if err != nil {
		t.Fatalf("Cache Redis connection failed: %v", err)
	}

	t.Log("Semantic cache Redis connection successful")

	// Test basic Redis operations without OpenAI
	testKey := "test:basic"
	testValue := "test value"
	
	err = cache.redis.Set(ctx, testKey, testValue, time.Minute).Err()
	if err != nil {
		t.Fatalf("Failed to set test key: %v", err)
	}

	result, err := cache.redis.Get(ctx, testKey).Result()
	if err != nil {
		t.Fatalf("Failed to get test key: %v", err)
	}

	if result != testValue {
		t.Fatalf("Expected %s, got %s", testValue, result)
	}

	// Clean up
	cache.redis.Del(ctx, testKey)
	
	t.Log("Basic Redis operations through cache successful")
}

func TestCacheStats(t *testing.T) {
	config := Config{
		RedisAddr:     "localhost:6379",
		RedisPassword: "",
		RedisDB:       1,
		OpenAIToken:   "test-token",
		Threshold:     0.8,
		CachePrefix:   "test_stats",
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before testing
	cache.Clear(ctx)

	stats, err := cache.Stats(ctx)
	if err != nil {
		t.Fatalf("Failed to get cache stats: %v", err)
	}

	if stats["total_entries"] != int64(0) {
		t.Fatalf("Expected 0 entries, got %v", stats["total_entries"])
	}

	if stats["threshold"] != 0.8 {
		t.Fatalf("Expected threshold 0.8, got %v", stats["threshold"])
	}

	if stats["prefix"] != "test_stats" {
		t.Fatalf("Expected prefix 'test_stats', got %v", stats["prefix"])
	}

	t.Log("Cache stats test successful")
}