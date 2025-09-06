package main

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestCircuitBreaker(t *testing.T) {
	cb := &CircuitBreaker{
		state:       CircuitClosed,
		timeout:     100 * time.Millisecond,
		maxFailures: 3,
	}

	// Test initial state
	if !cb.CanExecute() {
		t.Fatal("Circuit breaker should be closed initially")
	}

	// Test failure recording
	for i := 0; i < 3; i++ {
		cb.RecordFailure()
	}

	// Circuit should be open now
	if cb.CanExecute() {
		t.Fatal("Circuit breaker should be open after max failures")
	}

	// Test timeout
	time.Sleep(150 * time.Millisecond)
	if !cb.CanExecute() {
		t.Fatal("Circuit breaker should allow execution after timeout")
	}

	// Test success recording
	cb.RecordSuccess()
	if cb.GetState() != CircuitClosed {
		t.Fatal("Circuit breaker should be closed after success")
	}
}

func TestRetryWithBackoff(t *testing.T) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             6, // Use a different DB for testing
		Threshold:           0.8,
		CachePrefix:         "test_retry",
		MaxRetries:          2,
		RetryBaseDelay:      10 * time.Millisecond,
		RetryMaxDelay:       100 * time.Millisecond,
		CircuitBreakerTimeout: 1 * time.Minute,
		MaxFailures:         5,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Test successful operation
	attempts := 0
	err := cache.retryWithBackoff(ctx, func() error {
		attempts++
		return nil
	})

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if attempts != 1 {
		t.Fatalf("Expected 1 attempt, got %d", attempts)
	}

	// Test failing operation
	attempts = 0
	err = cache.retryWithBackoff(ctx, func() error {
		attempts++
		return errors.New("test error")
	})

	if err == nil {
		t.Fatal("Expected error, got nil")
	}

	if attempts != 3 { // MaxRetries + 1
		t.Fatalf("Expected 3 attempts, got %d", attempts)
	}

	// Test validation error (should not retry)
	attempts = 0
	err = cache.retryWithBackoff(ctx, func() error {
		attempts++
		return &CacheError{
			Type:    ErrorTypeValidation,
			Message: "validation error",
		}
	})

	if err == nil {
		t.Fatal("Expected error, got nil")
	}

	if attempts != 1 {
		t.Fatalf("Expected 1 attempt for validation error, got %d", attempts)
	}
}

func TestCacheError(t *testing.T) {
	// Test error without underlying error
	err := &CacheError{
		Type:    ErrorTypeRedis,
		Message: "Redis connection failed",
	}

	expected := "REDIS_ERROR: Redis connection failed"
	if err.Error() != expected {
		t.Fatalf("Expected '%s', got '%s'", expected, err.Error())
	}

	// Test error with underlying error
	underlyingErr := errors.New("connection timeout")
	err = &CacheError{
		Type:    ErrorTypeRedis,
		Message: "Redis connection failed",
		Err:     underlyingErr,
	}

	expected = "REDIS_ERROR: Redis connection failed (connection timeout)"
	if err.Error() != expected {
		t.Fatalf("Expected '%s', got '%s'", expected, err.Error())
	}

	// Test Unwrap
	if err.Unwrap() != underlyingErr {
		t.Fatal("Unwrap should return the underlying error")
	}
}

func TestErrorHandlingInOperations(t *testing.T) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             7, // Use a different DB for testing
		Threshold:           0.8,
		CachePrefix:         "test_error_handling",
		EmbeddingURL:        "", // Empty URL to trigger validation error
		MaxRetries:          2,
		RetryBaseDelay:      10 * time.Millisecond,
		RetryMaxDelay:       100 * time.Millisecond,
		CircuitBreakerTimeout: 1 * time.Minute,
		MaxFailures:         5,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before testing
	cache.Clear(ctx)

	// Test Set with validation error
	err := cache.Set(ctx, "test key", "test value", time.Hour)
	if err == nil {
		t.Fatal("Expected error for empty embedding URL, got nil")
	}

	if cacheErr, ok := err.(*CacheError); !ok {
		t.Fatal("Expected CacheError type")
	} else if cacheErr.Type != ErrorTypeValidation {
		t.Fatalf("Expected validation error, got %s", cacheErr.Type)
	}

	// Test Get with validation error
	_, _, err = cache.Get(ctx, "test key")
	if err == nil {
		t.Fatal("Expected error for empty embedding URL, got nil")
	}

	if cacheErr, ok := err.(*CacheError); !ok {
		t.Fatal("Expected CacheError type")
	} else if cacheErr.Type != ErrorTypeValidation {
		t.Fatalf("Expected validation error, got %s", cacheErr.Type)
	}
}

func TestCircuitBreakerIntegration(t *testing.T) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             8, // Use a different DB for testing
		Threshold:           0.8,
		CachePrefix:         "test_circuit_breaker",
		EmbeddingURL:        "http://invalid-url:9999/api/embeddings", // Invalid URL
		MaxRetries:          1,
		RetryBaseDelay:      10 * time.Millisecond,
		RetryMaxDelay:       100 * time.Millisecond,
		CircuitBreakerTimeout: 100 * time.Millisecond,
		MaxFailures:         2,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Clean up before testing
	cache.Clear(ctx)

	// Make requests to trigger circuit breaker
	for i := 0; i < 3; i++ {
		_, _, err := cache.Get(ctx, "test key")
		if err == nil {
			t.Fatal("Expected error for invalid URL, got nil")
		}
	}

	// Circuit breaker should be open now
	_, _, err := cache.Get(ctx, "test key")
	if err == nil {
		t.Fatal("Expected circuit breaker error, got nil")
	}

	if cacheErr, ok := err.(*CacheError); !ok {
		t.Fatal("Expected CacheError type")
	} else if cacheErr.Type != ErrorTypeCircuitBreaker {
		t.Fatalf("Expected circuit breaker error, got %s", cacheErr.Type)
	}

	// Wait for circuit breaker timeout
	time.Sleep(150 * time.Millisecond)

	// Should allow execution again (but will still fail due to invalid URL)
	_, _, err = cache.Get(ctx, "test key")
	if err == nil {
		t.Fatal("Expected error for invalid URL, got nil")
	}

	// Should not be circuit breaker error anymore
	if cacheErr, ok := err.(*CacheError); ok && cacheErr.Type == ErrorTypeCircuitBreaker {
		t.Fatal("Circuit breaker should be closed after timeout")
	}
}

func BenchmarkCircuitBreaker(b *testing.B) {
	cb := &CircuitBreaker{
		state:       CircuitClosed,
		timeout:     100 * time.Millisecond,
		maxFailures: 5,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.CanExecute()
	}
}

func BenchmarkRetryWithBackoff(b *testing.B) {
	config := Config{
		RedisAddr:           "localhost:6379",
		RedisPassword:       "",
		RedisDB:             9, // Use a different DB for benchmarking
		Threshold:           0.8,
		CachePrefix:         "benchmark_retry",
		MaxRetries:          3,
		RetryBaseDelay:      1 * time.Millisecond,
		RetryMaxDelay:       10 * time.Millisecond,
		CircuitBreakerTimeout: 1 * time.Minute,
		MaxFailures:         5,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.retryWithBackoff(ctx, func() error {
			return nil // Always succeed for benchmark
		})
	}
}
