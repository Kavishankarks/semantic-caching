package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-redis/redis/v8"
)

type CacheEntry struct {
	Key         string    `json:"key"`
	Value       string    `json:"value"`
	Embedding   []float64 `json:"embedding"`
	Timestamp   time.Time `json:"timestamp"`
	TTL         int64     `json:"ttl"`
	AccessCount int64     `json:"access_count"`
	LastAccess  time.Time `json:"last_access"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

type CacheGetRequest struct {
	Key string `json:"key"`
}

type CacheGetResponse struct {
	Value string `json:"value"`
	Found bool   `json:"found"`
}

type CacheSetRequest struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	TTL   int64  `json:"ttl"`
}

type CacheSetResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
}

type CacheDeleteRequest struct {
	Key string `json:"key"`
}

type CacheDeleteResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
}

// Batch operation types
type BatchSetRequest struct {
	Entries []BatchEntry `json:"entries"`
}

type BatchEntry struct {
	Key      string                 `json:"key"`
	Value    string                 `json:"value"`
	TTL      int64                  `json:"ttl"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type BatchSetResponse struct {
	Success   bool     `json:"success"`
	Processed int      `json:"processed"`
	Failed    []string `json:"failed,omitempty"`
	Message   string   `json:"message,omitempty"`
}

type BatchGetRequest struct {
	Keys []string `json:"keys"`
}

type BatchGetResponse struct {
	Results map[string]string `json:"results"`
	Found   map[string]bool   `json:"found"`
}

// Enhanced similarity types
type SimilarityResult struct {
	Key       string  `json:"key"`
	Value     string  `json:"value"`
	Score     float64 `json:"score"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type SearchRequest struct {
	Query     string  `json:"query"`
	Limit     int     `json:"limit,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
}

type SearchResponse struct {
	Results []SimilarityResult `json:"results"`
	Count   int                `json:"count"`
}

// Metrics and monitoring types
type CacheMetrics struct {
	TotalEntries     int64   `json:"total_entries"`
	HitRate          float64 `json:"hit_rate"`
	AverageResponseTime time.Duration `json:"average_response_time"`
	EmbeddingCacheHitRate float64 `json:"embedding_cache_hit_rate"`
	MemoryUsage      int64   `json:"memory_usage_bytes"`
	LastCleanup      time.Time `json:"last_cleanup"`
}

// Embedding provider interface
type EmbeddingProvider interface {
	GetEmbedding(ctx context.Context, text string) ([]float64, error)
	GetBatchEmbeddings(ctx context.Context, texts []string) ([][]float64, error)
}

// Circuit breaker states
type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
	CircuitHalfOpen
)

// Circuit breaker for embedding service
type CircuitBreaker struct {
	state         CircuitState
	failureCount  int64
	lastFailTime  time.Time
	timeout       time.Duration
	maxFailures   int64
	mutex         sync.RWMutex
}

// Retry configuration
type RetryConfig struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
	Multiplier float64
}

// Error types for better error handling
type CacheError struct {
	Type    string
	Message string
	Err     error
}

func (e *CacheError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s (%v)", e.Type, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

func (e *CacheError) Unwrap() error {
	return e.Err
}

// Error types
const (
	ErrorTypeEmbedding     = "EMBEDDING_ERROR"
	ErrorTypeRedis         = "REDIS_ERROR"
	ErrorTypeCircuitBreaker = "CIRCUIT_BREAKER_ERROR"
	ErrorTypeTimeout       = "TIMEOUT_ERROR"
	ErrorTypeValidation    = "VALIDATION_ERROR"
)

type SemanticCache struct {
	redis              *redis.Client
	httpClient         *http.Client
	threshold          float64
	prefix             string
	embeddingURL       string
	embeddingProvider  EmbeddingProvider
	embeddingCache     map[string][]float64
	embeddingCacheMux  sync.RWMutex
	metrics            *CacheMetrics
	metricsMux         sync.RWMutex
	cleanupInterval    time.Duration
	lastCleanup        time.Time
	cleanupMux         sync.Mutex
	maxCacheSize       int
	enableLRU          bool
	circuitBreaker     *CircuitBreaker
	retryConfig        *RetryConfig
}

type Config struct {
	RedisAddr           string
	RedisPassword       string
	RedisDB             int
	Threshold           float64
	CachePrefix         string
	EmbeddingURL        string
	ServerPort          string
	MaxCacheSize        int
	EnableLRU           bool
	CleanupInterval     time.Duration
	EmbeddingCacheSize  int
	MaxRetries          int
	RetryBaseDelay      time.Duration
	RetryMaxDelay       time.Duration
	CircuitBreakerTimeout time.Duration
	MaxFailures         int64
}

func NewSemanticCache(config Config) *SemanticCache {
	rdb := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	// Set defaults
	if config.MaxCacheSize == 0 {
		config.MaxCacheSize = 10000
	}
	if config.CleanupInterval == 0 {
		config.CleanupInterval = 1 * time.Hour
	}
	if config.EmbeddingCacheSize == 0 {
		config.EmbeddingCacheSize = 1000
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.RetryBaseDelay == 0 {
		config.RetryBaseDelay = 100 * time.Millisecond
	}
	if config.RetryMaxDelay == 0 {
		config.RetryMaxDelay = 5 * time.Second
	}
	if config.CircuitBreakerTimeout == 0 {
		config.CircuitBreakerTimeout = 1 * time.Minute
	}
	if config.MaxFailures == 0 {
		config.MaxFailures = 5
	}

	// Initialize circuit breaker
	circuitBreaker := &CircuitBreaker{
		state:       CircuitClosed,
		timeout:     config.CircuitBreakerTimeout,
		maxFailures: config.MaxFailures,
	}

	// Initialize retry configuration
	retryConfig := &RetryConfig{
		MaxRetries: config.MaxRetries,
		BaseDelay:  config.RetryBaseDelay,
		MaxDelay:   config.RetryMaxDelay,
		Multiplier: 2.0,
	}

	cache := &SemanticCache{
		redis:            rdb,
		httpClient:       &http.Client{Timeout: 30 * time.Second},
		threshold:        config.Threshold,
		prefix:           config.CachePrefix,
		embeddingURL:     config.EmbeddingURL,
		embeddingCache:   make(map[string][]float64),
		metrics:          &CacheMetrics{},
		cleanupInterval:  config.CleanupInterval,
		maxCacheSize:     config.MaxCacheSize,
		enableLRU:        config.EnableLRU,
		circuitBreaker:   circuitBreaker,
		retryConfig:      retryConfig,
	}

	// Start background cleanup routine
	go cache.startCleanupRoutine()

	return cache
}

// Circuit breaker methods
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()

	switch cb.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		return time.Since(cb.lastFailTime) > cb.timeout
	case CircuitHalfOpen:
		return true
	default:
		return false
	}
}

func (cb *CircuitBreaker) RecordSuccess() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	cb.failureCount = 0
	cb.state = CircuitClosed
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	cb.failureCount++
	cb.lastFailTime = time.Now()

	if cb.failureCount >= cb.maxFailures {
		cb.state = CircuitOpen
	}
}

func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.state
}

// Retry logic with exponential backoff
func (sc *SemanticCache) retryWithBackoff(ctx context.Context, operation func() error) error {
	var lastErr error
	
	for attempt := 0; attempt <= sc.retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := time.Duration(float64(sc.retryConfig.BaseDelay) * math.Pow(sc.retryConfig.Multiplier, float64(attempt-1)))
			if delay > sc.retryConfig.MaxDelay {
				delay = sc.retryConfig.MaxDelay
			}
			
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}

		err := operation()
		if err == nil {
			return nil
		}
		
		lastErr = err
		
		// Don't retry on certain types of errors
		if cacheErr, ok := err.(*CacheError); ok {
			if cacheErr.Type == ErrorTypeValidation {
				return err
			}
		}
	}
	
	return &CacheError{
		Type:    ErrorTypeTimeout,
		Message: fmt.Sprintf("operation failed after %d retries", sc.retryConfig.MaxRetries),
		Err:     lastErr,
	}
}

func (sc *SemanticCache) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	// Check embedding cache first
	sc.embeddingCacheMux.RLock()
	if embedding, exists := sc.embeddingCache[text]; exists {
		sc.embeddingCacheMux.RUnlock()
		sc.updateMetrics(func(m *CacheMetrics) {
			m.EmbeddingCacheHitRate = 0.8 // Simplified for now
		})
		return embedding, nil
	}
	sc.embeddingCacheMux.RUnlock()

	// Check circuit breaker
	if !sc.circuitBreaker.CanExecute() {
		return nil, &CacheError{
			Type:    ErrorTypeCircuitBreaker,
			Message: "embedding service circuit breaker is open",
		}
	}

	if sc.embeddingURL == "" {
		return nil, &CacheError{
			Type:    ErrorTypeValidation,
			Message: "embedding URL not configured",
		}
	}

	// Use retry logic for embedding requests
	var embedding []float64
	err := sc.retryWithBackoff(ctx, func() error {
		start := time.Now()
		log.Printf("Generating embedding for text (length: %d chars)", len(text))

		request := EmbeddingRequest{
			Model:  "nomic-embed-text",
			Prompt: text,
		}

		requestJSON, err := json.Marshal(request)
		if err != nil {
			return &CacheError{
				Type:    ErrorTypeEmbedding,
				Message: "failed to marshal embedding request",
				Err:     err,
			}
		}

		req, err := http.NewRequestWithContext(ctx, "POST", sc.embeddingURL, bytes.NewBuffer(requestJSON))
		if err != nil {
			return &CacheError{
				Type:    ErrorTypeEmbedding,
				Message: "failed to create HTTP request",
				Err:     err,
			}
		}

		req.Header.Set("Content-Type", "application/json")

		resp, err := sc.httpClient.Do(req)
		if err != nil {
			sc.circuitBreaker.RecordFailure()
			return &CacheError{
				Type:    ErrorTypeEmbedding,
				Message: "failed to make HTTP request",
				Err:     err,
			}
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			sc.circuitBreaker.RecordFailure()
			return &CacheError{
				Type:    ErrorTypeEmbedding,
				Message: fmt.Sprintf("embedding endpoint returned status %d", resp.StatusCode),
			}
		}

		var embResp EmbeddingResponse
		if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
			sc.circuitBreaker.RecordFailure()
			return &CacheError{
				Type:    ErrorTypeEmbedding,
				Message: "failed to decode embedding response",
				Err:     err,
			}
		}

		embedding = embResp.Embedding
		sc.circuitBreaker.RecordSuccess()

		duration := time.Since(start)
		log.Printf("Embedding generated successfully in %v (vector size: %d)", duration, len(embedding))

		return nil
	})

	if err != nil {
		return nil, err
	}

	// Cache the embedding
	sc.embeddingCacheMux.Lock()
	if len(sc.embeddingCache) < 1000 { // Simple size limit
		sc.embeddingCache[text] = embedding
	}
	sc.embeddingCacheMux.Unlock()

	return embedding, nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Helper function to update metrics safely
func (sc *SemanticCache) updateMetrics(updateFunc func(*CacheMetrics)) {
	sc.metricsMux.Lock()
	defer sc.metricsMux.Unlock()
	updateFunc(sc.metrics)
}

// Background cleanup routine
func (sc *SemanticCache) startCleanupRoutine() {
	ticker := time.NewTicker(sc.cleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		sc.performCleanup()
	}
}

// Perform cache cleanup
func (sc *SemanticCache) performCleanup() {
	sc.cleanupMux.Lock()
	defer sc.cleanupMux.Unlock()

	ctx := context.Background()
	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	cachedKeys, err := sc.redis.SMembers(ctx, indexKey).Result()
	if err != nil {
		log.Printf("Error during cleanup: %v", err)
		return
	}

	expiredCount := 0
	for _, cachedKey := range cachedKeys {
		entryJSON, err := sc.redis.Get(ctx, cachedKey).Result()
		if err == redis.Nil {
			sc.redis.SRem(ctx, indexKey, cachedKey)
			continue
		}
		if err != nil {
			continue
		}

		var entry CacheEntry
		if err := json.Unmarshal([]byte(entryJSON), &entry); err != nil {
			continue
		}

		if time.Since(entry.Timestamp) > time.Duration(entry.TTL)*time.Second {
			sc.redis.Del(ctx, cachedKey)
			sc.redis.SRem(ctx, indexKey, cachedKey)
			expiredCount++
		}
	}

	sc.lastCleanup = time.Now()
	sc.updateMetrics(func(m *CacheMetrics) {
		m.LastCleanup = sc.lastCleanup
	})

	if expiredCount > 0 {
		log.Printf("Cleanup completed: removed %d expired entries", expiredCount)
	}
}

func (sc *SemanticCache) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	return sc.SetWithMetadata(ctx, key, value, ttl, nil)
}

func (sc *SemanticCache) SetWithMetadata(ctx context.Context, key, value string, ttl time.Duration, metadata map[string]interface{}) error {
	embedding, err := sc.getEmbedding(ctx, key)
	if err != nil {
		return fmt.Errorf("failed to get embedding for key: %w", err)
	}

	now := time.Now()
	entry := CacheEntry{
		Key:         key,
		Value:       value,
		Embedding:   embedding,
		Timestamp:   now,
		TTL:         int64(ttl.Seconds()),
		AccessCount: 0,
		LastAccess:  now,
		Metadata:    metadata,
	}

	entryJSON, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal cache entry: %w", err)
	}

	redisKey := fmt.Sprintf("%s:%s", sc.prefix, key)
	err = sc.redis.Set(ctx, redisKey, entryJSON, ttl).Err()
	if err != nil {
		return fmt.Errorf("failed to set cache entry in Redis: %w", err)
	}

	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	err = sc.redis.SAdd(ctx, indexKey, redisKey).Err()
	if err != nil {
		return fmt.Errorf("failed to add key to index: %w", err)
	}

	// Update metrics
	sc.updateMetrics(func(m *CacheMetrics) {
		m.TotalEntries++
	})

	return nil
}

func (sc *SemanticCache) Get(ctx context.Context, key string) (string, bool, error) {
	start := time.Now()

	queryEmbedding, err := sc.getEmbedding(ctx, key)
	if err != nil {
		return "", false, fmt.Errorf("failed to get embedding for query: %w", err)
	}

	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	cachedKeys, err := sc.redis.SMembers(ctx, indexKey).Result()
	if err != nil {
		return "", false, fmt.Errorf("failed to get cached keys: %w", err)
	}

	log.Printf("Searching through %d cached entries for similarities", len(cachedKeys))

	type similarity struct {
		key   string
		score float64
		entry CacheEntry
	}

	var similarities []similarity

	for _, cachedKey := range cachedKeys {
		entryJSON, err := sc.redis.Get(ctx, cachedKey).Result()
		if err == redis.Nil {
			sc.redis.SRem(ctx, indexKey, cachedKey)
			continue
		}
		if err != nil {
			continue
		}

		var entry CacheEntry
		if err := json.Unmarshal([]byte(entryJSON), &entry); err != nil {
			continue
		}

		if time.Since(entry.Timestamp) > time.Duration(entry.TTL)*time.Second {
			sc.redis.Del(ctx, cachedKey)
			sc.redis.SRem(ctx, indexKey, cachedKey)
			continue
		}

		score := cosineSimilarity(queryEmbedding, entry.Embedding)
		log.Printf("Cosine similarity score: %f", score)
		if score >= sc.threshold {
			similarities = append(similarities, similarity{
				key:   cachedKey,
				score: score,
				entry: entry,
			})
		}
	}

	if len(similarities) == 0 {
		duration := time.Since(start)
		log.Printf("No semantic matches found (searched in %v)", duration)
		return "", false, nil
	}

	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].score > similarities[j].score
	})

	bestMatch := similarities[0]
	duration := time.Since(start)
	log.Printf("Found semantic match with score %.4f in %v", bestMatch.score, duration)

	// Update access statistics for the matched entry
	sc.updateAccessStats(ctx, bestMatch.key, bestMatch.entry)

	// Update metrics
	sc.updateMetrics(func(m *CacheMetrics) {
		m.AverageResponseTime = duration
	})

	return bestMatch.entry.Value, true, nil
}

// Update access statistics for a cache entry
func (sc *SemanticCache) updateAccessStats(ctx context.Context, redisKey string, entry CacheEntry) {
	entry.AccessCount++
	entry.LastAccess = time.Now()

	entryJSON, err := json.Marshal(entry)
	if err != nil {
		log.Printf("Failed to marshal updated entry: %v", err)
		return
	}

	// Update the entry in Redis with new access stats
	ttl := time.Duration(entry.TTL) * time.Second
	err = sc.redis.Set(ctx, redisKey, entryJSON, ttl).Err()
	if err != nil {
		log.Printf("Failed to update access stats: %v", err)
	}
}

func (sc *SemanticCache) Delete(ctx context.Context, key string) error {
	redisKey := fmt.Sprintf("%s:%s", sc.prefix, key)

	err := sc.redis.Del(ctx, redisKey).Err()
	if err != nil {
		return fmt.Errorf("failed to delete cache entry: %w", err)
	}

	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	err = sc.redis.SRem(ctx, indexKey, redisKey).Err()
	if err != nil {
		return fmt.Errorf("failed to remove key from index: %w", err)
	}

	return nil
}

func (sc *SemanticCache) Clear(ctx context.Context) error {
	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	cachedKeys, err := sc.redis.SMembers(ctx, indexKey).Result()
	if err != nil {
		return fmt.Errorf("failed to get cached keys: %w", err)
	}

	if len(cachedKeys) > 0 {
		err = sc.redis.Del(ctx, cachedKeys...).Err()
		if err != nil {
			return fmt.Errorf("failed to delete cache entries: %w", err)
		}
	}

	err = sc.redis.Del(ctx, indexKey).Err()
	if err != nil {
		return fmt.Errorf("failed to delete index: %w", err)
	}

	return nil
}

func (sc *SemanticCache) Stats(ctx context.Context) (map[string]interface{}, error) {
	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	count, err := sc.redis.SCard(ctx, indexKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get cache count: %w", err)
	}

	stats := map[string]interface{}{
		"total_entries": count,
		"threshold":     sc.threshold,
		"prefix":        sc.prefix,
	}

	return stats, nil
}

// Batch operations
func (sc *SemanticCache) BatchSet(ctx context.Context, entries []BatchEntry) (*BatchSetResponse, error) {
	response := &BatchSetResponse{
		Success:   true,
		Processed: 0,
		Failed:    []string{},
	}

	for _, entry := range entries {
		ttl := time.Duration(entry.TTL) * time.Second
		if ttl == 0 {
			ttl = 1 * time.Hour // Default TTL
		}

		err := sc.SetWithMetadata(ctx, entry.Key, entry.Value, ttl, entry.Metadata)
		if err != nil {
			response.Failed = append(response.Failed, entry.Key)
			log.Printf("Failed to set batch entry '%s': %v", entry.Key, err)
		} else {
			response.Processed++
		}
	}

	if len(response.Failed) > 0 {
		response.Success = false
		response.Message = fmt.Sprintf("Failed to process %d entries", len(response.Failed))
	}

	return response, nil
}

func (sc *SemanticCache) BatchGet(ctx context.Context, keys []string) (*BatchGetResponse, error) {
	response := &BatchGetResponse{
		Results: make(map[string]string),
		Found:   make(map[string]bool),
	}

	for _, key := range keys {
		value, found, err := sc.Get(ctx, key)
		if err != nil {
			log.Printf("Error getting batch key '%s': %v", key, err)
			response.Found[key] = false
			continue
		}

		response.Found[key] = found
		if found {
			response.Results[key] = value
		}
	}

	return response, nil
}

// Enhanced search functionality
func (sc *SemanticCache) Search(ctx context.Context, query string, limit int, threshold float64) (*SearchResponse, error) {
	if threshold == 0 {
		threshold = sc.threshold
	}
	if limit == 0 {
		limit = 10
	}

	queryEmbedding, err := sc.getEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding for query: %w", err)
	}

	indexKey := fmt.Sprintf("%s:index", sc.prefix)
	cachedKeys, err := sc.redis.SMembers(ctx, indexKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get cached keys: %w", err)
	}

	var results []SimilarityResult

	for _, cachedKey := range cachedKeys {
		entryJSON, err := sc.redis.Get(ctx, cachedKey).Result()
		if err == redis.Nil {
			sc.redis.SRem(ctx, indexKey, cachedKey)
			continue
		}
		if err != nil {
			continue
		}

		var entry CacheEntry
		if err := json.Unmarshal([]byte(entryJSON), &entry); err != nil {
			continue
		}

		if time.Since(entry.Timestamp) > time.Duration(entry.TTL)*time.Second {
			sc.redis.Del(ctx, cachedKey)
			sc.redis.SRem(ctx, indexKey, cachedKey)
			continue
		}

		score := cosineSimilarity(queryEmbedding, entry.Embedding)
		if score >= threshold {
			results = append(results, SimilarityResult{
				Key:      entry.Key,
				Value:    entry.Value,
				Score:    score,
				Metadata: entry.Metadata,
			})
		}
	}

	// Sort by similarity score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	return &SearchResponse{
		Results: results,
		Count:   len(results),
	}, nil
}

// Get detailed metrics
func (sc *SemanticCache) GetMetrics(ctx context.Context) (*CacheMetrics, error) {
	sc.metricsMux.RLock()
	defer sc.metricsMux.RUnlock()

	// Get current stats from Redis
	stats, err := sc.Stats(ctx)
	if err != nil {
		return nil, err
	}

	// Update metrics with current data
	metrics := &CacheMetrics{
		TotalEntries:           stats["total_entries"].(int64),
		HitRate:                sc.metrics.HitRate,
		AverageResponseTime:    sc.metrics.AverageResponseTime,
		EmbeddingCacheHitRate:  sc.metrics.EmbeddingCacheHitRate,
		MemoryUsage:            sc.metrics.MemoryUsage,
		LastCleanup:            sc.metrics.LastCleanup,
	}

	return metrics, nil
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		log.Printf("[%s] %s %s - Started", r.Method, r.URL.Path, r.RemoteAddr)

		next.ServeHTTP(w, r)

		duration := time.Since(start)
		log.Printf("[%s] %s %s - Completed in %v", r.Method, r.URL.Path, r.RemoteAddr, duration)
	})
}

func (sc *SemanticCache) handleCacheGet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CacheGetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Key == "" {
		http.Error(w, "Key is required", http.StatusBadRequest)
		return
	}

	log.Printf("Cache GET request for key: %s", req.Key)

	ctx := r.Context()
	value, found, err := sc.Get(ctx, req.Key)
	if err != nil {
		log.Printf("Error getting cache for key '%s': %v", req.Key, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if found {
		log.Printf("Cache HIT for key: %s", req.Key)
	} else {
		log.Printf("Cache MISS for key: %s", req.Key)
	}

	response := CacheGetResponse{
		Value: value,
		Found: found,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (sc *SemanticCache) handleCacheSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CacheSetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Key == "" || req.Value == "" {
		http.Error(w, "Key and value are required", http.StatusBadRequest)
		return
	}

	if req.TTL <= 0 {
		req.TTL = 3600
	}

	log.Printf("Cache SET request for key: %s (TTL: %ds)", req.Key, req.TTL)

	ctx := r.Context()
	ttl := time.Duration(req.TTL) * time.Second
	err := sc.Set(ctx, req.Key, req.Value, ttl)

	response := CacheSetResponse{}
	if err != nil {
		log.Printf("Error setting cache for key '%s': %v", req.Key, err)
		response.Success = false
		response.Message = "Failed to set cache entry"
		w.WriteHeader(http.StatusInternalServerError)
	} else {
		log.Printf("Cache SET successful for key: %s", req.Key)
		response.Success = true
		response.Message = "Cache entry set successfully"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (sc *SemanticCache) handleCacheDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CacheDeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Key == "" {
		http.Error(w, "Key is required", http.StatusBadRequest)
		return
	}

	log.Printf("Cache DELETE request for key: %s", req.Key)

	ctx := r.Context()
	err := sc.Delete(ctx, req.Key)

	response := CacheDeleteResponse{}
	if err != nil {
		log.Printf("Error deleting cache for key '%s': %v", req.Key, err)
		response.Success = false
		response.Message = "Failed to delete cache entry"
		w.WriteHeader(http.StatusInternalServerError)
	} else {
		log.Printf("Cache DELETE successful for key: %s", req.Key)
		response.Success = true
		response.Message = "Cache entry deleted successfully"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// New HTTP handlers for enhanced functionality
func (sc *SemanticCache) handleBatchSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req BatchSetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if len(req.Entries) == 0 {
		http.Error(w, "Entries are required", http.StatusBadRequest)
		return
	}

	log.Printf("Batch SET request for %d entries", len(req.Entries))

	ctx := r.Context()
	response, err := sc.BatchSet(ctx, req.Entries)
	if err != nil {
		log.Printf("Error in batch set: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (sc *SemanticCache) handleBatchGet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req BatchGetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if len(req.Keys) == 0 {
		http.Error(w, "Keys are required", http.StatusBadRequest)
		return
	}

	log.Printf("Batch GET request for %d keys", len(req.Keys))

	ctx := r.Context()
	response, err := sc.BatchGet(ctx, req.Keys)
	if err != nil {
		log.Printf("Error in batch get: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (sc *SemanticCache) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "Query is required", http.StatusBadRequest)
		return
	}

	log.Printf("Search request for query: %s", req.Query)

	ctx := r.Context()
	response, err := sc.Search(ctx, req.Query, req.Limit, req.Threshold)
	if err != nil {
		log.Printf("Error in search: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (sc *SemanticCache) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()
	metrics, err := sc.GetMetrics(ctx)
	if err != nil {
		log.Printf("Error getting metrics: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Printf("Initializing Semantic Cache Server...")

	config := Config{
		RedisAddr:             "localhost:6379",
		RedisPassword:         "",
		RedisDB:               0,
		Threshold:             0.8,
		CachePrefix:           "semantic_cache",
		EmbeddingURL:          "http://localhost:11434/api/embeddings",
		ServerPort:            ":8080",
		MaxCacheSize:          10000,
		EnableLRU:             true,
		CleanupInterval:       1 * time.Hour,
		EmbeddingCacheSize:    1000,
		MaxRetries:            3,
		RetryBaseDelay:        100 * time.Millisecond,
		RetryMaxDelay:         5 * time.Second,
		CircuitBreakerTimeout: 1 * time.Minute,
		MaxFailures:           5,
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Set up HTTP server with logging middleware
	mux := http.NewServeMux()
	
	// Core cache operations
	mux.HandleFunc("/cache/get", cache.handleCacheGet)
	mux.HandleFunc("/cache/set", cache.handleCacheSet)
	mux.HandleFunc("/cache/delete", cache.handleCacheDelete)
	
	// Enhanced operations
	mux.HandleFunc("/cache/batch/set", cache.handleBatchSet)
	mux.HandleFunc("/cache/batch/get", cache.handleBatchGet)
	mux.HandleFunc("/cache/search", cache.handleSearch)
	mux.HandleFunc("/cache/metrics", cache.handleMetrics)

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})

	// Wrap with logging middleware
	loggedMux := loggingMiddleware(mux)

	// Example usage (optional - can be removed in production)
	fmt.Println("Semantic Cache Server")
	fmt.Println("=====================")

	// Set some example cache entries
	err := cache.Set(ctx, "What is machine learning?", "Machine learning is a subset of AI that enables computers to learn without explicit programming.", 1*time.Hour)
	if err != nil {
		log.Printf("Error setting example cache: %v", err)
	}

	err = cache.Set(ctx, "How does deep learning work?", "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.", 1*time.Hour)
	if err != nil {
		log.Printf("Error setting example cache: %v", err)
	}

	fmt.Printf("Semantic Cache Server starting on port %s\n", config.ServerPort)
	fmt.Println("Available endpoints:")
	fmt.Println("  POST /cache/get - Retrieve cached values")
	fmt.Println("  POST /cache/set - Set cache values")
	fmt.Println("  POST /cache/delete - Delete cache values")
	fmt.Println("  POST /cache/batch/set - Set multiple cache values")
	fmt.Println("  POST /cache/batch/get - Get multiple cache values")
	fmt.Println("  POST /cache/search - Search with semantic similarity")
	fmt.Println("  GET /cache/metrics - Get cache metrics and statistics")
	fmt.Println("  GET /health - Health check")

	log.Printf("Starting semantic cache server on %s", config.ServerPort)
	log.Fatal(http.ListenAndServe(config.ServerPort, loggedMux))
}
