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
	"time"

	"github.com/go-redis/redis/v8"
)

type CacheEntry struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	Embedding []float64 `json:"embedding"`
	Timestamp time.Time `json:"timestamp"`
	TTL       int64     `json:"ttl"`
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

type SemanticCache struct {
	redis        *redis.Client
	httpClient   *http.Client
	threshold    float64
	prefix       string
	embeddingURL string
}

type Config struct {
	RedisAddr     string
	RedisPassword string
	RedisDB       int
	Threshold     float64
	CachePrefix   string
	EmbeddingURL  string
	ServerPort    string
}

func NewSemanticCache(config Config) *SemanticCache {
	rdb := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	return &SemanticCache{
		redis:        rdb,
		httpClient:   &http.Client{Timeout: 30 * time.Second},
		threshold:    config.Threshold,
		prefix:       config.CachePrefix,
		embeddingURL: config.EmbeddingURL,
	}
}

func (sc *SemanticCache) getEmbedding(ctx context.Context, text string) ([]float64, error) {
	if sc.embeddingURL == "" {
		return nil, fmt.Errorf("embedding URL not configured")
	}

	start := time.Now()
	log.Printf("Generating embedding for text (length: %d chars)", len(text))

	request := EmbeddingRequest{
		Model:  "nomic-embed-text",
		Prompt: text,
	}

	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", sc.embeddingURL, bytes.NewBuffer(requestJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := sc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding endpoint returned status %d", resp.StatusCode)
	}

	var embResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("failed to decode embedding response: %w", err)
	}

	duration := time.Since(start)
	log.Printf("Embedding generated successfully in %v (vector size: %d)", duration, len(embResp.Embedding))

	return embResp.Embedding, nil
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

func (sc *SemanticCache) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	embedding, err := sc.getEmbedding(ctx, key)
	if err != nil {
		return fmt.Errorf("failed to get embedding for key: %w", err)
	}

	entry := CacheEntry{
		Key:       key,
		Value:     value,
		Embedding: embedding,
		Timestamp: time.Now(),
		TTL:       int64(ttl.Seconds()),
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

	return bestMatch.entry.Value, true, nil
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

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Printf("Initializing Semantic Cache Server...")

	config := Config{
		RedisAddr:     "localhost:6379",
		RedisPassword: "",
		RedisDB:       0,
		Threshold:     0.8,
		CachePrefix:   "semantic_cache",
		EmbeddingURL:  "http://localhost:11434/api/embeddings",
		ServerPort:    ":8080",
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Set up HTTP server with logging middleware
	mux := http.NewServeMux()
	mux.HandleFunc("/cache/get", cache.handleCacheGet)
	mux.HandleFunc("/cache/set", cache.handleCacheSet)
	mux.HandleFunc("/cache/delete", cache.handleCacheDelete)

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
	fmt.Println("  GET /health - Health check")

	log.Printf("Starting semantic cache server on %s", config.ServerPort)
	log.Fatal(http.ListenAndServe(config.ServerPort, loggedMux))
}
