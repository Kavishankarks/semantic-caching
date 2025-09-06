# Enhanced Semantic Caching Service

A high-performance Go-based semantic caching service that uses Redis for storage and embedding models for semantic similarity matching. Instead of relying on exact key matches, this cache intelligently retrieves cached results based on semantic meaning with advanced features for production use.

## 🚀 Features

### Core Functionality
- **Semantic Matching**: Uses embedding models to find semantically similar cached entries
- **Redis Backend**: Leverages Redis for fast, persistent storage with connection pooling
- **Configurable Similarity Threshold**: Adjust how strict semantic matching should be
- **TTL Support**: Automatic expiration of cached entries with background cleanup

### Advanced Features
- **Batch Operations**: Efficient bulk set/get operations for improved performance
- **Embedding Caching**: In-memory cache for frequently used embeddings
- **Access Statistics**: Track usage patterns and cache hit rates
- **Rich Metadata**: Store and search with custom metadata
- **Background Cleanup**: Automatic removal of expired entries
- **Comprehensive Metrics**: Detailed performance and usage statistics
- **Enhanced Search**: Multi-result semantic search with scoring
- **LRU Support**: Optional Least Recently Used eviction policy

### Production Ready
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Health Checks**: Built-in health monitoring endpoints
- **Structured Logging**: Comprehensive logging with request tracking
- **Error Handling**: Robust error handling with retry logic and circuit breaker patterns
- **Performance Monitoring**: Real-time metrics and statistics
- **Resilience**: Circuit breaker protection against cascading failures
- **Graceful Degradation**: Continues operating even when external services fail

## How It Works

1. **Storage**: When storing a key-value pair, the service generates embeddings for the key using OpenAI's text embedding model
2. **Retrieval**: When querying, it generates embeddings for the query and compares against all cached embeddings using cosine similarity
3. **Matching**: If similarity exceeds the configured threshold, returns the cached value instead of requiring a new LLM call

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   go mod tidy
   ```
3. Start Redis server:
   ```bash
   redis-server
   ```
4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Example

```go
package main

import (
    "context"
    "time"
)

func main() {
    config := Config{
        RedisAddr:     "localhost:6379",
        RedisPassword: "",
        RedisDB:       0,
        OpenAIToken:   "your-openai-api-key",
        Threshold:     0.8,
        CachePrefix:   "semantic_cache",
    }

    cache := NewSemanticCache(config)
    ctx := context.Background()

    // Store a response
    cache.Set(ctx, "What is machine learning?", 
        "Machine learning is a subset of AI...", 
        1*time.Hour)

    // Retrieve with semantic matching
    value, found, err := cache.Get(ctx, "What is ML?")
    if found {
        fmt.Println("Found cached response:", value)
    }
}
```

### Configuration Options

- `RedisAddr`: Redis server address (default: "localhost:6379")
- `RedisPassword`: Redis password (default: "")
- `RedisDB`: Redis database number (default: 0)
- `OpenAIToken`: OpenAI API key (required)
- `Threshold`: Similarity threshold 0.0-1.0 (default: 0.8)
- `CachePrefix`: Prefix for Redis keys (default: "semantic_cache")

### API Methods

#### Core Operations
- `Set(ctx, key, value, ttl)`: Store a key-value pair with TTL
- `SetWithMetadata(ctx, key, value, ttl, metadata)`: Store with custom metadata
- `Get(ctx, key)`: Retrieve value by semantic similarity
- `Delete(ctx, key)`: Remove a specific cache entry
- `Clear(ctx)`: Remove all cache entries

#### Batch Operations
- `BatchSet(ctx, entries)`: Store multiple entries efficiently
- `BatchGet(ctx, keys)`: Retrieve multiple values

#### Advanced Operations
- `Search(ctx, query, limit, threshold)`: Multi-result semantic search
- `GetMetrics(ctx)`: Get detailed performance metrics
- `Stats(ctx)`: Get basic cache statistics

### HTTP API Endpoints

#### Core Endpoints
- `POST /cache/get` - Retrieve cached values
- `POST /cache/set` - Set cache values
- `POST /cache/delete` - Delete cache values

#### Enhanced Endpoints
- `POST /cache/batch/set` - Set multiple cache values
- `POST /cache/batch/get` - Get multiple cache values
- `POST /cache/search` - Search with semantic similarity
- `GET /cache/metrics` - Get cache metrics and statistics
- `GET /health` - Health check

## Use Cases

### LLM Response Caching
```go
// Instead of calling LLM for every query
response, found, err := cache.Get(ctx, userQuery)
if !found {
    response = callLLM(userQuery)
    cache.Set(ctx, userQuery, response, 24*time.Hour)
}
```

### FAQ Systems
```go
// Cache common questions and retrieve similar ones
cache.Set(ctx, "How do I reset my password?", detailedInstructions, 7*24*time.Hour)

// Later, queries like "forgot password" or "password reset" will match
```

### Knowledge Base
```go
// Store technical documentation
cache.Set(ctx, "How to optimize database queries?", optimizationGuide, time.Hour)

// Matches queries like "database performance" or "query optimization"
```

### Batch Operations
```go
// Efficiently store multiple entries
entries := []BatchEntry{
    {
        Key:   "What is AI?",
        Value: "AI is artificial intelligence...",
        TTL:   3600,
        Metadata: map[string]interface{}{
            "category": "AI",
            "difficulty": "beginner",
        },
    },
    {
        Key:   "How does ML work?",
        Value: "Machine learning works by...",
        TTL:   3600,
        Metadata: map[string]interface{}{
            "category": "ML",
            "difficulty": "intermediate",
        },
    },
}

response, err := cache.BatchSet(ctx, entries)
```

### Enhanced Search
```go
// Search with custom parameters
results, err := cache.Search(ctx, "artificial intelligence", 5, 0.8)
for _, result := range results.Results {
    fmt.Printf("Score: %.3f - %s\n", result.Score, result.Key)
    fmt.Printf("Metadata: %+v\n", result.Metadata)
}
```

### Performance Monitoring
```go
// Get detailed metrics
metrics, err := cache.GetMetrics(ctx)
fmt.Printf("Hit Rate: %.2f%%\n", metrics.HitRate*100)
fmt.Printf("Average Response Time: %v\n", metrics.AverageResponseTime)
fmt.Printf("Total Entries: %d\n", metrics.TotalEntries)
```

### Error Handling and Resilience
```go
// The cache automatically handles errors with retry logic and circuit breakers
// Check for specific error types
value, found, err := cache.Get(ctx, "query")
if err != nil {
    if cacheErr, ok := err.(*CacheError); ok {
        switch cacheErr.Type {
        case ErrorTypeCircuitBreaker:
            // Embedding service is down, use fallback
            fmt.Println("Using fallback response")
        case ErrorTypeValidation:
            // Invalid input, don't retry
            fmt.Println("Invalid query format")
        case ErrorTypeEmbedding:
            // Embedding service error, will retry automatically
            fmt.Println("Embedding service temporarily unavailable")
        }
    }
}
```

## Performance Benefits

- **Reduced API Costs**: Fewer calls to expensive LLM APIs
- **Faster Response Times**: Sub-second retrieval vs. multi-second LLM generation
- **Improved User Experience**: Instant responses for similar queries
- **Resource Efficiency**: Lower computational overhead

## Configuration Recommendations

- **High Precision** (fewer matches): Threshold = 0.9-0.95
- **Balanced** (moderate matches): Threshold = 0.8-0.85
- **High Recall** (more matches): Threshold = 0.7-0.8

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd semantic-caching

# Start all services (Redis, Ollama, Semantic Cache)
docker-compose up -d

# Wait for services to be ready
docker-compose logs -f

# Test the service
curl http://localhost:8080/health
```

### Manual Setup

```bash
# Install dependencies
go mod tidy

# Start Redis
redis-server

# Start Ollama (for embeddings)
ollama serve

# Run the service
go run main.go

# Run the enhanced demo
go run example/enhanced_demo.go
```

### Configuration

The service can be configured via environment variables or the `config.yaml` file:

```yaml
server:
  port: ":8080"

redis:
  addr: "localhost:6379"
  db: 0

cache:
  threshold: 0.8
  max_cache_size: 10000
  cleanup_interval: "1h"

embedding:
  url: "http://localhost:11434/api/embeddings"
  model: "nomic-embed-text"

# Error handling and resilience
error_handling:
  max_retries: 3
  retry_base_delay: "100ms"
  retry_max_delay: "5s"

# Circuit breaker configuration
circuit_breaker:
  timeout: "1m"
  max_failures: 5
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│ Semantic Cache   │───▶│     Redis       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  OpenAI API      │
                       │  (Embeddings)    │
                       └──────────────────┘
```

The service acts as an intelligent layer between your application and Redis, using embedding models to enable semantic search capabilities.

## Testing

### Run Tests
```bash
# Run all tests
go test -v

# Run tests with coverage
go test -v -cover

# Run benchmarks
go test -bench=.

# Run specific test
go test -v -run TestEnhancedSemanticCache
```

### Test Requirements
- Redis server running on localhost:6379
- Different Redis databases are used for different test suites to avoid conflicts

### Run Error Handling Tests
```bash
# Test circuit breaker functionality
go test -v -run TestCircuitBreaker

# Test retry logic
go test -v -run TestRetryWithBackoff

# Test error handling integration
go test -v -run TestErrorHandlingInOperations

# Test circuit breaker integration
go test -v -run TestCircuitBreakerIntegration

# Run all error handling tests
go test -v -run TestError
```

## Development

### Project Structure
```
semantic-caching/
├── main.go                 # Main application code
├── main_enhanced_test.go   # Enhanced test suite
├── error_handling_test.go  # Error handling and resilience tests
├── example/
│   ├── main.go            # Basic example
│   └── enhanced_demo.go   # Enhanced features demo
├── config.yaml            # Configuration file
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── README.md              # This file
```

### Adding New Features
1. Add new types and methods to `main.go`
2. Add corresponding HTTP handlers
3. Update tests in `main_enhanced_test.go`
4. Update documentation and examples
5. Test with Docker Compose setup

### Performance Considerations
- Embedding cache reduces API calls for repeated queries
- Batch operations improve throughput for bulk operations
- Background cleanup prevents memory bloat
- Connection pooling optimizes Redis performance
- Metrics help identify performance bottlenecks
- Circuit breaker prevents cascading failures
- Retry logic with exponential backoff handles transient failures
- Error handling provides graceful degradation

## License

MIT License