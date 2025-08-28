# Semantic Caching Service

A Go-based semantic caching service that uses Redis for storage and OpenAI embeddings for semantic similarity matching. Instead of relying on exact key matches, this cache intelligently retrieves cached results based on semantic meaning.

## Features

- **Semantic Matching**: Uses OpenAI embeddings to find semantically similar cached entries
- **Redis Backend**: Leverages Redis for fast, persistent storage
- **Configurable Similarity Threshold**: Adjust how strict semantic matching should be
- **TTL Support**: Automatic expiration of cached entries
- **Cache Management**: Full CRUD operations and statistics
- **Performance Optimization**: Reduces LLM API calls by reusing semantically similar responses

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

- `Set(ctx, key, value, ttl)`: Store a key-value pair with TTL
- `Get(ctx, key)`: Retrieve value by semantic similarity
- `Delete(ctx, key)`: Remove a specific cache entry
- `Clear(ctx)`: Remove all cache entries
- `Stats(ctx)`: Get cache statistics

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

## Performance Benefits

- **Reduced API Costs**: Fewer calls to expensive LLM APIs
- **Faster Response Times**: Sub-second retrieval vs. multi-second LLM generation
- **Improved User Experience**: Instant responses for similar queries
- **Resource Efficiency**: Lower computational overhead

## Configuration Recommendations

- **High Precision** (fewer matches): Threshold = 0.9-0.95
- **Balanced** (moderate matches): Threshold = 0.8-0.85
- **High Recall** (more matches): Threshold = 0.7-0.8

## Running the Demo

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run the example
go run example/main.go
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

The service acts as an intelligent layer between your application and Redis, using OpenAI's embedding API to enable semantic search capabilities.

## License

MIT License