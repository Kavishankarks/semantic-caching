package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/sashabaranov/go-openai"
)

// Import the semantic cache implementation
// In a real project, you would import this as a separate package

func main() {
	// Check if OpenAI API key is set
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		log.Fatal("Please set the OPENAI_API_KEY environment variable")
	}

	config := Config{
		RedisAddr:     "localhost:6379",
		RedisPassword: "",
		RedisDB:       0,
		OpenAIToken:   openaiKey,
		Threshold:     0.75, // Lower threshold for more matches
		CachePrefix:   "demo_cache",
	}

	cache := NewSemanticCache(config)
	ctx := context.Background()

	// Test Redis connection
	pong, err := cache.redis.Ping(ctx).Result()
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}
	fmt.Printf("Connected to Redis: %s\n", pong)

	// Clear any existing cache
	cache.Clear(ctx)
	
	fmt.Println("\n=== Semantic Caching Demo ===")
	
	// Populate cache with various topics
	cacheData := map[string]string{
		"What is artificial intelligence?": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
		"How does machine learning work?": "Machine learning is a method of data analysis that automates analytical model building, allowing computers to learn from data without being explicitly programmed.",
		"Explain neural networks": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
		"What is deep learning?": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
		"How to optimize database queries?": "Database query optimization involves using indexes, avoiding N+1 queries, using appropriate joins, and analyzing query execution plans.",
		"Best practices for API design": "Good API design includes consistent naming, proper HTTP status codes, versioning, documentation, and following RESTful principles.",
		"What is microservices architecture?": "Microservices architecture is a design approach where applications are built as a collection of loosely coupled, independently deployable services.",
	}

	fmt.Println("\nPopulating cache...")
	for key, value := range cacheData {
		err := cache.Set(ctx, key, value, 24*time.Hour)
		if err != nil {
			log.Printf("Error caching '%s': %v", key, err)
		} else {
			fmt.Printf("✓ Cached: %s\n", key)
		}
	}

	// Test semantic retrieval with similar but different queries
	testQueries := []string{
		"What is AI?",                    // Similar to "What is artificial intelligence?"
		"How does ML work?",              // Similar to "How does machine learning work?"
		"Explain deep neural networks",   // Similar to both neural networks and deep learning
		"Database query performance",     // Similar to database optimization
		"REST API best practices",       // Similar to API design
		"Microservice patterns",         // Similar to microservices architecture
		"What is quantum computing?",     // Should not match anything
	}

	fmt.Println("\n=== Testing Semantic Retrieval ===")
	
	for i, query := range testQueries {
		fmt.Printf("\n%d. Query: \"%s\"\n", i+1, query)
		
		start := time.Now()
		value, found, err := cache.Get(ctx, query)
		duration := time.Since(start)
		
		if err != nil {
			fmt.Printf("   ❌ Error: %v\n", err)
			continue
		}
		
		if found {
			fmt.Printf("   ✅ Match found (took %v)\n", duration)
			fmt.Printf("   📄 Cached response: %s\n", value)
		} else {
			fmt.Printf("   ❌ No semantic match found (took %v)\n", duration)
			fmt.Printf("   💡 This would trigger an LLM call in a real application\n")
		}
	}

	// Show cache statistics
	fmt.Println("\n=== Cache Statistics ===")
	stats, err := cache.Stats(ctx)
	if err != nil {
		log.Printf("Error getting stats: %v", err)
	} else {
		fmt.Printf("Cache entries: %v\n", stats["total_entries"])
		fmt.Printf("Similarity threshold: %v\n", stats["threshold"])
		fmt.Printf("Cache prefix: %v\n", stats["prefix"])
	}

	// Demonstrate cache management
	fmt.Println("\n=== Cache Management Demo ===")
	
	// Delete a specific entry
	fmt.Println("Deleting entry for 'What is artificial intelligence?'...")
	err = cache.Delete(ctx, "What is artificial intelligence?")
	if err != nil {
		log.Printf("Error deleting entry: %v", err)
	} else {
		fmt.Println("✓ Entry deleted successfully")
	}

	// Test if deleted entry affects semantic matching
	value, found, err := cache.Get(ctx, "What is AI?")
	if err != nil {
		log.Printf("Error testing deleted entry: %v", err)
	} else if found {
		fmt.Printf("Still found match: %s\n", value)
	} else {
		fmt.Println("No match found after deletion (as expected)")
	}

	fmt.Println("\n=== Demo Complete ===")
	fmt.Println("The semantic cache successfully:")
	fmt.Println("• Stores entries with vector embeddings")
	fmt.Println("• Retrieves semantically similar content")
	fmt.Println("• Manages TTL and cleanup")
	fmt.Println("• Provides cache statistics and management")
}