package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// Enhanced demo showcasing all the new features
func main() {
	baseURL := "http://localhost:8080"
	
	fmt.Println("=== Enhanced Semantic Cache Demo ===")
	fmt.Println("This demo showcases the improved semantic caching features:")
	fmt.Println("• Batch operations")
	fmt.Println("• Enhanced search with metadata")
	fmt.Println("• Performance metrics")
	fmt.Println("• Access statistics")
	fmt.Println()

	// Wait for server to be ready
	fmt.Println("Waiting for server to be ready...")
	if !waitForServer(baseURL) {
		log.Fatal("Server is not responding")
	}

	// Demo 1: Batch operations
	fmt.Println("\n=== Demo 1: Batch Operations ===")
	demoBatchOperations(baseURL)

	// Demo 2: Enhanced search with metadata
	fmt.Println("\n=== Demo 2: Enhanced Search with Metadata ===")
	demoEnhancedSearch(baseURL)

	// Demo 3: Performance metrics
	fmt.Println("\n=== Demo 3: Performance Metrics ===")
	demoMetrics(baseURL)

	// Demo 4: Access statistics
	fmt.Println("\n=== Demo 4: Access Statistics ===")
	demoAccessStatistics(baseURL)

	fmt.Println("\n=== Demo Complete ===")
	fmt.Println("The enhanced semantic cache now provides:")
	fmt.Println("• Batch operations for improved performance")
	fmt.Println("• Rich metadata support")
	fmt.Println("• Comprehensive metrics and monitoring")
	fmt.Println("• Access pattern tracking")
	fmt.Println("• Background cleanup and maintenance")
}

func waitForServer(baseURL string) bool {
	for i := 0; i < 10; i++ {
		resp, err := http.Get(baseURL + "/health")
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			return true
		}
		time.Sleep(1 * time.Second)
	}
	return false
}

func demoBatchOperations(baseURL string) {
	// Batch set operation
	batchSetReq := map[string]interface{}{
		"entries": []map[string]interface{}{
			{
				"key":   "What is machine learning?",
				"value": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
				"ttl":   3600,
				"metadata": map[string]interface{}{
					"category": "AI",
					"difficulty": "beginner",
					"tags": []string{"ml", "ai", "learning"},
				},
			},
			{
				"key":   "How do neural networks work?",
				"value": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
				"ttl":   3600,
				"metadata": map[string]interface{}{
					"category": "AI",
					"difficulty": "intermediate",
					"tags": []string{"neural", "networks", "deep-learning"},
				},
			},
			{
				"key":   "What is database optimization?",
				"value": "Database optimization involves using indexes, avoiding N+1 queries, using appropriate joins, and analyzing query execution plans.",
				"ttl":   3600,
				"metadata": map[string]interface{}{
					"category": "Database",
					"difficulty": "advanced",
					"tags": []string{"database", "performance", "optimization"},
				},
			},
		},
	}

	reqBody, _ := json.Marshal(batchSetReq)
	resp, err := http.Post(baseURL+"/cache/batch/set", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("Error in batch set: %v", err)
		return
	}
	defer resp.Body.Close()

	var batchSetResp map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&batchSetResp)
	fmt.Printf("Batch set result: %+v\n", batchSetResp)

	// Batch get operation
	batchGetReq := map[string]interface{}{
		"keys": []string{"What is machine learning?", "How do neural networks work?", "What is database optimization?"},
	}

	reqBody, _ = json.Marshal(batchGetReq)
	resp, err = http.Post(baseURL+"/cache/batch/get", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("Error in batch get: %v", err)
		return
	}
	defer resp.Body.Close()

	var batchGetResp map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&batchGetResp)
	fmt.Printf("Batch get results: %+v\n", batchGetResp)
}

func demoEnhancedSearch(baseURL string) {
	// Search with different queries
	searchQueries := []string{
		"What is ML?",                    // Should match machine learning
		"Neural network basics",          // Should match neural networks
		"Database performance tips",      // Should match database optimization
		"Quantum computing explained",    // Should not match anything
	}

	for _, query := range searchQueries {
		fmt.Printf("\nSearching for: '%s'\n", query)
		
		searchReq := map[string]interface{}{
			"query":     query,
			"limit":     5,
			"threshold": 0.7,
		}

		reqBody, _ := json.Marshal(searchReq)
		resp, err := http.Post(baseURL+"/cache/search", "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			log.Printf("Error in search: %v", err)
			continue
		}

		var searchResp map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&searchResp)
		resp.Body.Close()

		results := searchResp["results"].([]interface{})
		if len(results) > 0 {
			fmt.Printf("  Found %d results:\n", len(results))
			for i, result := range results {
				r := result.(map[string]interface{})
				fmt.Printf("    %d. Score: %.3f - %s\n", i+1, r["score"], r["key"])
				if metadata, ok := r["metadata"].(map[string]interface{}); ok {
					fmt.Printf("       Category: %v, Difficulty: %v\n", metadata["category"], metadata["difficulty"])
				}
			}
		} else {
			fmt.Printf("  No results found\n")
		}
	}
}

func demoMetrics(baseURL string) {
	resp, err := http.Get(baseURL + "/cache/metrics")
	if err != nil {
		log.Printf("Error getting metrics: %v", err)
		return
	}
	defer resp.Body.Close()

	var metrics map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&metrics)

	fmt.Println("Cache Metrics:")
	fmt.Printf("  Total entries: %v\n", metrics["total_entries"])
	fmt.Printf("  Hit rate: %v\n", metrics["hit_rate"])
	fmt.Printf("  Average response time: %v\n", metrics["average_response_time"])
	fmt.Printf("  Embedding cache hit rate: %v\n", metrics["embedding_cache_hit_rate"])
	fmt.Printf("  Memory usage: %v bytes\n", metrics["memory_usage_bytes"])
	fmt.Printf("  Last cleanup: %v\n", metrics["last_cleanup"])
}

func demoAccessStatistics(baseURL string) {
	// Make several requests to generate access statistics
	queries := []string{
		"What is machine learning?",
		"What is ML?",
		"How do neural networks work?",
		"Neural network basics",
		"What is database optimization?",
		"Database performance tips",
	}

	fmt.Println("Making multiple requests to generate access statistics...")
	for i, query := range queries {
		fmt.Printf("  Request %d: %s\n", i+1, query)
		
		getReq := map[string]interface{}{
			"key": query,
		}

		reqBody, _ := json.Marshal(getReq)
		resp, err := http.Post(baseURL+"/cache/get", "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			log.Printf("Error in get request: %v", err)
			continue
		}

		var getResp map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&getResp)
		resp.Body.Close()

		if getResp["found"].(bool) {
			fmt.Printf("    ✓ Found match\n")
		} else {
			fmt.Printf("    ✗ No match found\n")
		}
	}

	// Get updated metrics
	fmt.Println("\nUpdated metrics after access:")
	demoMetrics(baseURL)
}
