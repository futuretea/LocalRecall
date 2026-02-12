package reranker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/mudler/localrecall/rag/types"
)

// Reranker reorders search results to improve ranking quality.
type Reranker interface {
	Rerank(query string, results []types.Result, topN int) ([]types.Result, error)
}

// ---------------------------------------------------------------------------
// KeywordReranker — built-in, zero external dependencies
// ---------------------------------------------------------------------------

// KeywordReranker combines the original vector similarity score with a
// term-overlap keyword score. The combined score is:
//
//	combinedScore = vectorWeight * similarity + keywordWeight * keywordScore
type KeywordReranker struct {
	VectorWeight  float64
	KeywordWeight float64
}

// NewKeywordReranker creates a KeywordReranker with the given weights.
// Default weights (0.7 vector, 0.3 keyword) are used when both are zero.
func NewKeywordReranker(vectorWeight, keywordWeight float64) *KeywordReranker {
	if vectorWeight == 0 && keywordWeight == 0 {
		vectorWeight = 0.7
		keywordWeight = 0.3
	}
	return &KeywordReranker{
		VectorWeight:  vectorWeight,
		KeywordWeight: keywordWeight,
	}
}

func (kr *KeywordReranker) Rerank(query string, results []types.Result, topN int) ([]types.Result, error) {
	queryTerms := normalizeTerms(query)
	if len(queryTerms) == 0 || len(results) == 0 {
		return truncate(results, topN), nil
	}

	type scored struct {
		result   types.Result
		combined float64
	}

	items := make([]scored, len(results))
	for i, r := range results {
		kwScore := termOverlap(queryTerms, normalizeTerms(r.Content))
		combined := kr.VectorWeight*float64(r.Similarity) + kr.KeywordWeight*kwScore
		items[i] = scored{result: r, combined: combined}
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].combined > items[j].combined
	})

	out := make([]types.Result, 0, min(topN, len(items)))
	for i := 0; i < len(items) && i < topN; i++ {
		r := items[i].result
		r.Similarity = float32(items[i].combined)
		out = append(out, r)
	}
	return out, nil
}

// normalizeTerms lower-cases and deduplicates whitespace-separated tokens.
func normalizeTerms(text string) map[string]struct{} {
	words := strings.Fields(strings.ToLower(text))
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

// termOverlap returns |intersection| / sqrt(|query| * |doc|), a TF-free cosine-like score in [0,1].
func termOverlap(query, doc map[string]struct{}) float64 {
	if len(query) == 0 || len(doc) == 0 {
		return 0
	}
	var overlap int
	for q := range query {
		if _, ok := doc[q]; ok {
			overlap++
		}
	}
	return float64(overlap) / math.Sqrt(float64(len(query))*float64(len(doc)))
}

// ---------------------------------------------------------------------------
// ExternalReranker — calls a /rerank HTTP endpoint (Cohere/Jina format)
// ---------------------------------------------------------------------------

// ExternalReranker calls an external rerank API (Cohere/Jina compatible format).
type ExternalReranker struct {
	URL    string
	APIKey string
	Model  string
	Client *http.Client
}

// NewExternalReranker creates an ExternalReranker with the given URL and optional API key/model.
func NewExternalReranker(url, apiKey, model string) *ExternalReranker {
	return &ExternalReranker{
		URL:    url,
		APIKey: apiKey,
		Model:  model,
		Client: &http.Client{Timeout: 30 * time.Second},
	}
}

type rerankRequest struct {
	Model     string   `json:"model,omitempty"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

func (er *ExternalReranker) Rerank(query string, results []types.Result, topN int) ([]types.Result, error) {
	if len(results) == 0 {
		return results, nil
	}

	docs := make([]string, len(results))
	for i, r := range results {
		docs[i] = r.Content
	}

	body := rerankRequest{
		Model:     er.Model,
		Query:     query,
		Documents: docs,
		TopN:      topN,
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal rerank request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, er.URL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if er.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+er.APIKey)
	}

	resp, err := er.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rerank API returned status %d", resp.StatusCode)
	}

	var rr rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&rr); err != nil {
		return nil, fmt.Errorf("failed to decode rerank response: %w", err)
	}

	// Sort by relevance score descending.
	sort.Slice(rr.Results, func(i, j int) bool {
		return rr.Results[i].RelevanceScore > rr.Results[j].RelevanceScore
	})

	out := make([]types.Result, 0, min(topN, len(rr.Results)))
	for i := 0; i < len(rr.Results) && i < topN; i++ {
		idx := rr.Results[i].Index
		if idx < 0 || idx >= len(results) {
			continue
		}
		r := results[idx]
		r.Similarity = float32(rr.Results[i].RelevanceScore)
		out = append(out, r)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func truncate(results []types.Result, n int) []types.Result {
	if n <= 0 || n >= len(results) {
		return results
	}
	return results[:n]
}
