package main

import (
	"os"
	"strconv"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/mudler/localrecall/rag"
	"github.com/mudler/localrecall/rag/reranker"
	"github.com/mudler/localrecall/rag/sources"
	"github.com/sashabaranov/go-openai"
)

var (
	collectionDBPath = os.Getenv("COLLECTION_DB_PATH")
	embeddingModel   = os.Getenv("EMBEDDING_MODEL")
	fileAssets       = os.Getenv("FILE_ASSETS")
	openAIKey        = os.Getenv("OPENAI_API_KEY")
	openAIBaseURL    = os.Getenv("OPENAI_BASE_URL")
	listeningAddress = os.Getenv("LISTENING_ADDRESS")
	vectorEngine     = os.Getenv("VECTOR_ENGINE")
	maxChunkingSize  = os.Getenv("MAX_CHUNKING_SIZE")
	chunkOverlap     = os.Getenv("CHUNK_OVERLAP")
	apiKeys          = os.Getenv("API_KEYS")
	chunkStrategy    = os.Getenv("CHUNKING_STRATEGY")
	rerankerEnabled  = os.Getenv("RERANKER_ENABLED")
	rerankerURL      = os.Getenv("RERANKER_URL")
	rerankerAPIKey   = os.Getenv("RERANKER_API_KEY")
	rerankerModel    = os.Getenv("RERANKER_MODEL")
	rerankerVecW     = os.Getenv("RERANKER_VECTOR_WEIGHT")
	rerankerKwW      = os.Getenv("RERANKER_KEYWORD_WEIGHT")
	gitPrivateKey    = os.Getenv("GIT_PRIVATE_KEY")
	sourceManager    = rag.NewSourceManager(&sources.Config{
		GitPrivateKey: gitPrivateKey,
	})
)

func init() {
	if collectionDBPath == "" {
		collectionDBPath = "collections"
	}
	os.MkdirAll(collectionDBPath, 0755)

	if fileAssets == "" {
		fileAssets = "assets"
		os.MkdirAll(fileAssets, 0755)
	}

	if listeningAddress == "" {
		listeningAddress = ":8080"
	}

	if vectorEngine == "" {
		vectorEngine = "chromem"
	}

	// Start the source manager
	sourceManager.Start()
}

func startAPI(listenAddress string) {
	e := echo.New()
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	config := openai.DefaultConfig(openAIKey)
	config.BaseURL = openAIBaseURL

	openAIClient := openai.NewClientWithConfig(config)

	registerStaticHandler(e)

	keys := []string{}
	if apiKeys != "" {
		keys = strings.Split(apiKeys, ",")
	}

	chunkingSize := 400
	if maxChunkingSize != "" {
		var err error
		chunkingSize, err = strconv.Atoi(maxChunkingSize)
		if err != nil {
			e.Logger.Fatal("Failed to convert MAX_CHUNKING_SIZE to integer")
		}
	}

	overlap := 0
	if chunkOverlap != "" {
		var err error
		overlap, err = strconv.Atoi(chunkOverlap)
		if err != nil {
			e.Logger.Fatal("Failed to convert CHUNK_OVERLAP to integer")
		}
	}

	// Build reranker (nil when disabled — preserves original behaviour).
	var rr reranker.Reranker
	if strings.EqualFold(rerankerEnabled, "true") {
		if rerankerURL != "" {
			rr = reranker.NewExternalReranker(rerankerURL, rerankerAPIKey, rerankerModel)
		} else {
			vw, kw := 0.0, 0.0
			if rerankerVecW != "" {
				vw, _ = strconv.ParseFloat(rerankerVecW, 64)
			}
			if rerankerKwW != "" {
				kw, _ = strconv.ParseFloat(rerankerKwW, 64)
			}
			rr = reranker.NewKeywordReranker(vw, kw)
		}
	}

	registerAPIRoutes(e, openAIClient, chunkingSize, overlap, keys, chunkStrategy, rr)

	e.Logger.Fatal(e.Start(listenAddress))
}

func main() {
	startAPI(listeningAddress)
}
