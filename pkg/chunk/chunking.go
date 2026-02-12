package chunk

import (
	"strings"
)

// Options configures paragraph chunking.
type Options struct {
	// MaxSize is the maximum characters per chunk (required, must be > 0).
	MaxSize int
	// Overlap is the overlap in characters between consecutive chunks, word-aligned (0 = no overlap).
	// Must be < MaxSize; values >= MaxSize are clamped to MaxSize-1.
	Overlap int
	// SplitLongWords, when true, splits words longer than MaxSize into smaller chunks so no chunk exceeds MaxSize (default true).
	SplitLongWords bool
}

// splitLongString splits s into pieces of at most maxSize characters.
// Returns a slice of substrings; each has length <= maxSize.
func splitLongString(s string, maxSize int) []string {
	if maxSize <= 0 || len(s) <= maxSize {
		return []string{s}
	}
	var pieces []string
	for len(s) > 0 {
		n := maxSize
		if n > len(s) {
			n = len(s)
		}
		pieces = append(pieces, s[:n])
		s = s[n:]
	}
	return pieces
}

// overlapTail returns the suffix of chunk that is at most overlap characters and word-aligned (whole words only).
// If overlap is 0 or chunk is empty, returns "".
func overlapTail(chunk string, overlap int) string {
	if overlap <= 0 || chunk == "" {
		return ""
	}
	words := strings.Fields(chunk)
	if len(words) == 0 {
		return ""
	}
	// Take words from the end until we would exceed overlap (length includes spaces between words).
	var tail []string
	length := 0
	for i := len(words) - 1; i >= 0; i-- {
		w := words[i]
		addLen := len(w)
		if len(tail) > 0 {
			addLen++ // space before this word
		}
		if length+addLen > overlap {
			break
		}
		tail = append([]string{w}, tail...)
		length += addLen
	}
	return strings.Join(tail, " ")
}

// SplitParagraphIntoChunksWithOptions splits a paragraph into chunks according to opts.
// Chunks are word-boundary aligned; consecutive chunks may overlap by opts.Overlap characters (word-aligned).
// Words longer than opts.MaxSize are split into smaller chunks when opts.SplitLongWords is true.
func SplitParagraphIntoChunksWithOptions(paragraph string, opts Options) []string {
	maxSize := opts.MaxSize
	if maxSize <= 0 {
		maxSize = 1
	}
	overlap := opts.Overlap
	if overlap >= maxSize {
		overlap = maxSize - 1
	}
	if overlap < 0 {
		overlap = 0
	}
	splitLongWords := opts.SplitLongWords

	// Empty or single-chunk within limit (no overlap needed)
	if paragraph == "" {
		return []string{""}
	}
	if len(paragraph) <= maxSize && overlap == 0 {
		words := strings.Fields(paragraph)
		needSplit := false
		for _, w := range words {
			if len(w) > maxSize && splitLongWords {
				needSplit = true
				break
			}
		}
		if !needSplit {
			return []string{paragraph}
		}
	}

	words := strings.Fields(paragraph)
	var chunks []string
	var currentChunk strings.Builder
	var overlapPrefix string // word-aligned prefix for next chunk (from previous chunk's tail)

	for _, word := range words {
		// Long word: split into pieces when SplitLongWords is true
		if len(word) > maxSize && splitLongWords {
			// Flush current chunk first
			if currentChunk.Len() > 0 {
				chunks = append(chunks, currentChunk.String())
				if overlap > 0 {
					overlapPrefix = overlapTail(currentChunk.String(), overlap)
				} else {
					overlapPrefix = ""
				}
				currentChunk.Reset()
			}
			pieces := splitLongString(word, maxSize)
			for _, p := range pieces {
				chunks = append(chunks, p)
				if overlap > 0 {
					overlapPrefix = overlapTail(p, overlap)
				}
			}
			continue
		}

		// Normal word: compute length if we add this word
		var nextLen int
		if currentChunk.Len() > 0 {
			nextLen = currentChunk.Len() + 1 + len(word)
		} else if overlapPrefix != "" {
			nextLen = len(overlapPrefix) + 1 + len(word)
		} else {
			nextLen = len(word)
		}

		if nextLen > maxSize {
			// Flush current chunk
			if currentChunk.Len() > 0 {
				chunks = append(chunks, currentChunk.String())
				if overlap > 0 {
					overlapPrefix = overlapTail(currentChunk.String(), overlap)
				} else {
					overlapPrefix = ""
				}
				currentChunk.Reset()
			}
			// Start new chunk with overlap prefix only if it fits with the word
			if overlapPrefix != "" && len(overlapPrefix)+1+len(word) <= maxSize {
				currentChunk.WriteString(overlapPrefix)
				currentChunk.WriteString(" ")
				currentChunk.WriteString(word)
				overlapPrefix = ""
			} else {
				currentChunk.WriteString(word)
				overlapPrefix = ""
			}
		} else {
			if currentChunk.Len() == 0 && overlapPrefix != "" {
				currentChunk.WriteString(overlapPrefix)
				currentChunk.WriteString(" ")
				currentChunk.WriteString(word)
				overlapPrefix = ""
			} else if currentChunk.Len() > 0 {
				currentChunk.WriteString(" ")
				currentChunk.WriteString(word)
			} else {
				currentChunk.WriteString(word)
			}
		}
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

// defaultSeparators is the hierarchy of separators used by recursive splitting.
// Each level attempts to preserve larger semantic boundaries first.
var defaultSeparators = []string{"\n\n", "\n", ". ", " "}

// RecursiveSplitWithOptions splits text by trying separators in order of
// decreasing granularity (paragraph → line → sentence → word). Segments that
// fit within opts.MaxSize are kept as-is; oversized segments are recursively
// split with the next separator. This preserves natural boundaries better than
// fixed-size word-boundary splitting.
//
// The function reuses overlapTail and splitLongString for overlap and fallback.
func RecursiveSplitWithOptions(text string, opts Options) []string {
	maxSize := opts.MaxSize
	if maxSize <= 0 {
		maxSize = 1
	}
	overlap := opts.Overlap
	if overlap >= maxSize {
		overlap = maxSize - 1
	}
	if overlap < 0 {
		overlap = 0
	}

	if text == "" {
		return []string{""}
	}
	if len(text) <= maxSize {
		return []string{text}
	}

	chunks := recursiveSplit(text, defaultSeparators, maxSize, opts.SplitLongWords)
	if overlap == 0 || len(chunks) <= 1 {
		return chunks
	}

	// Apply overlap between consecutive chunks.
	result := make([]string, 0, len(chunks))
	for i, c := range chunks {
		if i == 0 {
			result = append(result, c)
			continue
		}
		tail := overlapTail(chunks[i-1], overlap)
		if tail != "" && len(tail)+1+len(c) <= maxSize {
			result = append(result, tail+" "+c)
		} else if tail != "" && len(tail) < len(c) {
			// Overlap would exceed maxSize — try trimming c from the end
			// to fit. If that's not possible, just keep c as-is.
			result = append(result, c)
		} else {
			result = append(result, c)
		}
	}
	return result
}

// recursiveSplit is the core recursive splitting logic.
func recursiveSplit(text string, separators []string, maxSize int, splitLong bool) []string {
	if len(text) <= maxSize {
		return []string{text}
	}

	// No separators left — fall back to hard character split.
	if len(separators) == 0 {
		if splitLong {
			return splitLongString(text, maxSize)
		}
		return []string{text}
	}

	sep := separators[0]
	rest := separators[1:]
	parts := strings.Split(text, sep)

	// If the separator didn't actually split anything, try the next one.
	if len(parts) == 1 {
		return recursiveSplit(text, rest, maxSize, splitLong)
	}

	var chunks []string
	var current strings.Builder

	for i, part := range parts {
		if part == "" && i < len(parts)-1 {
			// Preserve separator in accumulation (e.g. consecutive \n\n).
			if current.Len() > 0 {
				current.WriteString(sep)
			}
			continue
		}

		// What the chunk would look like if we append this part.
		var candidate int
		if current.Len() > 0 {
			candidate = current.Len() + len(sep) + len(part)
		} else {
			candidate = len(part)
		}

		if candidate <= maxSize {
			if current.Len() > 0 {
				current.WriteString(sep)
			}
			current.WriteString(part)
		} else {
			// Flush accumulated text.
			if current.Len() > 0 {
				chunks = append(chunks, current.String())
				current.Reset()
			}
			// If this single part exceeds maxSize, recurse with next separator.
			if len(part) > maxSize {
				sub := recursiveSplit(part, rest, maxSize, splitLong)
				chunks = append(chunks, sub...)
			} else {
				current.WriteString(part)
			}
		}
	}

	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	return chunks
}

// SplitParagraphIntoChunks takes a paragraph and a maxChunkSize as input,
// and returns a slice of strings where each string is a chunk of the paragraph
// that is at most maxChunkSize long, ensuring that words are not split.
// Words longer than maxChunkSize are split into smaller chunks.
// For overlap and other options, use SplitParagraphIntoChunksWithOptions.
func SplitParagraphIntoChunks(paragraph string, maxChunkSize int) []string {
	return SplitParagraphIntoChunksWithOptions(paragraph, Options{
		MaxSize:        maxChunkSize,
		Overlap:        0,
		SplitLongWords: true,
	})
}
