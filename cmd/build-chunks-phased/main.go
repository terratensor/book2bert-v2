package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Sentence struct {
	Text string `json:"text"`
}

// HTTPTokenizer — клиент к HTTP-сервису токенизации
type HTTPTokenizer struct {
	baseURL string
	client  *http.Client
	clsID   int
	sepID   int
	padID   int
}

func NewHTTPTokenizer(baseURL string) (*HTTPTokenizer, error) {
	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 100,
			MaxIdleConns:        100,
		},
	}

	t := &HTTPTokenizer{
		baseURL: baseURL,
		client:  client,
	}

	// Ждем готовности сервиса
	log.Printf("Waiting for tokenizer service at %s...", baseURL)
	for i := 0; i < 30; i++ {
		resp, err := client.Get(baseURL + "/health")
		if err == nil {
			resp.Body.Close()
			break
		}
		time.Sleep(1 * time.Second)
	}

	// Получаем специальные токены
	resp, err := client.Get(baseURL + "/special_tokens")
	if err != nil {
		return nil, fmt.Errorf("failed to get special tokens: %w", err)
	}
	defer resp.Body.Close()

	var data struct {
		Cls int `json:"cls"`
		Sep int `json:"sep"`
		Pad int `json:"pad"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to decode special tokens: %w", err)
	}

	t.clsID = data.Cls
	t.sepID = data.Sep
	t.padID = data.Pad

	log.Printf("HTTP Tokenizer ready: [CLS]=%d, [SEP]=%d, [PAD]=%d", t.clsID, t.sepID, t.padID)
	return t, nil
}

func (t *HTTPTokenizer) EncodeAsIds(text string) ([]int, error) {
	body := struct {
		Text string `json:"text"`
	}{Text: text}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	resp, err := t.client.Post(t.baseURL+"/tokenize", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		IDs   []int  `json:"ids"`
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Error != "" {
		return nil, fmt.Errorf(result.Error)
	}

	return result.IDs, nil
}

func (t *HTTPTokenizer) EncodeBatch(texts []string) ([][]int, error) {
	body := struct {
		Texts []string `json:"texts"`
	}{Texts: texts}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	resp, err := t.client.Post(t.baseURL+"/tokenize_batch", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Results [][]int `json:"results"`
		Error   string  `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Error != "" {
		return nil, fmt.Errorf(result.Error)
	}

	return result.Results, nil
}

func (t *HTTPTokenizer) ClsID() int { return t.clsID }
func (t *HTTPTokenizer) SepID() int { return t.sepID }

func cleanChunkText(text string) string {
	// 1. Неразрывные пробелы → обычный пробел
	nbspRunes := []rune{
		'\u00A0', '\u2007', '\u202F', '\u2060', '\uFEFF',
		'\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006',
		'\u2008', '\u2009', '\u200A', '\u205F', '\u3000',
	}
	for _, r := range nbspRunes {
		text = strings.ReplaceAll(text, string(r), " ")
	}

	// 2. Zero-width и невидимые символы → удалить
	zeroWidthRunes := []rune{
		'\u200B', '\u200C', '\u200D', '\u00AD', '\u034F', '\u061C', '\u180E',
		'\uFEFF', '\u202A', '\u202B', '\u202C', '\u202D', '\u202E',
		'\u2061', '\u2062', '\u2063', '\u2064',
		'\u2066', '\u2067', '\u2068', '\u2069',
		'\u206A', '\u206B', '\u206C', '\u206D', '\u206E', '\u206F',
	}
	for _, r := range zeroWidthRunes {
		text = strings.ReplaceAll(text, string(r), "")
	}

	// 3. Специфичные замены
	text = strings.ReplaceAll(text, "…", "...")
	text = strings.ReplaceAll(text, "–", "-")
	text = strings.ReplaceAll(text, "—", "-")
	text = strings.ReplaceAll(text, "―", "-")
	text = strings.ReplaceAll(text, "−", "-")

	// 4. Удаление управляющих и приватных символов
	var cleaned strings.Builder
	for _, r := range text {
		if r < 0x20 || (r >= 0x7F && r <= 0x9F) {
			continue
		}
		if r >= 0xE000 && r <= 0xF8FF {
			continue
		}
		cleaned.WriteRune(r)
	}

	return cleaned.String()
}

type ChunkBuilder struct {
	tokenizer *HTTPTokenizer
	maxLength int
	chunkChan chan<- string
}

func (b *ChunkBuilder) processBook(bookFile string) (int, error) {
	file, err := os.Open(bookFile)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	var sentences []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var s Sentence
		if err := json.Unmarshal(line, &s); err != nil {
			continue
		}

		text := cleanChunkText(s.Text)
		text = strings.ReplaceAll(text, "\n", " ")
		text = strings.ReplaceAll(text, "\r", " ")
		text = strings.Join(strings.Fields(text), " ")

		if text != "" {
			sentences = append(sentences, text)
		}
	}

	if len(sentences) == 0 {
		return 0, nil
	}

	// Собираем все тексты для пакетной токенизации
	texts := make([]string, len(sentences))
	for i, s := range sentences {
		texts[i] = s
	}

	// Пакетная токенизация (батчами по 100)
	batchSize := 100
	allIDs := make([][]int, len(texts))

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		ids, err := b.tokenizer.EncodeBatch(batch)
		if err != nil {
			log.Printf("ERROR: batch tokenize failed: %v, falling back to single", err)
			// Fallback: по одному
			for j, text := range batch {
				ids, _ := b.tokenizer.EncodeAsIds(text)
				allIDs[i+j] = ids
			}
		} else {
			for j, idList := range ids {
				allIDs[i+j] = idList
			}
		}
	}

	chunksGenerated := 0
	currentChunk := []string{}
	currentLength := 0

	for i, text := range sentences {
		ids := allIDs[i]
		sentLen := len(ids)

		if sentLen == 0 {
			continue
		}

		if sentLen > b.maxLength-2 {
			continue
		}

		if currentLength+sentLen > b.maxLength-2 {
			if len(currentChunk) > 0 {
				b.chunkChan <- strings.Join(currentChunk, " ")
				chunksGenerated++
			}
			currentChunk = []string{text}
			currentLength = sentLen
		} else {
			currentChunk = append(currentChunk, text)
			currentLength += sentLen
		}
	}

	if len(currentChunk) > 0 {
		b.chunkChan <- strings.Join(currentChunk, " ")
		chunksGenerated++
	}

	return chunksGenerated, nil
}

func buildPhase(books []string, tokenizerURL string, maxLength int, outputPath string, workers int) error {
	log.Printf("Phase max_length=%d: %d books, output=%s", maxLength, len(books), outputPath)
	startTime := time.Now()

	// Создаем ОДИН HTTP-клиент (общий для всех воркеров)
	tokenizer, err := NewHTTPTokenizer(tokenizerURL)
	if err != nil {
		return fmt.Errorf("failed to create tokenizer: %w", err)
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 8*1024*1024)
	defer writer.Flush()

	chunkChan := make(chan string, 10000)
	var totalChunks int64
	var processedBooks int64
	var failedBooks int64

	var writerWg sync.WaitGroup
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		for chunk := range chunkChan {
			writer.WriteString(chunk + "\n")
			atomic.AddInt64(&totalChunks, 1)
		}
	}()

	tasks := make(chan string, len(books))
	for _, b := range books {
		tasks <- b
	}
	close(tasks)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			builder := &ChunkBuilder{
				tokenizer: tokenizer,
				maxLength: maxLength,
				chunkChan: chunkChan,
			}

			for book := range tasks {
				bookName := filepath.Base(book)

				chunks, err := builder.processBook(book)
				if err != nil {
					log.Printf("Worker %d: ERROR on %s: %v (skipping)", workerID, bookName, err)
					atomic.AddInt64(&failedBooks, 1)
				} else {
					if chunks > 0 {
						log.Printf("Worker %d: done %s, %d chunks", workerID, bookName, chunks)
					}
				}

				processed := atomic.AddInt64(&processedBooks, 1)
				if processed%100 == 0 {
					log.Printf("Phase %d: %d/%d books, %d chunks, %d failed",
						maxLength, processed, len(books),
						atomic.LoadInt64(&totalChunks), atomic.LoadInt64(&failedBooks))
				}
			}
		}(i)
	}

	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			processed := atomic.LoadInt64(&processedBooks)
			if processed >= int64(len(books)) {
				return
			}
			chunks := atomic.LoadInt64(&totalChunks)
			failed := atomic.LoadInt64(&failedBooks)
			elapsed := time.Since(startTime)
			rate := float64(processed) / elapsed.Seconds()
			log.Printf("Phase %d: %d/%d books (%.1f%%), %d chunks, %d failed, %.1f books/sec, elapsed: %v",
				maxLength, processed, len(books),
				float64(processed)/float64(len(books))*100,
				chunks, failed, rate, elapsed.Round(time.Second))
		}
	}()

	wg.Wait()
	close(chunkChan)
	writerWg.Wait()
	writer.Flush()

	elapsed := time.Since(startTime)
	failed := atomic.LoadInt64(&failedBooks)
	log.Printf("Phase max_length=%d COMPLETED: %d chunks, %d failed books in %v",
		maxLength, totalChunks, failed, elapsed.Round(time.Second))

	return nil
}

func main() {
	var (
		cleanedDir   = flag.String("cleaned", "data/cleaned", "директория с JSONL")
		tokenizerURL = flag.String("tokenizer", "http://localhost:8093", "URL сервиса токенизации")
		outputDir    = flag.String("output", "data/bert", "выходная директория")
		workers      = flag.Int("workers", 16, "количество воркеров")
	)
	flag.Parse()

	log.Printf("=== Build Chunks Phased (HTTP Tokenizer) ===")
	log.Printf("Cleaned dir: %s", *cleanedDir)
	log.Printf("Tokenizer URL: %s", *tokenizerURL)
	log.Printf("Output dir: %s", *outputDir)
	log.Printf("Workers: %d", *workers)

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output dir: %v", err)
	}

	books, err := filepath.Glob(filepath.Join(*cleanedDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("Glob error: %v", err)
	}

	log.Printf("Found %d books", len(books))
	sort.Strings(books)

	n := len(books)
	phaseSize := n / 3

	phase1Books := books[:phaseSize]
	phase2Books := books[phaseSize : 2*phaseSize]
	phase3Books := books[2*phaseSize:]

	log.Printf("Phase 1 (128): %d books", len(phase1Books))
	log.Printf("Phase 2 (256): %d books", len(phase2Books))
	log.Printf("Phase 3 (512): %d books", len(phase3Books))

	if err := buildPhase(phase1Books, *tokenizerURL, 128,
		filepath.Join(*outputDir, "phase1_128.txt"), *workers); err != nil {
		log.Fatalf("Phase 1 failed: %v", err)
	}

	if err := buildPhase(phase2Books, *tokenizerURL, 256,
		filepath.Join(*outputDir, "phase2_256.txt"), *workers); err != nil {
		log.Fatalf("Phase 2 failed: %v", err)
	}

	if err := buildPhase(phase3Books, *tokenizerURL, 512,
		filepath.Join(*outputDir, "phase3_512.txt"), *workers); err != nil {
		log.Fatalf("Phase 3 failed: %v", err)
	}

	log.Printf("=== All phases completed! ===")
}
