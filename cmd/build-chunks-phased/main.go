package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
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

	t := &HTTPTokenizer{baseURL: baseURL, client: client}

	log.Printf("Waiting for tokenizer service at %s...", baseURL)
	for i := 0; i < 30; i++ {
		resp, err := client.Get(baseURL + "/health")
		if err == nil {
			resp.Body.Close()
			break
		}
		time.Sleep(1 * time.Second)
	}

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
	json.NewDecoder(resp.Body).Decode(&data)

	t.clsID = data.Cls
	t.sepID = data.Sep
	t.padID = data.Pad

	log.Printf("HTTP Tokenizer ready: [CLS]=%d, [SEP]=%d, [PAD]=%d", t.clsID, t.sepID, t.padID)
	return t, nil
}

func (t *HTTPTokenizer) EncodeBatch(texts []string) ([][]int, error) {
	body := struct {
		Texts []string `json:"texts"`
	}{Texts: texts}

	jsonBody, _ := json.Marshal(body)
	resp, err := t.client.Post(t.baseURL+"/tokenize_batch", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Results [][]int `json:"results"`
		Error   string  `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&result)

	if result.Error != "" {
		return nil, fmt.Errorf(result.Error)
	}
	return result.Results, nil
}

func cleanChunkText(text string) string {
	nbspRunes := []rune{'\u00A0', '\u2007', '\u202F', '\u2060', '\uFEFF', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2008', '\u2009', '\u200A', '\u205F', '\u3000'}
	for _, r := range nbspRunes {
		text = strings.ReplaceAll(text, string(r), " ")
	}
	zeroWidthRunes := []rune{'\u200B', '\u200C', '\u200D', '\u00AD', '\u034F', '\u061C', '\u180E', '\uFEFF', '\u202A', '\u202B', '\u202C', '\u202D', '\u202E', '\u2061', '\u2062', '\u2063', '\u2064', '\u2066', '\u2067', '\u2068', '\u2069', '\u206A', '\u206B', '\u206C', '\u206D', '\u206E', '\u206F'}
	for _, r := range zeroWidthRunes {
		text = strings.ReplaceAll(text, string(r), "")
	}
	text = strings.ReplaceAll(text, "…", "...")
	text = strings.ReplaceAll(text, "–", "-")
	text = strings.ReplaceAll(text, "—", "-")
	text = strings.ReplaceAll(text, "―", "-")
	text = strings.ReplaceAll(text, "−", "-")

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
	trainChan chan<- string
	valChan   chan<- string
	valRatio  float64
}

func (b *ChunkBuilder) processBook(bookFile string) (trainChunks, valChunks int, err error) {
	file, err := os.Open(bookFile)
	if err != nil {
		return 0, 0, err
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
		return 0, 0, nil
	}

	// Токенизируем все предложения батчами
	texts := make([]string, len(sentences))
	for i, s := range sentences {
		texts[i] = s
	}

	allIDs := make([][]int, len(texts))
	batchSize := 100
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]
		ids, err := b.tokenizer.EncodeBatch(batch)
		if err != nil {
			return 0, 0, err
		}
		for j, idList := range ids {
			allIDs[i+j] = idList
		}
	}

	// Случайно выбираем val-предложения на уровне предложений
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	isVal := make([]bool, len(sentences))
	for i := range sentences {
		if rng.Float64() < b.valRatio {
			isVal[i] = true
		}
	}

	// Функция для построения чанков из подмножества предложений
	buildChunks := func(selectVal bool) int {
		currentChunk := []string{}
		currentLength := 0
		chunksGenerated := 0

		for i, text := range sentences {
			// Пропускаем предложения, которые не соответствуют выборке
			if selectVal && !isVal[i] {
				continue
			}
			if !selectVal && isVal[i] {
				continue
			}

			ids := allIDs[i]
			sentLen := len(ids)
			if sentLen == 0 || sentLen > b.maxLength-2 {
				continue
			}
			if currentLength+sentLen > b.maxLength-2 {
				if len(currentChunk) > 0 {
					chunkText := strings.Join(currentChunk, " ")
					if selectVal {
						b.valChan <- chunkText
					} else {
						b.trainChan <- chunkText
					}
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
			chunkText := strings.Join(currentChunk, " ")
			if selectVal {
				b.valChan <- chunkText
			} else {
				b.trainChan <- chunkText
			}
			chunksGenerated++
		}
		return chunksGenerated
	}

	trainCount := buildChunks(false) // false = train
	valCount := buildChunks(true)    // true = val

	return trainCount, valCount, nil
}

func buildPhase(books []string, tokenizerURL string, maxLength int, trainOutput, valOutput string, workers int, valRatio float64) error {
	log.Printf("Phase max_length=%d: %d books", maxLength, len(books))
	log.Printf("  Train output: %s", trainOutput)
	log.Printf("  Val output: %s", valOutput)
	log.Printf("  Val ratio: %.1f%%", valRatio*100)
	startTime := time.Now()

	tokenizer, err := NewHTTPTokenizer(tokenizerURL)
	if err != nil {
		return fmt.Errorf("failed to create tokenizer: %w", err)
	}

	// Открываем выходные файлы
	trainFile, err := os.Create(trainOutput)
	if err != nil {
		return err
	}
	defer trainFile.Close()
	trainWriter := bufio.NewWriterSize(trainFile, 8*1024*1024)
	defer trainWriter.Flush()

	valFile, err := os.Create(valOutput)
	if err != nil {
		return err
	}
	defer valFile.Close()
	valWriter := bufio.NewWriterSize(valFile, 8*1024*1024)
	defer valWriter.Flush()

	trainChan := make(chan string, 10000)
	valChan := make(chan string, 10000)

	var totalTrainChunks, totalValChunks int64
	var processedBooks, failedBooks int64

	// Writer для train
	var trainWg sync.WaitGroup
	trainWg.Add(1)
	go func() {
		defer trainWg.Done()
		for chunk := range trainChan {
			trainWriter.WriteString(chunk + "\n")
			atomic.AddInt64(&totalTrainChunks, 1)
		}
	}()

	// Writer для val
	var valWg sync.WaitGroup
	valWg.Add(1)
	go func() {
		defer valWg.Done()
		for chunk := range valChan {
			valWriter.WriteString(chunk + "\n")
			atomic.AddInt64(&totalValChunks, 1)
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
				trainChan: trainChan,
				valChan:   valChan,
				valRatio:  valRatio,
			}
			for book := range tasks {
				bookName := filepath.Base(book)
				trainChunks, valChunks, err := builder.processBook(book)
				if err != nil {
					log.Printf("Worker %d: ERROR on %s: %v", workerID, bookName, err)
					atomic.AddInt64(&failedBooks, 1)
				} else {
					if trainChunks+valChunks > 0 {
						log.Printf("Worker %d: %s -> train=%d, val=%d", workerID, bookName, trainChunks, valChunks)
					}
				}
				processed := atomic.AddInt64(&processedBooks, 1)
				if processed%100 == 0 {
					log.Printf("Phase %d: %d/%d books, train=%d, val=%d chunks, %d failed",
						maxLength, processed, len(books),
						atomic.LoadInt64(&totalTrainChunks), atomic.LoadInt64(&totalValChunks),
						atomic.LoadInt64(&failedBooks))
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
			train := atomic.LoadInt64(&totalTrainChunks)
			val := atomic.LoadInt64(&totalValChunks)
			failed := atomic.LoadInt64(&failedBooks)
			elapsed := time.Since(startTime)
			rate := float64(processed) / elapsed.Seconds()
			log.Printf("Phase %d: %d/%d books (%.1f%%), train=%d, val=%d, %d failed, %.1f books/sec, %v",
				maxLength, processed, len(books),
				float64(processed)/float64(len(books))*100,
				train, val, failed, rate, elapsed.Round(time.Second))
		}
	}()

	wg.Wait()
	close(trainChan)
	close(valChan)
	trainWg.Wait()
	valWg.Wait()
	trainWriter.Flush()
	valWriter.Flush()

	elapsed := time.Since(startTime)
	log.Printf("Phase max_length=%d COMPLETED: train=%d, val=%d chunks, %d failed in %v",
		maxLength, totalTrainChunks, totalValChunks, failedBooks, elapsed.Round(time.Second))

	return nil
}

func main() {
	var (
		cleanedDir   = flag.String("cleaned", "data/cleaned", "директория с JSONL")
		tokenizerURL = flag.String("tokenizer", "http://localhost:8091", "URL сервиса токенизации")
		outputDir    = flag.String("output", "data/bert", "выходная директория")
		workers      = flag.Int("workers", 16, "количество воркеров")
		valRatio     = flag.Float64("val-ratio", 0.02, "доля предложений на валидацию")
	)
	flag.Parse()

	log.Printf("=== Build Chunks Phased (Train/Val Split per Book) ===")
	log.Printf("Cleaned dir: %s", *cleanedDir)
	log.Printf("Tokenizer URL: %s", *tokenizerURL)
	log.Printf("Output dir: %s", *outputDir)
	log.Printf("Workers: %d", *workers)
	log.Printf("Val ratio: %.1f%%", *valRatio*100)

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output dir: %v", err)
	}

	books, err := filepath.Glob(filepath.Join(*cleanedDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("Glob error: %v", err)
	}

	log.Printf("Found %d books", len(books))
	sort.Strings(books)

	// Разбиваем книги на три фазы (разные книги для разных длин)
	n := len(books)
	phaseSize := n / 3

	phase1Books := books[:phaseSize]
	phase2Books := books[phaseSize : 2*phaseSize]
	phase3Books := books[2*phaseSize:]

	log.Printf("Phase 1 (128): %d books", len(phase1Books))
	log.Printf("Phase 2 (256): %d books", len(phase2Books))
	log.Printf("Phase 3 (512): %d books", len(phase3Books))

	// Фаза 1
	if err := buildPhase(phase1Books, *tokenizerURL, 128,
		filepath.Join(*outputDir, "phase1_128_train.txt"),
		filepath.Join(*outputDir, "phase1_128_val.txt"),
		*workers, *valRatio); err != nil {
		log.Fatalf("Phase 1 failed: %v", err)
	}

	// Фаза 2
	if err := buildPhase(phase2Books, *tokenizerURL, 256,
		filepath.Join(*outputDir, "phase2_256_train.txt"),
		filepath.Join(*outputDir, "phase2_256_val.txt"),
		*workers, *valRatio); err != nil {
		log.Fatalf("Phase 2 failed: %v", err)
	}

	// Фаза 3
	if err := buildPhase(phase3Books, *tokenizerURL, 512,
		filepath.Join(*outputDir, "phase3_512_train.txt"),
		filepath.Join(*outputDir, "phase3_512_val.txt"),
		*workers, *valRatio); err != nil {
		log.Fatalf("Phase 3 failed: %v", err)
	}

	log.Printf("=== All phases completed! ===")
}
