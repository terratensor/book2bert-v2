package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Sentence struct {
	Text string `json:"text"`
}

// TokenizerClient — один Python-процесс, который стримит токенизацию
type TokenizerClient struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	mu     sync.Mutex
	clsID  int
	sepID  int
	padID  int
}

func NewTokenizerClient(modelPath string) (*TokenizerClient, error) {
	// Python скрипт, который читает строки и пишет токены
	pythonScript := `
import sys
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(sys.argv[1])

# Читаем специальные токены
cls_id = sp.PieceToId("[CLS]")
sep_id = sp.PieceToId("[SEP]")
pad_id = sp.PieceToId("[PAD]")

# Первая строка — специальные токены
print(f"{cls_id},{sep_id},{pad_id}")
sys.stdout.flush()

# Дальше обрабатываем строки
for line in sys.stdin:
    text = line.strip()
    if not text:
        print("")
        sys.stdout.flush()
        continue
    ids = sp.EncodeAsIds(text)
    print(",".join(map(str, ids)))
    sys.stdout.flush()
`

	cmd := exec.Command("python3", "-c", pythonScript, modelPath)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(stdout)

	// Читаем первую строку — специальные токены
	if !scanner.Scan() {
		return nil, fmt.Errorf("failed to read special tokens")
	}

	parts := strings.Split(scanner.Text(), ",")
	if len(parts) != 3 {
		return nil, fmt.Errorf("invalid special tokens: %s", scanner.Text())
	}

	clsID, _ := strconv.Atoi(parts[0])
	sepID, _ := strconv.Atoi(parts[1])
	padID, _ := strconv.Atoi(parts[2])

	log.Printf("Tokenizer ready: [CLS]=%d, [SEP]=%d, [PAD]=%d", clsID, sepID, padID)

	return &TokenizerClient{
		cmd:    cmd,
		stdin:  stdin,
		stdout: scanner,
		clsID:  clsID,
		sepID:  sepID,
		padID:  padID,
	}, nil
}

func (t *TokenizerClient) EncodeAsIds(text string) ([]int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	text = strings.ReplaceAll(text, "\n", " ")
	text = strings.TrimSpace(text)

	if _, err := io.WriteString(t.stdin, text+"\n"); err != nil {
		return nil, err
	}

	if !t.stdout.Scan() {
		return nil, t.stdout.Err()
	}

	line := t.stdout.Text()
	if line == "" {
		return []int{}, nil
	}

	parts := strings.Split(line, ",")
	ids := make([]int, len(parts))
	for i, p := range parts {
		ids[i], _ = strconv.Atoi(p)
	}

	return ids, nil
}

func (t *TokenizerClient) Close() error {
	t.stdin.Close()
	return t.cmd.Wait()
}

func (t *TokenizerClient) ClsID() int { return t.clsID }
func (t *TokenizerClient) SepID() int { return t.sepID }

// TokenizerPool — пул клиентов (по одному на воркера)
type TokenizerPool struct {
	modelPath string
	mu        sync.Mutex
	clients   []*TokenizerClient
}

func NewTokenizerPool(modelPath string, size int) (*TokenizerPool, error) {
	pool := &TokenizerPool{
		modelPath: modelPath,
		clients:   make([]*TokenizerClient, size),
	}

	// Создаём size клиентов
	for i := 0; i < size; i++ {
		client, err := NewTokenizerClient(modelPath)
		if err != nil {
			return nil, fmt.Errorf("client %d: %w", i, err)
		}
		pool.clients[i] = client
	}

	log.Printf("Tokenizer pool created: %d clients", size)
	return pool, nil
}

func (p *TokenizerPool) GetClient(workerID int) *TokenizerClient {
	return p.clients[workerID%len(p.clients)]
}

func (p *TokenizerPool) Close() {
	for _, c := range p.clients {
		c.Close()
	}
}

type ChunkBuilder struct {
	tokenizer *TokenizerClient
	maxLength int
	chunkChan chan<- string
}

func cleanChunkText(text string) string {
	// 1. Неразрывные пробелы → обычный пробел
	nbspRunes := []rune{
		'\u00A0', // &nbsp;
		'\u2007', // Figure space
		'\u202F', // Narrow no-break space
		'\u2060', // Word joiner
		'\uFEFF', // Zero-width no-break space (BOM)
		'\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006',
		'\u2008', '\u2009', '\u200A', '\u205F', '\u3000',
	}
	for _, r := range nbspRunes {
		text = strings.ReplaceAll(text, string(r), " ")
	}

	// 2. Zero-width и невидимые символы → удалить
	zeroWidthRunes := []rune{
		'\u200B',                                         // Zero-width space
		'\u200C',                                         // Zero-width non-joiner
		'\u200D',                                         // Zero-width joiner
		'\u00AD',                                         // Soft hyphen
		'\u00AD',                                         // Soft hyphen (еще раз для надежности)
		'\u034F',                                         // Combining grapheme joiner
		'\u061C',                                         // Arabic letter mark
		'\u180E',                                         // Mongolian vowel separator
		'\uFEFF',                                         // BOM (also zero-width)
		'\u202A', '\u202B', '\u202C', '\u202D', '\u202E', // Directional formatting
		'\u2061', '\u2062', '\u2063', '\u2064', // Invisible operators
		'\u2066', '\u2067', '\u2068', '\u2069', // Directional isolates
		'\u206A', '\u206B', '\u206C', '\u206D', '\u206E', '\u206F', // Deprecated formatting
	}
	for _, r := range zeroWidthRunes {
		text = strings.ReplaceAll(text, string(r), "")
	}

	// 3. Специфичные замены
	text = strings.ReplaceAll(text, "…", "...") // Многоточие
	text = strings.ReplaceAll(text, "–", "-")   // En dash
	text = strings.ReplaceAll(text, "—", "-")   // Em dash
	text = strings.ReplaceAll(text, "―", "-")   // Horizontal bar
	text = strings.ReplaceAll(text, "−", "-")   // Minus sign

	// 4. Удаление управляющих и приватных символов
	var cleaned strings.Builder
	for _, r := range text {
		// Управляющие C0, C1, DEL
		if r < 0x20 || (r >= 0x7F && r <= 0x9F) {
			continue
		}
		// Приватные зоны
		if r >= 0xE000 && r <= 0xF8FF {
			continue
		}
		cleaned.WriteRune(r)
	}

	return cleaned.String()
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

		// === ПОРЯДОК ОЧИСТКИ ===
		text := s.Text

		// 1. Очистка проблемных Unicode-символов
		text = cleanChunkText(text)

		// 2. Замена переносов строк на пробелы
		text = strings.ReplaceAll(text, "\n", " ")
		text = strings.ReplaceAll(text, "\r", " ")

		// 3. Сжатие пробелов
		text = strings.Join(strings.Fields(text), " ")

		if text != "" {
			sentences = append(sentences, text)
		}
	}

	if len(sentences) == 0 {
		return 0, nil
	}

	chunksGenerated := 0
	currentChunk := []string{}
	currentLength := 0

	for _, text := range sentences {
		ids, err := b.tokenizer.EncodeAsIds(text)
		if err != nil {
			continue
		}
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

func buildPhase(books []string, pool *TokenizerPool, maxLength int, outputPath string, workers int) error {
	log.Printf("Phase max_length=%d: %d books, output=%s", maxLength, len(books), outputPath)
	startTime := time.Now()

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

			client := pool.GetClient(workerID)
			builder := &ChunkBuilder{
				tokenizer: client,
				maxLength: maxLength,
				chunkChan: chunkChan,
			}

			for book := range tasks {
				if _, err := builder.processBook(book); err != nil {
					log.Printf("Worker %d: error %s: %v", workerID, book, err)
				}

				processed := atomic.AddInt64(&processedBooks, 1)
				if processed%100 == 0 {
					log.Printf("Phase %d: %d/%d books, %d chunks",
						maxLength, processed, len(books), atomic.LoadInt64(&totalChunks))
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
			elapsed := time.Since(startTime)
			rate := float64(processed) / elapsed.Seconds()
			log.Printf("Phase %d: %d/%d books (%.1f%%), %d chunks, %.1f books/sec, elapsed: %v",
				maxLength, processed, len(books),
				float64(processed)/float64(len(books))*100,
				chunks, rate, elapsed.Round(time.Second))
		}
	}()

	wg.Wait()
	close(chunkChan)
	writerWg.Wait()
	writer.Flush()

	elapsed := time.Since(startTime)
	log.Printf("Phase max_length=%d COMPLETED: %d chunks in %v",
		maxLength, totalChunks, elapsed.Round(time.Second))

	return nil
}

func main() {
	var (
		cleanedDir    = flag.String("cleaned", "data/cleaned", "директория с JSONL")
		tokenizerPath = flag.String("tokenizer", "models/tokenizer/final/sp_100k.model", "путь к модели")
		outputDir     = flag.String("output", "data/bert", "выходная директория")
		workers       = flag.Int("workers", 32, "количество воркеров")
	)
	flag.Parse()

	log.Printf("=== Build Chunks Phased (Streaming Python) ===")
	log.Printf("Cleaned dir: %s", *cleanedDir)
	log.Printf("Tokenizer: %s", *tokenizerPath)
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

	// Создаём пул токенизаторов (по одному на воркера)
	pool, err := NewTokenizerPool(*tokenizerPath, *workers)
	if err != nil {
		log.Fatalf("Failed to create tokenizer pool: %v", err)
	}
	defer pool.Close()

	if err := buildPhase(phase1Books, pool, 128,
		filepath.Join(*outputDir, "phase1_128.txt"), *workers); err != nil {
		log.Fatalf("Phase 1 failed: %v", err)
	}

	if err := buildPhase(phase2Books, pool, 256,
		filepath.Join(*outputDir, "phase2_256.txt"), *workers); err != nil {
		log.Fatalf("Phase 2 failed: %v", err)
	}

	if err := buildPhase(phase3Books, pool, 512,
		filepath.Join(*outputDir, "phase3_512.txt"), *workers); err != nil {
		log.Fatalf("Phase 3 failed: %v", err)
	}

	log.Printf("=== All phases completed! ===")
}
