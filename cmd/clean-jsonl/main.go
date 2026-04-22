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
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// ============================================================================
// КОНФИГУРАЦИЯ И ТИПЫ
// ============================================================================

// CleanStats хранит статистику очистки
type CleanStats struct {
	Processed             int64
	Kept                  int64
	Removed               int64
	EmptyRemoved          int64
	AdjacentDupesRemoved  int64
	IndexRemoved          int64
	GarbagePatternRemoved int64
	HighDigitRemoved      int64
	OCRGarbageRemoved     int64 // новое
	LanguageRemoved       int64 // новое
}

// LanguageDetectorClient HTTP-клиент к сервису определения языка
type LanguageDetectorClient struct {
	baseURL string
	client  *http.Client
}

// NewLanguageDetectorClient создает нового клиента
func NewLanguageDetectorClient(baseURL string) *LanguageDetectorClient {
	return &LanguageDetectorClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				MaxIdleConnsPerHost: 100,
				MaxIdleConns:        100,
			},
		},
	}
}

// DetectResult результат определения языка
type DetectResult struct {
	Lang          string  `json:"lang"`
	Confidence    float64 `json:"confidence"`
	CyrillicRatio float64 `json:"cyrillic_ratio"`
	Keep          bool    `json:"keep"`
}

// DetectBatch определяет язык для батча текстов
func (c *LanguageDetectorClient) DetectBatch(texts []string) ([]DetectResult, error) {
	body := struct {
		Texts []string `json:"texts"`
	}{Texts: texts}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Post(c.baseURL+"/detect_batch", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Results []DetectResult `json:"results"`
		Error   string         `json:"error,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Error != "" {
		return nil, fmt.Errorf("service error: %s", result.Error)
	}

	return result.Results, nil
}

// ============================================================================
// РЕГУЛЯРКИ ДЛЯ ОЧИСТКИ
// ============================================================================

var (
	urlRegex   = regexp.MustCompile(`https?://[^\s]+`)
	isbnRegex  = regexp.MustCompile(`\bISBN\s*\d{3}-\d{1,5}-\d{1,7}-\d{1,7}-\d{1,7}\b`)
	udkRegex   = regexp.MustCompile(`\bУДК\s*\d+(?:\.\d+)+\b`)
	bbkRegex   = regexp.MustCompile(`\bББК\s*\d+(?:\.\d+)+\b`)
	emailRegex = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)

	// Мусорные паттерны (одиночные буквы/цифры со скобками)
	garbagePattern1 = regexp.MustCompile(`^[\s\[\]\(\)\{\}\d\.\,\;\-]+$`)
	garbagePattern2 = regexp.MustCompile(`^[\[\(]\d+[\]\)]\.?\s*$`)
	garbagePattern3 = regexp.MustCompile(`^[Сс]\.\s*\d+\.?\s*$`)
)

// ============================================================================
// ФУНКЦИИ ОЧИСТКИ
// ============================================================================

// isIndexEntry проверяет, является ли текст предметным указателем
func isIndexEntry(text string) bool {
	lines := strings.Split(text, "\n")
	if len(lines) < 3 {
		return false
	}

	pattern := regexp.MustCompile(`.*?[,\s]+\d+[\s,]*$`)

	matches := 0
	nonEmpty := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			nonEmpty++
			if pattern.MatchString(line) {
				matches++
			}
		}
	}

	if nonEmpty == 0 {
		return false
	}

	return float64(matches)/float64(nonEmpty) > 0.7
}

// hasGarbagePatterns проверяет наличие мусорных паттернов
func hasGarbagePatterns(text string) bool {
	if urlRegex.MatchString(text) ||
		isbnRegex.MatchString(text) ||
		udkRegex.MatchString(text) ||
		bbkRegex.MatchString(text) ||
		emailRegex.MatchString(text) {
		return true
	}

	// Короткие мусорные строки
	trimmed := strings.TrimSpace(text)
	if len([]rune(trimmed)) < 10 {
		if garbagePattern1.MatchString(trimmed) ||
			garbagePattern2.MatchString(trimmed) ||
			garbagePattern3.MatchString(trimmed) {
			return true
		}
	}

	return false
}

// isOCRGarbage проверяет строки с OCR-артефактами
func isOCRGarbage(text string) bool {
	// 1. Разреженные слова: "б а р а б а н щ и к"
	if regexp.MustCompile(`\b\w( \w){4,}\b`).MatchString(text) {
		return true
	}

	// 2. Одиночные буквы, разделенные пробелами, в начале
	if regexp.MustCompile(`^[А-ЯЁ]\s+[а-яё]\s+[а-яё]`).MatchString(text) {
		return true
	}

	// 3. Перемешаны кириллица и латиница в одном "слове"
	if regexp.MustCompile(`\b[а-яё]+[a-z]+[а-яё]*\b`).MatchString(strings.ToLower(text)) {
		return true
	}

	// 4. Слишком много одиночных букв
	words := strings.Fields(text)
	if len(words) > 5 {
		singleLetters := 0
		for _, w := range words {
			if len([]rune(w)) == 1 && unicode.IsLetter([]rune(w)[0]) {
				singleLetters++
			}
		}
		if float64(singleLetters)/float64(len(words)) > 0.5 {
			return true
		}
	}

	// 5. Много не-буквенных символов подряд
	if regexp.MustCompile(`[^а-яё\s]{5,}`).MatchString(strings.ToLower(text)) {
		return true
	}

	return false
}

// digitRatio возвращает долю цифр в тексте
func digitRatio(text string) float64 {
	digits := 0
	runes := []rune(text)
	if len(runes) == 0 {
		return 0
	}
	for _, r := range runes {
		if r >= '0' && r <= '9' {
			digits++
		}
	}
	return float64(digits) / float64(len(runes))
}

// letterRatio возвращает долю букв в тексте
func letterRatio(text string) float64 {
	letters := 0
	runes := []rune(text)
	if len(runes) == 0 {
		return 0
	}
	for _, r := range runes {
		if unicode.IsLetter(r) {
			letters++
		}
	}
	return float64(letters) / float64(len(runes))
}

// cleanSentence очищает одно предложение (без проверки языка)
func cleanSentence(text string) (string, bool, string) {
	text = strings.TrimSpace(text)

	// 1. Пустые
	if len(text) == 0 {
		return "", false, "empty"
	}

	// 2. Мусорные паттерны (URL, ISBN, email, короткий мусор)
	if hasGarbagePatterns(text) {
		return "", false, "garbage_pattern"
	}

	// 3. Предметные указатели
	if isIndexEntry(text) {
		return "", false, "index"
	}

	// 4. OCR-мусор
	if isOCRGarbage(text) {
		return "", false, "ocr_garbage"
	}

	// 5. Только цифры и пунктуация (нет букв)
	lr := letterRatio(text)
	if lr < 0.1 {
		dr := digitRatio(text)
		if dr > 0.5 {
			return "", false, "high_digit_no_letters"
		}
	}

	// 6. Убираем множественные пробелы
	text = regexp.MustCompile(`[ \t]+`).ReplaceAllString(text, " ")

	return text, true, ""
}

// ============================================================================
// ОБРАБОТКА ФАЙЛА
// ============================================================================

// processFile обрабатывает один JSONL файл
func processFile(
	inputPath, outputPath string,
	stats *CleanStats,
	langClient *LanguageDetectorClient,
	filterByLanguage bool,
) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return err
	}
	defer inputFile.Close()

	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return err
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outputFile.Close()

	scanner := bufio.NewScanner(inputFile)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)
	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()

	var sentences []map[string]interface{}
	var prevText string
	var localKept int64

	// Буфер для батчевой проверки языка
	batchSize := 100
	textBatch := make([]string, 0, batchSize)
	sentBatch := make([]map[string]interface{}, 0, batchSize)

	for scanner.Scan() {
		line := scanner.Bytes()
		var sent map[string]interface{}
		if err := json.Unmarshal(line, &sent); err != nil {
			continue
		}

		text, ok := sent["text"].(string)
		if !ok {
			continue
		}

		// Локальная очистка (без проверки языка)
		cleaned, keep, reason := cleanSentence(text)
		if !keep {
			atomic.AddInt64(&stats.Removed, 1)
			switch reason {
			case "empty":
				atomic.AddInt64(&stats.EmptyRemoved, 1)
			case "garbage_pattern":
				atomic.AddInt64(&stats.GarbagePatternRemoved, 1)
			case "index":
				atomic.AddInt64(&stats.IndexRemoved, 1)
			case "high_digit_no_letters":
				atomic.AddInt64(&stats.HighDigitRemoved, 1)
			case "ocr_garbage":
				atomic.AddInt64(&stats.OCRGarbageRemoved, 1)
			}
			continue
		}

		// Проверка на точный дубль подряд
		if cleaned == prevText {
			atomic.AddInt64(&stats.AdjacentDupesRemoved, 1)
			continue
		}

		sent["text"] = cleaned

		if filterByLanguage {
			// Добавляем в батч для проверки языка
			textBatch = append(textBatch, cleaned)
			sentBatch = append(sentBatch, sent)

			if len(textBatch) >= batchSize {
				// Проверяем батч
				results, err := langClient.DetectBatch(textBatch)
				if err != nil {
					log.Printf("ERROR language detection: %v, keeping all", err)
					// При ошибке оставляем все
					for _, s := range sentBatch {
						sentences = append(sentences, s)
						prevText = s["text"].(string)
						localKept++
					}
				} else {
					for i, result := range results {
						if result.Keep {
							sentences = append(sentences, sentBatch[i])
							prevText = sentBatch[i]["text"].(string)
							localKept++
						} else {
							atomic.AddInt64(&stats.Removed, 1)
							atomic.AddInt64(&stats.LanguageRemoved, 1)
						}
					}
				}
				textBatch = textBatch[:0]
				sentBatch = sentBatch[:0]
			}
		} else {
			// Без фильтрации языка
			sentences = append(sentences, sent)
			prevText = cleaned
			localKept++
		}
	}

	// Обрабатываем остатки батча
	if filterByLanguage && len(textBatch) > 0 {
		results, err := langClient.DetectBatch(textBatch)
		if err != nil {
			for _, s := range sentBatch {
				sentences = append(sentences, s)
				localKept++
			}
		} else {
			for i, result := range results {
				if result.Keep {
					sentences = append(sentences, sentBatch[i])
					localKept++
				} else {
					atomic.AddInt64(&stats.Removed, 1)
					atomic.AddInt64(&stats.LanguageRemoved, 1)
				}
			}
		}
	}

	// Перезаписываем position (сохраняем порядок)
	for i, sent := range sentences {
		sent["position"] = i
		data, err := json.Marshal(sent)
		if err != nil {
			continue
		}
		writer.Write(data)
		writer.Write([]byte("\n"))
	}

	atomic.AddInt64(&stats.Processed, 1)
	atomic.AddInt64(&stats.Kept, localKept)

	return scanner.Err()
}

// ============================================================================
// MAIN
// ============================================================================

type Progress struct {
	processed int64
	total     int64
	startTime time.Time
}

func main() {
	var (
		inputDir           = flag.String("input", "", "входная директория с JSONL файлами")
		outputDir          = flag.String("output", "", "выходная директория")
		workers            = flag.Int("workers", 32, "количество воркеров")
		langDetectorURL    = flag.String("lang-detector", "http://localhost:8092", "URL сервиса определения языка")
		filterByLanguage   = flag.Bool("filter-lang", true, "фильтровать по языку (только русский и mixed)")
		skipLanguageFilter = flag.Bool("skip-lang", false, "пропустить фильтрацию по языку")
	)
	flag.Parse()

	if *inputDir == "" || *outputDir == "" {
		log.Fatal("--input and --output are required")
	}

	// Определяем, фильтровать ли по языку
	filterLang := *filterByLanguage && !*skipLanguageFilter

	log.Printf("=== JSONL Cleaner v2 ===")
	log.Printf("Input dir:  %s", *inputDir)
	log.Printf("Output dir: %s", *outputDir)
	log.Printf("Workers:    %d", *workers)
	log.Printf("Filter by language: %v", filterLang)
	if filterLang {
		log.Printf("Language detector URL: %s", *langDetectorURL)
	}

	// Собираем файлы
	files, err := filepath.Glob(filepath.Join(*inputDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d files", len(files))

	if len(files) == 0 {
		log.Fatal("No JSONL files found")
	}

	// Создаем клиент детектора языка (если нужен)
	var langClient *LanguageDetectorClient
	if filterLang {
		langClient = NewLanguageDetectorClient(*langDetectorURL)
		// Проверяем доступность сервиса
		log.Printf("Checking language detector at %s...", *langDetectorURL)
		resp, err := http.Get(*langDetectorURL + "/health")
		if err != nil {
			log.Fatalf("Language detector not available: %v", err)
		}
		resp.Body.Close()
		log.Printf("Language detector OK")
	}

	stats := &CleanStats{}
	progress := &Progress{
		total:     int64(len(files)),
		startTime: time.Now(),
	}

	// Канал задач
	taskChan := make(chan struct {
		input  string
		output string
	}, len(files))

	for _, f := range files {
		base := filepath.Base(f)
		taskChan <- struct {
			input  string
			output string
		}{f, filepath.Join(*outputDir, base)}
	}
	close(taskChan)

	// Прогресс
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			processed := atomic.LoadInt64(&progress.processed)
			if processed >= progress.total {
				return
			}
			elapsed := time.Since(progress.startTime)
			speed := float64(processed) / elapsed.Seconds()
			percent := float64(processed) / float64(progress.total) * 100
			kept := atomic.LoadInt64(&stats.Kept)
			removed := atomic.LoadInt64(&stats.Removed)
			log.Printf("[PROGRESS] %d/%d files (%.1f%%), speed: %.1f files/sec, kept: %d, removed: %d, elapsed: %v",
				processed, progress.total, percent, speed, kept, removed, elapsed.Round(time.Second))
		}
	}()

	// Воркеры
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskChan {
				if err := processFile(task.input, task.output, stats, langClient, filterLang); err != nil {
					log.Printf("ERROR processing %s: %v", task.input, err)
				}
				atomic.AddInt64(&progress.processed, 1)
			}
		}()
	}

	wg.Wait()
	totalTime := time.Since(progress.startTime)

	// Сохраняем статистику
	summary := map[string]interface{}{
		"total_files":             len(files),
		"processed":               stats.Processed,
		"total_kept":              stats.Kept,
		"total_removed":           stats.Removed,
		"empty_removed":           stats.EmptyRemoved,
		"adjacent_dupes_removed":  stats.AdjacentDupesRemoved,
		"index_removed":           stats.IndexRemoved,
		"garbage_pattern_removed": stats.GarbagePatternRemoved,
		"high_digit_removed":      stats.HighDigitRemoved,
		"ocr_garbage_removed":     stats.OCRGarbageRemoved,
		"language_removed":        stats.LanguageRemoved,
		"removed_percent":         float64(stats.Removed) / float64(stats.Removed+stats.Kept) * 100,
		"cleaning_time_seconds":   totalTime.Seconds(),
		"filter_by_language":      filterLang,
	}

	summaryPath := filepath.Join(*outputDir, "cleaning_summary.json")
	summaryData, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(summaryPath, summaryData, 0644)

	// Вывод в консоль
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("JSONL CLEANING COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total files:      %d\n", len(files))
	fmt.Printf("Total kept:       %d\n", stats.Kept)
	fmt.Printf("Total removed:    %d (%.2f%%)\n", stats.Removed, float64(stats.Removed)/float64(stats.Removed+stats.Kept)*100)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Empty:            %d\n", stats.EmptyRemoved)
	fmt.Printf("Adjacent dupes:   %d\n", stats.AdjacentDupesRemoved)
	fmt.Printf("Index entries:    %d\n", stats.IndexRemoved)
	fmt.Printf("Garbage patterns: %d\n", stats.GarbagePatternRemoved)
	fmt.Printf("High digit:       %d\n", stats.HighDigitRemoved)
	fmt.Printf("OCR garbage:      %d\n", stats.OCRGarbageRemoved)
	if filterLang {
		fmt.Printf("Language filter:  %d\n", stats.LanguageRemoved)
	}
	fmt.Printf("Time:             %v\n", totalTime.Round(time.Second))
	fmt.Println(strings.Repeat("=", 60))
}
