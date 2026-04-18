package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// Stats для очистки
type CleanStats struct {
	Processed            int64
	Kept                 int64
	Removed              int64
	AdjacentDupesRemoved int64
	IndexRemoved         int64
	URLRemoved           int64
	GarbageShortRemoved  int64
	HighDigitRemoved     int64
}

var (
	urlRegex   = regexp.MustCompile(`https?://[^\s]+`)
	isbnRegex  = regexp.MustCompile(`\bISBN\s*\d{3}-\d{1,5}-\d{1,7}-\d{1,7}-\d{1,7}\b`)
	udkRegex   = regexp.MustCompile(`\bУДК\s*\d+(?:\.\d+)+\b`)
	bbkRegex   = regexp.MustCompile(`\bББК\s*\d+(?:\.\d+)+\b`)
	emailRegex = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
)

// isIndexEntry проверяет, является ли текст предметным указателем
func isIndexEntry(text string) bool {
	lines := strings.Split(text, "\n")
	if len(lines) < 3 {
		return false
	}

	// Паттерн: текст заканчивается на цифры (возможно с запятой)
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

	// Если >70% строк соответствуют паттерну — это индекс
	return float64(matches)/float64(nonEmpty) > 0.7
}

// isGarbageShort проверяет короткий мусор
func isGarbageShort(text string) bool {
	text = strings.TrimSpace(text)
	runes := []rune(text)

	// Слишком короткое
	if len(runes) < 3 {
		letters := 0
		for _, r := range runes {
			if unicode.IsLetter(r) {
				letters++
			}
		}
		// Если нет букв или только одна буква — мусор
		if letters == 0 {
			return true
		}
		if letters == 1 && len(runes) <= 2 {
			return true
		}
	}

	// Только цифры и пунктуация
	letters := 0
	for _, r := range runes {
		if unicode.IsLetter(r) {
			letters++
		}
	}
	if letters == 0 && len(runes) < 20 {
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

// hasGarbagePatterns проверяет наличие мусорных паттернов
func hasGarbagePatterns(text string) bool {
	return urlRegex.MatchString(text) ||
		isbnRegex.MatchString(text) ||
		udkRegex.MatchString(text) ||
		bbkRegex.MatchString(text) ||
		emailRegex.MatchString(text)
}

// normalizeListMarkers нормализует маркеры списков
func normalizeListMarkers(text string) string {
	// Заменяем маркеры вида "1. ", "1) " на "• "
	listMarkerRegex := regexp.MustCompile(`^[\d]+[\.\)]\s+`)
	if listMarkerRegex.MatchString(text) {
		text = listMarkerRegex.ReplaceAllString(text, "• ")
	}

	// Заменяем дефисы в начале строки на •
	dashMarkerRegex := regexp.MustCompile(`^[-–—]\s+`)
	if dashMarkerRegex.MatchString(text) {
		text = dashMarkerRegex.ReplaceAllString(text, "• ")
	}

	return text
}

// normalizePunctuation нормализует кавычки и тире
func normalizePunctuation(text string) string {
	// Кавычки
	text = strings.ReplaceAll(text, "«", "\"")
	text = strings.ReplaceAll(text, "»", "\"")
	text = strings.ReplaceAll(text, "„", "\"")
	text = strings.ReplaceAll(text, "“", "\"")
	text = strings.ReplaceAll(text, "”", "\"")
	text = strings.ReplaceAll(text, "'", "'")
	text = strings.ReplaceAll(text, "'", "'")

	// Тире и дефисы
	text = strings.ReplaceAll(text, "—", "-")
	text = strings.ReplaceAll(text, "–", "-")
	text = strings.ReplaceAll(text, "−", "-")

	// Многоточие
	text = regexp.MustCompile(`\.{3,}`).ReplaceAllString(text, "...")
	text = strings.ReplaceAll(text, "…", "...")

	return text
}

// cleanSentence очищает одно предложение
func cleanSentence(text string) (string, bool, string) {
	text = strings.TrimSpace(text)

	// 1. Пустые
	if len(text) == 0 {
		return "", false, "empty"
	}

	// 2. Мусорные паттерны (URL, ISBN, etc)
	if hasGarbagePatterns(text) {
		return "", false, "garbage_pattern"
	}

	// 3. Предметные указатели
	if isIndexEntry(text) {
		return "", false, "index"
	}

	// 4. Короткий мусор
	if isGarbageShort(text) {
		return "", false, "short_garbage"
	}

	// 5. Много цифр + мало букв
	dr := digitRatio(text)
	lr := letterRatio(text)
	if dr > 0.5 && lr < 0.3 {
		return "", false, "high_digit"
	}

	// 6. Нормализация пунктуации
	text = normalizePunctuation(text)

	// 7. Нормализация маркеров списков
	text = normalizeListMarkers(text)

	// 8. Убираем множественные пробелы
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	return text, true, ""
}

// processFile обрабатывает один JSONL файл
func processFile(inputPath, outputPath string, stats *CleanStats) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return err
	}
	defer inputFile.Close()

	// Создаем выходную директорию если нужно
	os.MkdirAll(filepath.Dir(outputPath), 0755)

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

		cleaned, keep, reason := cleanSentence(text)
		if !keep {
			atomic.AddInt64(&stats.Removed, 1)
			switch reason {
			case "index":
				atomic.AddInt64(&stats.IndexRemoved, 1)
			case "garbage_pattern":
				atomic.AddInt64(&stats.URLRemoved, 1)
			case "short_garbage":
				atomic.AddInt64(&stats.GarbageShortRemoved, 1)
			case "high_digit":
				atomic.AddInt64(&stats.HighDigitRemoved, 1)
			}
			continue
		}

		// Проверка на точный дубль подряд
		if cleaned == prevText {
			atomic.AddInt64(&stats.AdjacentDupesRemoved, 1)
			continue
		}

		sent["text"] = cleaned
		sentences = append(sentences, sent)
		prevText = cleaned
		localKept++
	}

	// Перезаписываем position
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

// Progress отслеживает прогресс
type Progress struct {
	processed int64
	total     int64
	startTime time.Time
}

func main() {
	var (
		inputDir  = flag.String("input", "", "входная директория с JSONL файлами")
		outputDir = flag.String("output", "", "выходная директория")
		workers   = flag.Int("workers", 32, "количество воркеров")
	)
	flag.Parse()

	if *inputDir == "" || *outputDir == "" {
		log.Fatal("--input and --output are required")
	}

	// Собираем файлы
	files, err := filepath.Glob(filepath.Join(*inputDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d files", len(files))

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
			log.Printf("[PROGRESS] %d/%d files (%.1f%%), speed: %.1f files/sec, elapsed: %v",
				processed, progress.total, percent, speed, elapsed.Round(time.Second))
		}
	}()

	// Воркеры
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskChan {
				if err := processFile(task.input, task.output, stats); err != nil {
					log.Printf("ERROR processing %s: %v", task.input, err)
				}
				atomic.AddInt64(&progress.processed, 1)
			}
		}()
	}

	wg.Wait()
	totalTime := time.Since(progress.startTime)

	// Сохраняем статистику очистки
	summary := map[string]interface{}{
		"total_files":            len(files),
		"processed":              stats.Processed,
		"total_kept":             stats.Kept,
		"total_removed":          stats.Removed,
		"adjacent_dupes_removed": stats.AdjacentDupesRemoved,
		"index_removed":          stats.IndexRemoved,
		"url_removed":            stats.URLRemoved,
		"garbage_short_removed":  stats.GarbageShortRemoved,
		"high_digit_removed":     stats.HighDigitRemoved,
		"removed_percent":        float64(stats.Removed) / float64(stats.Removed+stats.Kept) * 100,
		"cleaning_time_seconds":  totalTime.Seconds(),
	}

	summaryPath := filepath.Join(*outputDir, "cleaning_summary.json")
	summaryData, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(summaryPath, summaryData, 0644)

	// Вывод
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("JSONL CLEANING COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total files:      %d\n", len(files))
	fmt.Printf("Total kept:       %d\n", stats.Kept)
	fmt.Printf("Total removed:    %d (%.2f%%)\n", stats.Removed, float64(stats.Removed)/float64(stats.Removed+stats.Kept)*100)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Adjacent dupes:   %d\n", stats.AdjacentDupesRemoved)
	fmt.Printf("Index entries:    %d\n", stats.IndexRemoved)
	fmt.Printf("URL/ISBN/etc:     %d\n", stats.URLRemoved)
	fmt.Printf("Short garbage:    %d\n", stats.GarbageShortRemoved)
	fmt.Printf("High digit:       %d\n", stats.HighDigitRemoved)
	fmt.Printf("Time:             %v\n", totalTime.Round(time.Second))
	fmt.Println(strings.Repeat("=", 60))
}
