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

// hasGarbagePatterns проверяет наличие мусорных паттернов
func hasGarbagePatterns(text string) bool {
	return urlRegex.MatchString(text) ||
		isbnRegex.MatchString(text) ||
		udkRegex.MatchString(text) ||
		bbkRegex.MatchString(text) ||
		emailRegex.MatchString(text)
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

// cleanSentence очищает одно предложение
// Возвращает: очищенный текст, флаг "оставить", причина удаления
func cleanSentence(text string) (string, bool, string) {
	text = strings.TrimSpace(text)

	// 1. Пустые
	if len(text) == 0 {
		return "", false, "empty"
	}

	// 2. Мусорные паттерны (URL, ISBN, email)
	if hasGarbagePatterns(text) {
		return "", false, "garbage_pattern"
	}

	// 3. Предметные указатели
	if isIndexEntry(text) {
		return "", false, "index"
	}

	// 4. Только цифры и пунктуация (нет букв)
	lr := letterRatio(text)
	if lr < 0.1 {
		dr := digitRatio(text)
		if dr > 0.5 {
			return "", false, "high_digit_no_letters"
		}
	}

	// 5. Убираем множественные пробелы (но сохраняем \n внутри!)
	// Не трогаем \n, так как они могут быть частью структуры (списки)
	text = regexp.MustCompile(`[ \t]+`).ReplaceAllString(text, " ")

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
			case "empty":
				atomic.AddInt64(&stats.EmptyRemoved, 1)
			case "garbage_pattern":
				atomic.AddInt64(&stats.GarbagePatternRemoved, 1)
			case "index":
				atomic.AddInt64(&stats.IndexRemoved, 1)
			case "high_digit_no_letters":
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

	if len(files) == 0 {
		log.Fatal("No JSONL files found")
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
		"total_files":             len(files),
		"processed":               stats.Processed,
		"total_kept":              stats.Kept,
		"total_removed":           stats.Removed,
		"empty_removed":           stats.EmptyRemoved,
		"adjacent_dupes_removed":  stats.AdjacentDupesRemoved,
		"index_removed":           stats.IndexRemoved,
		"garbage_pattern_removed": stats.GarbagePatternRemoved,
		"high_digit_removed":      stats.HighDigitRemoved,
		"removed_percent":         float64(stats.Removed) / float64(stats.Removed+stats.Kept) * 100,
		"cleaning_time_seconds":   totalTime.Seconds(),
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
	fmt.Printf("Time:             %v\n", totalTime.Round(time.Second))
	fmt.Println(strings.Repeat("=", 60))
}
