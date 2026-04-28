package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// ТИПЫ ДАННЫХ
// ============================================================================

// BookLanguageInfo информация о языке книги
type BookLanguageInfo struct {
	BookID        string  `json:"book_id"`
	TotalLines    int     `json:"total_lines"`
	Language      string  `json:"language"` // "ru", "en", "mixed"
	CyrillicRatio float64 `json:"cyrillic_ratio"`
	LatinRatio    float64 `json:"latin_ratio"`
}

// ============================================================================
// АНАЛИЗ КНИГИ
// ============================================================================

// countLines быстро считает количество строк в файле
func countLines(jsonlPath string) int {
	file, err := os.Open(jsonlPath)
	if err != nil {
		return 0
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	count := 0
	for scanner.Scan() {
		count++
	}
	return count
}

// analyzeBookLanguage определяет язык книги по сэмплу предложений
func analyzeBookLanguage(jsonlPath string, sampleSize int) (language string, cyrillicRatio, latinRatio float64) {
	file, err := os.Open(jsonlPath)
	if err != nil {
		return "ru", 0, 0
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	var totalCyrillic, totalLatin int64
	totalLines := 0

	if sampleSize > 0 {
		// Режим сэмпла
		totalLinesAll := countLines(jsonlPath)
		if totalLinesAll == 0 {
			return "ru", 0, 0
		}
		step := totalLinesAll / sampleSize
		if step < 1 {
			step = 1
		}
		lineNum := 0
		for scanner.Scan() && totalLines < sampleSize {
			if lineNum%step == 0 || totalLines == 0 {
				var sent struct {
					Text string `json:"text"`
				}
				if err := json.Unmarshal(scanner.Bytes(), &sent); err == nil {
					cyrillic, latin := countLetters(sent.Text)
					totalCyrillic += int64(cyrillic)
					totalLatin += int64(latin)
					totalLines++
				}
			}
			lineNum++
		}
	} else {
		// Режим полного анализа
		for scanner.Scan() {
			var sent struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(scanner.Bytes(), &sent); err == nil {
				cyrillic, latin := countLetters(sent.Text)
				totalCyrillic += int64(cyrillic)
				totalLatin += int64(latin)
				totalLines++
			}
		}
	}

	if totalLines == 0 {
		return "ru", 0, 0
	}

	cyrillicRatio = float64(totalCyrillic) / float64(totalCyrillic+totalLatin+1)
	latinRatio = float64(totalLatin) / float64(totalCyrillic+totalLatin+1)

	switch {
	case cyrillicRatio > 0.85:
		language = "ru"
	case latinRatio > 0.85:
		language = "en"
	default:
		language = "mixed"
	}

	return
}

// countLetters подсчитывает кириллические и латинские буквы
func countLetters(text string) (cyrillic, latin int) {
	for _, r := range text {
		if (r >= 'А' && r <= 'я') || r == 'Ё' || r == 'ё' {
			cyrillic++
		} else if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
			latin++
		}
	}
	return
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	var (
		cleanedDir = flag.String("cleaned", "data/cleaned", "директория с JSONL файлами")
		outputFile = flag.String("output", "data/index/book_index.jsonl", "выходной файл индекса")
		workers    = flag.Int("workers", 32, "количество воркеров")
		sampleSize = flag.Int("sample", 100, "размер сэмпла для анализа книги")
	)
	flag.Parse()

	log.Printf("=== Book Language Analyzer ===")
	log.Printf("Cleaned dir: %s", *cleanedDir)
	log.Printf("Output: %s", *outputFile)
	log.Printf("Workers: %d", *workers)
	log.Printf("Sample size: %d per book", *sampleSize)

	// Создаём выходную директорию
	if err := os.MkdirAll(filepath.Dir(*outputFile), 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Находим все JSONL файлы
	files, err := filepath.Glob(filepath.Join(*cleanedDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d books", len(files))

	if len(files) == 0 {
		log.Fatal("No JSONL files found")
	}

	// Сортируем для детерминированности
	sort.Strings(files)

	// Каналы для параллельной обработки
	tasks := make(chan string, len(files))
	results := make(chan BookLanguageInfo, len(files))

	for _, f := range files {
		tasks <- f
	}
	close(tasks)

	// Воркеры
	var wg sync.WaitGroup
	var processed int64
	startTime := time.Now()

	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for f := range tasks {
				bookID := strings.TrimSuffix(filepath.Base(f), ".jsonl")

				lang, cyrRatio, latRatio := analyzeBookLanguage(f, *sampleSize)
				totalLines := countLines(f)

				info := BookLanguageInfo{
					BookID:        bookID,
					TotalLines:    totalLines,
					Language:      lang,
					CyrillicRatio: round(cyrRatio, 4),
					LatinRatio:    round(latRatio, 4),
				}

				results <- info

				proc := atomic.AddInt64(&processed, 1)
				if proc%1000 == 0 {
					elapsed := time.Since(startTime)
					log.Printf("  Processed %d/%d books (%.1f%%), elapsed: %v",
						proc, len(files), float64(proc)/float64(len(files))*100,
						elapsed.Round(time.Second))
				}
			}
		}(i)
	}

	// Ждём завершения и закрываем канал результатов
	go func() {
		wg.Wait()
		close(results)
	}()

	// Собираем результаты
	var allResults []BookLanguageInfo
	for r := range results {
		allResults = append(allResults, r)
	}

	// Сортируем по BookID для детерминированности
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].BookID < allResults[j].BookID
	})

	// Сохраняем индекс
	outFile, err := os.Create(*outputFile)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1024*1024)
	encoder := json.NewEncoder(writer)

	for _, info := range allResults {
		if err := encoder.Encode(info); err != nil {
			log.Printf("ERROR encoding: %v", err)
		}
	}
	writer.Flush()

	// Статистика
	var ruCount, enCount, mixedCount int
	var ruLines, enLines, mixedLines int64

	for _, info := range allResults {
		switch info.Language {
		case "ru":
			ruCount++
			ruLines += int64(info.TotalLines)
		case "en":
			enCount++
			enLines += int64(info.TotalLines)
		case "mixed":
			mixedCount++
			mixedLines += int64(info.TotalLines)
		}
	}

	elapsed := time.Since(startTime)
	totalBooks := len(allResults)
	totalLines := ruLines + enLines + mixedLines

	log.Printf("=== Done ===")
	log.Printf("Total books: %d", totalBooks)
	log.Printf("Total lines: %d", totalLines)
	log.Printf("Time: %v", elapsed.Round(time.Second))
	log.Println()
	log.Printf("Language distribution:")
	log.Printf("  RU:    %d books (%.1f%%), %d lines (%.1f%%)",
		ruCount, float64(ruCount)/float64(totalBooks)*100,
		ruLines, float64(ruLines)/float64(totalLines)*100)
	log.Printf("  EN:    %d books (%.1f%%), %d lines (%.1f%%)",
		enCount, float64(enCount)/float64(totalBooks)*100,
		enLines, float64(enLines)/float64(totalLines)*100)
	log.Printf("  Mixed: %d books (%.1f%%), %d lines (%.1f%%)",
		mixedCount, float64(mixedCount)/float64(totalBooks)*100,
		mixedLines, float64(mixedLines)/float64(totalLines)*100)
	log.Printf("Index saved to: %s", *outputFile)
}

// round округляет float64 до указанного числа знаков
func round(val float64, precision int) float64 {
	format := fmt.Sprintf("%%.%df", precision)
	str := fmt.Sprintf(format, val)
	var result float64
	fmt.Sscanf(str, "%f", &result)
	return result
}
