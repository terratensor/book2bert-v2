package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// ТИПЫ ДАННЫХ
// ============================================================================

// BookInfo метаданные книги из books_meta.jsonl
type BookInfo struct {
	BookID     string `json:"book_id"`
	Title      string `json:"title"`
	Author     string `json:"author"`
	Genre      string `json:"genre"`
	SourceFile string `json:"source_file"`
}

// BookStats статистика книги
type BookStats struct {
	BookID     string
	TotalLines int
	Language   string // "ru", "en", "mixed"
}

// ============================================================================
// ОПРЕДЕЛЕНИЕ ЯЗЫКА КНИГИ (УПРОЩЁННО)
// ============================================================================

// detectBookLanguage определяет преобладающий язык книги по сэмплу строк
func detectBookLanguage(jsonlPath string, sampleSize int) string {
	file, err := os.Open(jsonlPath)
	if err != nil {
		return "ru" // по умолчанию
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	cyrillic := 0
	latin := 0
	total := 0

	for scanner.Scan() && total < sampleSize {
		var sent struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &sent); err != nil {
			continue
		}

		for _, r := range sent.Text {
			if (r >= 'А' && r <= 'я') || r == 'Ё' || r == 'ё' {
				cyrillic++
			} else if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				latin++
			}
		}
		total++
	}

	if total == 0 {
		return "ru"
	}

	cyrillicRatio := float64(cyrillic) / float64(cyrillic+latin+1)

	if cyrillicRatio > 0.85 {
		return "ru"
	} else if cyrillicRatio < 0.15 {
		return "en"
	} else {
		return "mixed"
	}
}

// ============================================================================
// СБОР СТАТИСТИКИ ПО КНИГАМ
// ============================================================================

// countLines считает количество строк в JSONL файле
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

// analyzeBooks собирает статистику по всем книгам
func analyzeBooks(cleanedDir string, workers int) []BookStats {
	files, err := filepath.Glob(filepath.Join(cleanedDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}

	log.Printf("Analyzing %d books...", len(files))

	var stats []BookStats
	var mu sync.Mutex
	var processed int64

	tasks := make(chan string, len(files))
	for _, f := range files {
		tasks <- f
	}
	close(tasks)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range tasks {
				bookID := strings.TrimSuffix(filepath.Base(f), ".jsonl")

				bs := BookStats{
					BookID:     bookID,
					TotalLines: countLines(f),
				}

				// Определяем язык книги по сэмплу 100 строк
				if bs.TotalLines > 0 {
					bs.Language = detectBookLanguage(f, 100)
				} else {
					bs.Language = "ru"
				}

				mu.Lock()
				stats = append(stats, bs)
				mu.Unlock()

				proc := atomic.AddInt64(&processed, 1)
				if proc%1000 == 0 {
					log.Printf("  Analyzed %d/%d books", proc, len(files))
				}
			}
		}()
	}

	wg.Wait()
	log.Printf("Analysis complete: %d books", len(stats))

	return stats
}

// ============================================================================
// СТРАТИФИЦИРОВАННАЯ ВЫБОРКА
// ============================================================================

// sampleFromBook выбирает count случайных строк из книги, равномерно распределённых
// В функции sampleFromBook:
func sampleFromBook(jsonlPath string, count int, minLength int) []string {
	if count <= 0 {
		return nil
	}

	file, err := os.Open(jsonlPath)
	if err != nil {
		return nil
	}
	defer file.Close()

	totalLines := countLines(jsonlPath)
	if totalLines == 0 {
		return nil
	}

	// Запрашиваем с запасом: в 3 раза больше, чтобы точно хватило после фильтрации
	requestCount := count * 3
	if requestCount > totalLines {
		requestCount = totalLines
	}

	step := totalLines / requestCount
	if step < 1 {
		step = 1
	}

	var sentences []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	lineNum := 0
	// Продолжаем, пока не наберём count годных предложений
	for scanner.Scan() && len(sentences) < count {
		// Берём каждую step-ю строку
		if lineNum%step == 0 || len(sentences) == 0 {
			var sent struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(scanner.Bytes(), &sent); err == nil {
				text := strings.TrimSpace(sent.Text)

				// ИСПРАВЛЕНИЕ 1: Замена \n на пробел
				text = strings.ReplaceAll(text, "\n", " ")
				text = strings.ReplaceAll(text, "\r", " ")
				text = strings.Join(strings.Fields(text), " ")

				if len([]rune(text)) >= minLength {
					sentences = append(sentences, text)
				}
			}
		}
		lineNum++
	}

	// Если не хватило (книга маленькая) — возвращаем сколько есть
	return sentences
}

// buildStratifiedSample собирает стратифицированную выборку
func buildStratifiedSample(
	books []BookStats,
	cleanedDir string,
	totalSampleSize int,
	ruRatio, enRatio, mixedRatio float64,
	minLength int,
) []string {
	// Группируем книги по языкам
	var ruBooks, enBooks, mixedBooks []BookStats
	var ruLines, enLines, mixedLines int64

	for _, b := range books {
		switch b.Language {
		case "ru":
			ruBooks = append(ruBooks, b)
			ruLines += int64(b.TotalLines)
		case "en":
			enBooks = append(enBooks, b)
			enLines += int64(b.TotalLines)
		case "mixed":
			mixedBooks = append(mixedBooks, b)
			mixedLines += int64(b.TotalLines)
		}
	}

	log.Printf("Books by language:")
	log.Printf("  RU:    %d books, %d lines", len(ruBooks), ruLines)
	log.Printf("  EN:    %d books, %d lines", len(enBooks), enLines)
	log.Printf("  Mixed: %d books, %d lines", len(mixedBooks), mixedLines)

	// Вычисляем квоты
	ruQuota := int(float64(totalSampleSize) * ruRatio)
	enQuota := int(float64(totalSampleSize) * enRatio)
	mixedQuota := totalSampleSize - ruQuota - enQuota

	// Вычисляем предложений на книгу
	ruPerBook := ruQuota / len(ruBooks)
	enPerBook := enQuota / len(enBooks)
	mixedPerBook := mixedQuota / len(mixedBooks)

	log.Printf("Sampling quotas:")
	log.Printf("  RU:    %d sentences / %d books = %d per book", ruQuota, len(ruBooks), ruPerBook)
	log.Printf("  EN:    %d sentences / %d books = %d per book", enQuota, len(enBooks), enPerBook)
	log.Printf("  Mixed: %d sentences / %d books = %d per book", mixedQuota, len(mixedBooks), mixedPerBook)

	// Собираем сэмпл
	var allSentences []string
	var mu sync.Mutex

	// Создаём ОБЩИЙ канал задач для ВСЕХ книг
	type bookTask struct {
		book    BookStats
		perBook int
		label   string
	}

	tasks := make(chan bookTask, len(books))

	// Заполняем канал ВСЕМИ книгами
	for _, b := range ruBooks {
		tasks <- bookTask{b, ruPerBook, "RU"}
	}
	for _, b := range enBooks {
		tasks <- bookTask{b, enPerBook, "EN"}
	}
	for _, b := range mixedBooks {
		tasks <- bookTask{b, mixedPerBook, "Mixed"}
	}
	close(tasks)

	totalTasks := len(ruBooks) + len(enBooks) + len(mixedBooks)
	var processed int64

	log.Printf("Processing %d books with 32 workers...", totalTasks)
	startTime := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range tasks {
				jsonlPath := filepath.Join(cleanedDir, task.book.BookID+".jsonl")
				sentences := sampleFromBook(jsonlPath, task.perBook, minLength)

				mu.Lock()
				allSentences = append(allSentences, sentences...)
				current := len(allSentences)
				mu.Unlock()

				proc := atomic.AddInt64(&processed, 1)
				if proc%1000 == 0 {
					log.Printf("  [%s] %d/%d books, %d valid sentences collected",
						task.label, proc, totalTasks, current)
				}
			}
		}()
	}

	wg.Wait()

	elapsed := time.Since(startTime)
	log.Printf("Collected %d sentences from %d books in %v",
		len(allSentences), totalTasks, elapsed.Round(time.Second))

	return allSentences
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	var (
		cleanedDir = flag.String("cleaned", "data/cleaned", "директория с JSONL")
		indexFile  = flag.String("index", "", "файл индекса book_index.jsonl")
		outputFile = flag.String("output", "data/tokenizer/sample_20M.txt", "выходной файл")
		sampleSize = flag.Int("sample", 20000000, "размер сэмпла")
		ruRatio    = flag.Float64("ru", 0.87, "доля русских предложений")
		enRatio    = flag.Float64("en", 0.08, "доля английских предложений")
		mixedRatio = flag.Float64("mixed", 0.05, "доля mixed предложений")
		workers    = flag.Int("workers", 32, "количество воркеров")
		minLength  = flag.Int("min-length", 30, "минимальная длина предложения в символах")
	)
	flag.Parse()

	log.Printf("=== Stratified Corpus Sampler ===")
	log.Printf("Cleaned dir: %s", *cleanedDir)
	log.Printf("Sample size: %d", *sampleSize)
	log.Printf("Ratios: RU=%.0f%%, EN=%.0f%%, Mixed=%.0f%%", *ruRatio*100, *enRatio*100, *mixedRatio*100)
	log.Printf("Workers: %d", *workers)

	// Создаём выходную директорию
	if err := os.MkdirAll(filepath.Dir(*outputFile), 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	var books []BookStats

	if *indexFile != "" {
		// Загружаем индекс
		log.Printf("Loading index from %s...", *indexFile)
		books = loadIndex(*indexFile)
		log.Printf("Loaded %d books from index", len(books))
	} else {
		// Анализируем книги (старый способ)
		books = analyzeBooks(*cleanedDir, *workers)
	}

	if len(books) == 0 {
		log.Fatal("No books found")
	}

	// Собираем стратифицированную выборку
	sentences := buildStratifiedSample(
		books, *cleanedDir, *sampleSize,
		*ruRatio, *enRatio, *mixedRatio,
		*minLength, // ← добавить параметр
	)

	// Глобально перемешиваем
	log.Printf("Shuffling %d sentences...", len(sentences))
	rand.Shuffle(len(sentences), func(i, j int) {
		sentences[i], sentences[j] = sentences[j], sentences[i]
	})

	// Сохраняем
	log.Printf("Saving to %s...", *outputFile)
	outFile, err := os.Create(*outputFile)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1024*1024)
	for _, sent := range sentences {
		writer.WriteString(sent + "\n")
	}
	writer.Flush()

	// Статистика
	log.Printf("=== Done ===")
	log.Printf("Total sentences: %d", len(sentences))

	// Проверяем распределение языков в сэмпле
	ruCount, enCount, mixedCount := 0, 0, 0
	for _, sent := range sentences {
		cyrillic := 0
		latin := 0
		for _, r := range sent {
			if (r >= 'А' && r <= 'я') || r == 'Ё' || r == 'ё' {
				cyrillic++
			} else if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				latin++
			}
		}
		if cyrillic > latin*5 {
			ruCount++
		} else if latin > cyrillic*5 {
			enCount++
		} else {
			mixedCount++
		}
	}

	total := len(sentences)
	log.Printf("Sample language distribution:")
	log.Printf("  RU:    %d (%.1f%%)", ruCount, float64(ruCount)/float64(total)*100)
	log.Printf("  EN:    %d (%.1f%%)", enCount, float64(enCount)/float64(total)*100)
	log.Printf("  Mixed: %d (%.1f%%)", mixedCount, float64(mixedCount)/float64(total)*100)

	log.Printf("Output saved to: %s", *outputFile)
}

// loadIndex загружает индекс книг из JSONL файла
func loadIndex(indexPath string) []BookStats {
	file, err := os.Open(indexPath)
	if err != nil {
		log.Fatalf("open index: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	var books []BookStats
	for scanner.Scan() {
		var info struct {
			BookID        string  `json:"book_id"`
			TotalLines    int     `json:"total_lines"`
			Language      string  `json:"language"`
			CyrillicRatio float64 `json:"cyrillic_ratio"`
			LatinRatio    float64 `json:"latin_ratio"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &info); err != nil {
			continue
		}

		books = append(books, BookStats{
			BookID:     info.BookID,
			TotalLines: info.TotalLines,
			Language:   info.Language,
		})
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("read index: %v", err)
	}

	return books
}
