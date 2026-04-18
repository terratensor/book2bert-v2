package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// Stats хранит агрегированную статистику
type Stats struct {
	// Базовые метрики
	TotalSentences int64
	TotalChars     int64
	TotalFiles     int64
	EmptySentences int64

	// Гистограмма длин (1001 бакетов: 0-10,10-20,...,9990-10000, >10000)
	lengthBuckets [1001]int64
	totalLengths  int64 // общее количество предложений для перцентилей

	// Языковой состав
	RussianOnly        int64
	EnglishOnly        int64
	MixedCyrillicLatin int64
	Other              int64

	// Качество
	TooShort20      int64
	TooShort50      int64
	TooLong1000     int64
	TooLong2000     int64
	HighDigit10     int64
	HighDigit30     int64
	HighDigit50     int64
	HighPunct30     int64
	HighPunct50     int64
	ListMarker      int64
	HasISBN         int64
	HasUDK          int64
	HasBBK          int64
	HasURL          int64
	HasEmail        int64
	HighUppercase50 int64
	HighUppercase80 int64

	// Новые метрики
	AdjacentDuplicates      int64 // точные дубли подряд
	RunningHeaderCandidates int64 // кандидаты в колонтитулы (будет заполнено после внешней сортировки)
	BrokenSentences         int64 // разорванные предложения
	IndexEntries            int64 // предметные указатели
	HyphenatedWordsLeft     int64 // оставшиеся переносы слов
	GreekLettersCount       int64 // количество греческих символов
	MathSymbolsCount        int64 // количество математических символов
}

// FileStats хранит статистику по одному файлу
type FileStats struct {
	Path         string
	Sentences    int
	AvgLength    float64
	RussianRatio float64
	EnglishRatio float64
	MixedRatio   float64
	DigitRatio30 float64
	ListRatio    float64
	GarbageRatio float64
	Examples     []string // примеры проблемных предложений
}

// Progress отслеживает прогресс обработки
type Progress struct {
	processedFiles int64
	totalFiles     int64
	startTime      time.Time
}

// truncate обрезает строку до maxLen
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}

func isRussian(text string) bool {
	for _, r := range text {
		if r >= 0x0400 && r <= 0x04FF {
			return true
		}
	}
	return false
}

func isEnglish(text string) bool {
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return true
		}
	}
	return false
}

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

func punctRatio(text string) float64 {
	punct := 0
	runes := []rune(text)
	if len(runes) == 0 {
		return 0
	}
	for _, r := range runes {
		if unicode.IsPunct(r) {
			punct++
		}
	}
	return float64(punct) / float64(len(runes))
}

func uppercaseRatio(text string) float64 {
	upper := 0
	letters := 0
	for _, r := range text {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	if letters == 0 {
		return 0
	}
	return float64(upper) / float64(letters)
}

var listMarkerRegex = regexp.MustCompile(`^\s*[•\-*•\d]+[\.\)]\s+`)

func hasListMarker(text string) bool {
	return listMarkerRegex.MatchString(text)
}

var (
	isbnRegex  = regexp.MustCompile(`\bISBN\s*\d{3}-\d{1,5}-\d{1,7}-\d{1,7}-\d{1,7}\b`)
	udkRegex   = regexp.MustCompile(`\bУДК\s*\d+(?:\.\d+)+\b`)
	bbkRegex   = regexp.MustCompile(`\bББК\s*\d+(?:\.\d+)+\b`)
	urlRegex   = regexp.MustCompile(`https?://[^\s]+`)
	emailRegex = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
)

func hasGarbagePatterns(text string) (hasISBN, hasUDK, hasBBK, hasURL, hasEmail bool) {
	hasISBN = isbnRegex.MatchString(text)
	hasUDK = udkRegex.MatchString(text)
	hasBBK = bbkRegex.MatchString(text)
	hasURL = urlRegex.MatchString(text)
	hasEmail = emailRegex.MatchString(text)
	return
}

// isBrokenEnding проверяет, заканчивается ли предложение на предлог/союз
func isBrokenEnding(text string) bool {
	words := strings.Fields(text)
	if len(words) == 0 {
		return false
	}
	lastWord := strings.ToLower(words[len(words)-1])
	lastWord = strings.TrimRight(lastWord, ".,;:!?()[]{}\"'")

	brokenEndings := []string{
		"в", "на", "с", "по", "к", "у", "о", "об", "от", "до", "без", "для",
		"из", "за", "над", "под", "при", "про", "через", "между", "перед",
		"и", "а", "но", "или", "либо", "что", "чтобы", "если", "когда",
		"in", "on", "at", "by", "for", "with", "to", "from", "of", "and", "or", "but",
	}

	for _, ending := range brokenEndings {
		if lastWord == ending {
			return true
		}
	}
	return false
}

// startsWithLower проверяет, начинается ли строка с маленькой буквы
func startsWithLower(text string) bool {
	if len(text) == 0 {
		return false
	}
	firstRune := []rune(text)[0]
	return unicode.IsLower(firstRune)
}

// isIndexEntry проверяет, является ли текст предметным указателем
func isIndexEntry(text string) bool {
	lines := strings.Split(text, "\n")
	if len(lines) < 3 {
		return false
	}

	pattern := regexp.MustCompile(`.*?[,\s]+\d+[\s,]*$`)

	matches := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && pattern.MatchString(line) {
			matches++
		}
	}

	return float64(matches)/float64(len(lines)) > 0.7
}

// hasHyphenatedWord проверяет наличие оставшихся переносов вида "слово-\nслово"
var hyphenRegex = regexp.MustCompile(`\p{L}+-\s*\n\s*\p{L}+`)

func hasHyphenatedWord(text string) bool {
	return hyphenRegex.MatchString(text)
}

// countGreek считает количество греческих символов
func countGreek(text string) int {
	count := 0
	for _, r := range text {
		if (r >= 0x0370 && r <= 0x03FF) || // Greek and Coptic
			(r >= 0x1F00 && r <= 0x1FFF) { // Greek Extended
			count++
		}
	}
	return count
}

// countMathSymbols считает количество математических символов
func countMathSymbols(text string) int {
	count := 0
	for _, r := range text {
		if r >= 0x2200 && r <= 0x22FF { // Mathematical Operators
			count++
		}
	}
	return count
}

// addLength добавляет длину в гистограмму (потокобезопасно)
func (s *Stats) addLength(l int64) {
	idx := l / 10
	if idx > 1000 {
		idx = 1000
	}
	atomic.AddInt64(&s.lengthBuckets[idx], 1)
	atomic.AddInt64(&s.totalLengths, 1)
}

// percentile вычисляет p-й перцентиль (p от 0 до 100) по гистограмме
func (s *Stats) percentile(p float64) int64 {
	target := int64(float64(s.totalLengths) * p / 100)
	var cum int64
	for i, count := range s.lengthBuckets {
		cum += count
		if cum >= target {
			// Возвращаем середину бина
			return int64(i)*10 + 5
		}
	}
	return 10005 // если не нашли (например, для p=100)
}

// analyzeFile обрабатывает один JSONL файл
func analyzeFile(filePath string, stats *Stats, fileStatsChan chan<- FileStats, wg *sync.WaitGroup, sem chan struct{}, progress *Progress, phraseChan chan<- string) {
	defer wg.Done()
	sem <- struct{}{}
	defer func() { <-sem }()
	defer atomic.AddInt64(&progress.processedFiles, 1)

	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("Error opening %s: %v", filePath, err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	var localLengths []int64 // для локального вычисления AvgLength
	var localStats FileStats
	localStats.Path = filePath
	localStats.Examples = make([]string, 0, 10)

	var prevText string
	var prevPosition int

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var data map[string]interface{}
		if err := json.Unmarshal(line, &data); err != nil {
			continue
		}

		text, ok := data["text"].(string)
		if !ok || text == "" {
			atomic.AddInt64(&stats.EmptySentences, 1)
			continue
		}

		position := 0
		if pos, ok := data["position"].(float64); ok {
			position = int(pos)
		}

		atomic.AddInt64(&stats.TotalSentences, 1)
		localStats.Sentences++
		length := int64(len([]rune(text)))
		atomic.AddInt64(&stats.TotalChars, length)
		localLengths = append(localLengths, length)

		// Добавляем длину в гистограмму
		stats.addLength(length)

		// Языковой состав
		hasRu := isRussian(text)
		hasEn := isEnglish(text)

		if hasRu && !hasEn {
			atomic.AddInt64(&stats.RussianOnly, 1)
			localStats.RussianRatio++
		} else if !hasRu && hasEn {
			atomic.AddInt64(&stats.EnglishOnly, 1)
			localStats.EnglishRatio++
		} else if hasRu && hasEn {
			atomic.AddInt64(&stats.MixedCyrillicLatin, 1)
			localStats.MixedRatio++
		} else {
			atomic.AddInt64(&stats.Other, 1)
		}

		// Качество - длина
		if length < 20 {
			atomic.AddInt64(&stats.TooShort20, 1)
		}
		if length < 50 {
			atomic.AddInt64(&stats.TooShort50, 1)
		}
		if length > 1000 {
			atomic.AddInt64(&stats.TooLong1000, 1)
		}
		if length > 2000 {
			atomic.AddInt64(&stats.TooLong2000, 1)
		}

		// Цифры
		dRatio := digitRatio(text)
		if dRatio > 0.1 {
			atomic.AddInt64(&stats.HighDigit10, 1)
		}
		if dRatio > 0.3 {
			atomic.AddInt64(&stats.HighDigit30, 1)
			localStats.DigitRatio30++
		}
		if dRatio > 0.5 {
			atomic.AddInt64(&stats.HighDigit50, 1)
		}

		// Пунктуация
		pRatio := punctRatio(text)
		if pRatio > 0.3 {
			atomic.AddInt64(&stats.HighPunct30, 1)
		}
		if pRatio > 0.5 {
			atomic.AddInt64(&stats.HighPunct50, 1)
		}

		// Маркеры списков
		if hasListMarker(text) {
			atomic.AddInt64(&stats.ListMarker, 1)
			localStats.ListRatio++
		}

		// Мусорные паттерны
		hasISBN, hasUDK, hasBBK, hasURL, hasEmail := hasGarbagePatterns(text)
		if hasISBN {
			atomic.AddInt64(&stats.HasISBN, 1)
			localStats.GarbageRatio++
		}
		if hasUDK {
			atomic.AddInt64(&stats.HasUDK, 1)
			localStats.GarbageRatio++
		}
		if hasBBK {
			atomic.AddInt64(&stats.HasBBK, 1)
			localStats.GarbageRatio++
		}
		if hasURL {
			atomic.AddInt64(&stats.HasURL, 1)
			localStats.GarbageRatio++
		}
		if hasEmail {
			atomic.AddInt64(&stats.HasEmail, 1)
			localStats.GarbageRatio++
		}

		// Заглавные буквы
		uRatio := uppercaseRatio(text)
		if uRatio > 0.5 {
			atomic.AddInt64(&stats.HighUppercase50, 1)
		}
		if uRatio > 0.8 {
			atomic.AddInt64(&stats.HighUppercase80, 1)
		}

		// НОВЫЕ МЕТРИКИ

		// Точные дубли подряд
		if text == prevText && position == prevPosition+1 {
			atomic.AddInt64(&stats.AdjacentDuplicates, 1)
			if len(localStats.Examples) < 5 {
				localStats.Examples = append(localStats.Examples, "DUPE: "+truncate(text, 80))
			}
		}

		// Разорванные предложения
		if prevText != "" {
			if isBrokenEnding(prevText) && startsWithLower(text) {
				atomic.AddInt64(&stats.BrokenSentences, 1)
				if len(localStats.Examples) < 5 {
					localStats.Examples = append(localStats.Examples, "BROKEN: "+truncate(prevText+" "+text, 80))
				}
			}
		}

		// Предметные указатели
		if isIndexEntry(text) {
			atomic.AddInt64(&stats.IndexEntries, 1)
			if len(localStats.Examples) < 5 {
				localStats.Examples = append(localStats.Examples, "INDEX: "+truncate(strings.ReplaceAll(text, "\n", " / "), 80))
			}
		}

		// Оставшиеся переносы
		if hasHyphenatedWord(text) {
			atomic.AddInt64(&stats.HyphenatedWordsLeft, 1)
		}

		// Греческие символы
		if greek := countGreek(text); greek > 0 {
			atomic.AddInt64(&stats.GreekLettersCount, int64(greek))
		}

		// Математические символы
		if math := countMathSymbols(text); math > 0 {
			atomic.AddInt64(&stats.MathSymbolsCount, int64(math))
		}

		// Сбор фраз для анализа колонтитулов (отправляем в канал)
		if length > 10 && length < 100 {
			select {
			case phraseChan <- text:
			default:
				// если канал переполнен, пропускаем (маловероятно)
			}
		}

		prevText = text
		prevPosition = position
	}

	// Вычисляем среднюю длину для файла
	if localStats.Sentences > 0 {
		var sum int64
		for _, l := range localLengths {
			sum += l
		}
		localStats.AvgLength = float64(sum) / float64(localStats.Sentences)
		localStats.RussianRatio /= float64(localStats.Sentences)
		localStats.EnglishRatio /= float64(localStats.Sentences)
		localStats.MixedRatio /= float64(localStats.Sentences)
		localStats.DigitRatio30 /= float64(localStats.Sentences)
		localStats.ListRatio /= float64(localStats.Sentences)
		localStats.GarbageRatio /= float64(localStats.Sentences)
	}
	fileStatsChan <- localStats
}

func main() {
	var (
		sentencesDir = flag.String("dir", "", "директория с JSONL файлами")
		workers      = flag.Int("workers", 32, "количество воркеров")
		outputDir    = flag.String("output", "data/analysis", "выходная директория")
	)
	flag.Parse()

	if *sentencesDir == "" {
		log.Fatal("--dir is required")
	}

	files, err := filepath.Glob(filepath.Join(*sentencesDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d files", len(files))

	stats := &Stats{}

	progress := &Progress{
		totalFiles: int64(len(files)),
		startTime:  time.Now(),
	}

	// Канал для фраз (колонтитулы) с буфером
	phraseChan := make(chan string, 10000)

	// Горутина-писатель для сырых фраз
	var phraseWg sync.WaitGroup
	phraseWg.Add(1)
	go func() {
		defer phraseWg.Done()
		rawPath := filepath.Join(*outputDir, "phrases_raw.txt")
		f, err := os.Create(rawPath)
		if err != nil {
			log.Printf("ERROR creating phrases_raw.txt: %v", err)
			return
		}
		defer f.Close()
		w := bufio.NewWriter(f)
		defer w.Flush()
		for phrase := range phraseChan {
			w.WriteString(phrase)
			w.WriteByte('\n')
		}
	}()

	// Канал для пофайловой статистики (буфер 5000)
	fileStatsChan := make(chan FileStats, 5000)
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	// Прогресс
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			processed := atomic.LoadInt64(&progress.processedFiles)
			if processed >= progress.totalFiles {
				return
			}
			elapsed := time.Since(progress.startTime)
			speed := float64(processed) / elapsed.Seconds()
			percent := float64(processed) / float64(progress.totalFiles) * 100
			log.Printf("[PROGRESS] %d/%d files processed (%.1f%%), speed: %.1f files/sec, elapsed: %v",
				processed, progress.totalFiles, percent, speed, elapsed.Round(time.Second))
		}
	}()

	// Запуск воркеров
	for _, f := range files {
		wg.Add(1)
		go analyzeFile(f, stats, fileStatsChan, &wg, sem, progress, phraseChan)
	}

	// Горутина для закрытия каналов после завершения обработки файлов
	go func() {
		wg.Wait()
		close(fileStatsChan)
		close(phraseChan)
	}()

	// Создаём директорию для выходных файлов
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Cannot create output dir: %v", err)
	}

	// Потоковая запись пофайловой статистики
	fileStatsPath := filepath.Join(*outputDir, "stats_per_file.csv")
	fileStatsFile, err := os.Create(fileStatsPath)
	if err != nil {
		log.Fatalf("Cannot create stats_per_file.csv: %v", err)
	}
	defer fileStatsFile.Close()
	fsWriter := csv.NewWriter(fileStatsFile)
	fsWriter.Write([]string{"file", "sentences", "russian_ratio", "english_ratio", "mixed_ratio", "digit_ratio_30", "list_ratio", "garbage_ratio", "examples"})

	for fs := range fileStatsChan {
		examplesStr := strings.Join(fs.Examples, " | ")
		fsWriter.Write([]string{
			filepath.Base(fs.Path),
			fmt.Sprintf("%d", fs.Sentences),
			fmt.Sprintf("%.4f", fs.RussianRatio),
			fmt.Sprintf("%.4f", fs.EnglishRatio),
			fmt.Sprintf("%.4f", fs.MixedRatio),
			fmt.Sprintf("%.4f", fs.DigitRatio30),
			fmt.Sprintf("%.4f", fs.ListRatio),
			fmt.Sprintf("%.4f", fs.GarbageRatio),
			examplesStr,
		})
	}
	fsWriter.Flush()
	log.Printf("Saved per-file stats to %s", fileStatsPath)

	// Ждём завершения записи сырых фраз
	phraseWg.Wait()

	totalTime := time.Since(progress.startTime)
	log.Printf("[PROGRESS] COMPLETE: %d/%d files processed in %v",
		atomic.LoadInt64(&progress.processedFiles), progress.totalFiles, totalTime.Round(time.Second))

	// Внешняя сортировка для частых фраз
	rawPhrasesPath := filepath.Join(*outputDir, "phrases_raw.txt")
	frequentPath := filepath.Join(*outputDir, "frequent_phrases.csv")
	log.Printf("Sorting and counting frequent phrases...")
	// Используем sort | uniq -c | sort -nr
	cmd := exec.Command("bash", "-c",
		fmt.Sprintf("sort -S 4G '%s' | uniq -c | sort -nr > '%s'", rawPhrasesPath, frequentPath))
	if err := cmd.Run(); err != nil {
		log.Printf("ERROR during external sort: %v", err)
	} else {
		log.Printf("Saved frequent phrases to %s", frequentPath)
		// Опционально удаляем сырой файл
		os.Remove(rawPhrasesPath)
	}

	// Подсчёт кандидатов в колонтитулы (фразы с частотой >10)
	// Читаем первую колонку из frequent_phrases.csv
	frequentFile, err := os.Open(frequentPath)
	if err == nil {
		defer frequentFile.Close()
		scanner := bufio.NewScanner(frequentFile)
		var candidates int64
		for scanner.Scan() {
			line := scanner.Text()
			// Формат: "  12345 phrase text"
			line = strings.TrimSpace(line)
			parts := strings.SplitN(line, " ", 2)
			if len(parts) >= 1 {
				var count int
				fmt.Sscanf(parts[0], "%d", &count)
				if count > 10 {
					candidates++
				}
			}
		}
		stats.RunningHeaderCandidates = candidates
	}

	// Вычисляем перцентили по гистограмме
	percentiles := []float64{1, 5, 10, 25, 50, 75, 90, 95, 99}
	percentileValues := make(map[string]int64)
	for _, p := range percentiles {
		percentileValues[fmt.Sprintf("%.0f", p)] = stats.percentile(p)
	}

	// Минимальная и максимальная длина (приблизительно по гистограмме)
	var minLength, maxLength int64
	for i, count := range stats.lengthBuckets {
		if count > 0 {
			minLength = int64(i) * 10
			break
		}
	}
	for i := len(stats.lengthBuckets) - 1; i >= 0; i-- {
		if stats.lengthBuckets[i] > 0 {
			maxLength = int64(i)*10 + 10
			break
		}
	}

	// Сохраняем общую статистику в JSON
	summary := map[string]interface{}{
		"total_sentences":           stats.TotalSentences,
		"total_chars":               stats.TotalChars,
		"total_files":               len(files),
		"empty_sentences":           stats.EmptySentences,
		"min_length":                minLength,
		"max_length":                maxLength,
		"avg_length":                float64(stats.TotalChars) / float64(stats.TotalSentences),
		"percentiles":               percentileValues,
		"russian_only":              stats.RussianOnly,
		"english_only":              stats.EnglishOnly,
		"mixed_cyrillic_latin":      stats.MixedCyrillicLatin,
		"other":                     stats.Other,
		"too_short_20":              stats.TooShort20,
		"too_short_50":              stats.TooShort50,
		"too_long_1000":             stats.TooLong1000,
		"too_long_2000":             stats.TooLong2000,
		"high_digit_10":             stats.HighDigit10,
		"high_digit_30":             stats.HighDigit30,
		"high_digit_50":             stats.HighDigit50,
		"high_punct_30":             stats.HighPunct30,
		"high_punct_50":             stats.HighPunct50,
		"list_marker":               stats.ListMarker,
		"has_isbn":                  stats.HasISBN,
		"has_udk":                   stats.HasUDK,
		"has_bbk":                   stats.HasBBK,
		"has_url":                   stats.HasURL,
		"has_email":                 stats.HasEmail,
		"high_uppercase_50":         stats.HighUppercase50,
		"high_uppercase_80":         stats.HighUppercase80,
		"adjacent_duplicates":       stats.AdjacentDuplicates,
		"running_header_candidates": stats.RunningHeaderCandidates,
		"broken_sentences":          stats.BrokenSentences,
		"index_entries":             stats.IndexEntries,
		"hyphenated_words_left":     stats.HyphenatedWordsLeft,
		"greek_letters_count":       stats.GreekLettersCount,
		"math_symbols_count":        stats.MathSymbolsCount,
		"analysis_time_seconds":     totalTime.Seconds(),
	}

	jsonPath := filepath.Join(*outputDir, "stats_summary.json")
	jsonData, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		log.Printf("ERROR marshalling JSON: %v", err)
	} else {
		if err := os.WriteFile(jsonPath, jsonData, 0644); err != nil {
			log.Printf("ERROR writing JSON: %v", err)
		} else {
			log.Printf("Saved summary to %s", jsonPath)
		}
	}

	// Сохраняем гистограмму длин
	histPath := filepath.Join(*outputDir, "stats_length_histogram.csv")
	histFile, err := os.Create(histPath)
	if err != nil {
		log.Printf("ERROR creating histogram file: %v", err)
	} else {
		defer histFile.Close()
		histWriter := csv.NewWriter(histFile)
		histWriter.Write([]string{"bucket", "count"})
		for i, count := range stats.lengthBuckets {
			if count > 0 {
				start := int64(i) * 10
				end := start + 10
				if i == 1000 {
					histWriter.Write([]string{fmt.Sprintf(">%d", start), fmt.Sprintf("%d", count)})
				} else {
					histWriter.Write([]string{fmt.Sprintf("%d-%d", start, end), fmt.Sprintf("%d", count)})
				}
			}
		}
		histWriter.Flush()
		log.Printf("Saved histogram to %s", histPath)
	}

	// Вывод в консоль
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("CORPUS ANALYSIS COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total files:      %d\n", len(files))
	fmt.Printf("Total sentences:  %d\n", stats.TotalSentences)
	fmt.Printf("Total chars:      %d\n", stats.TotalChars)
	fmt.Printf("Average length:   %.2f chars\n", float64(stats.TotalChars)/float64(stats.TotalSentences))
	fmt.Printf("Min length:       %d\n", minLength)
	fmt.Printf("Max length:       %d\n", maxLength)
	fmt.Printf("Time:             %v\n", totalTime.Round(time.Second))
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Russian only:     %d (%.2f%%)\n", stats.RussianOnly, float64(stats.RussianOnly)/float64(stats.TotalSentences)*100)
	fmt.Printf("English only:     %d (%.2f%%)\n", stats.EnglishOnly, float64(stats.EnglishOnly)/float64(stats.TotalSentences)*100)
	fmt.Printf("Mixed (Ru+En):    %d (%.2f%%)\n", stats.MixedCyrillicLatin, float64(stats.MixedCyrillicLatin)/float64(stats.TotalSentences)*100)
	fmt.Printf("Other:            %d (%.2f%%)\n", stats.Other, float64(stats.Other)/float64(stats.TotalSentences)*100)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Too short (<20):  %d (%.2f%%)\n", stats.TooShort20, float64(stats.TooShort20)/float64(stats.TotalSentences)*100)
	fmt.Printf("Too long (>1000): %d (%.2f%%)\n", stats.TooLong1000, float64(stats.TooLong1000)/float64(stats.TotalSentences)*100)
	fmt.Printf("High digit (>30%%): %d (%.2f%%)\n", stats.HighDigit30, float64(stats.HighDigit30)/float64(stats.TotalSentences)*100)
	fmt.Printf("List marker:      %d (%.2f%%)\n", stats.ListMarker, float64(stats.ListMarker)/float64(stats.TotalSentences)*100)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Adjacent dupes:   %d (%.2f%%)\n", stats.AdjacentDuplicates, float64(stats.AdjacentDuplicates)/float64(stats.TotalSentences)*100)
	fmt.Printf("Broken sentences: %d (%.2f%%)\n", stats.BrokenSentences, float64(stats.BrokenSentences)/float64(stats.TotalSentences)*100)
	fmt.Printf("Index entries:    %d (%.2f%%)\n", stats.IndexEntries, float64(stats.IndexEntries)/float64(stats.TotalSentences)*100)
	fmt.Printf("Running headers:  %d candidates\n", stats.RunningHeaderCandidates)
	fmt.Printf("Hyphenated left:  %d\n", stats.HyphenatedWordsLeft)
	fmt.Printf("Greek letters:    %d\n", stats.GreekLettersCount)
	fmt.Printf("Math symbols:     %d\n", stats.MathSymbolsCount)
	fmt.Println(strings.Repeat("=", 60))
}
