package main

import (
	"bufio"
	"flag"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

func main() {
	var (
		inputFile  = flag.String("input", "", "входной файл корпуса")
		outputDir  = flag.String("output", "data/tokenizer", "выходная директория")
		sampleSize = flag.Int("sample", 10000000, "размер сэмпла (количество строк)")
		testSize   = flag.Int("test", 1000000, "размер тестового сэмпла")
	)
	flag.Parse()

	if *inputFile == "" {
		log.Fatal("--input is required")
	}

	// Создаем выходную директорию
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Сначала считаем общее количество строк
	log.Printf("Counting lines in %s...", *inputFile)
	totalLines := countLines(*inputFile)
	log.Printf("Total lines: %d", totalLines)

	if totalLines == 0 {
		log.Fatal("No lines found in input file")
	}

	// Генерируем случайные индексы для сэмпла
	log.Printf("Generating random indices for %d samples...", *sampleSize)
	rand.Seed(time.Now().UnixNano())

	sampleIndices := make(map[int]bool, *sampleSize)
	for len(sampleIndices) < *sampleSize {
		idx := rand.Intn(totalLines)
		sampleIndices[idx] = true
	}

	// Генерируем случайные индексы для тестового сэмпла (не пересекающиеся с основным)
	log.Printf("Generating random indices for %d test samples...", *testSize)
	testIndices := make(map[int]bool, *testSize)
	for len(testIndices) < *testSize {
		idx := rand.Intn(totalLines)
		if !sampleIndices[idx] {
			testIndices[idx] = true
		}
	}

	// Открываем входной файл
	input, err := os.Open(*inputFile)
	if err != nil {
		log.Fatalf("open input: %v", err)
	}
	defer input.Close()

	// Создаем выходные файлы
	samplePath := filepath.Join(*outputDir, "sample_10M.txt")
	testPath := filepath.Join(*outputDir, "sample_test_1M.txt")

	sampleFile, err := os.Create(samplePath)
	if err != nil {
		log.Fatalf("create sample file: %v", err)
	}
	defer sampleFile.Close()

	testFile, err := os.Create(testPath)
	if err != nil {
		log.Fatalf("create test file: %v", err)
	}
	defer testFile.Close()

	sampleWriter := bufio.NewWriterSize(sampleFile, 1024*1024)
	defer sampleWriter.Flush()

	testWriter := bufio.NewWriterSize(testFile, 1024*1024)
	defer testWriter.Flush()

	// Проходим по файлу и выбираем строки
	log.Printf("Extracting samples...")
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	var lineNum int
	sampleCount := 0
	testCount := 0
	lastReport := time.Now()

	for scanner.Scan() {
		line := scanner.Bytes()

		if sampleIndices[lineNum] {
			sampleWriter.Write(line)
			sampleWriter.Write([]byte("\n"))
			sampleCount++
		} else if testIndices[lineNum] {
			testWriter.Write(line)
			testWriter.Write([]byte("\n"))
			testCount++
		}

		lineNum++

		// Прогресс каждые 5 секунд
		if time.Since(lastReport) > 5*time.Second {
			progress := float64(lineNum) / float64(totalLines) * 100
			log.Printf("[PROGRESS] %.1f%% (%d/%d lines), sample: %d, test: %d",
				progress, lineNum, totalLines, sampleCount, testCount)
			lastReport = time.Now()
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("scan: %v", err)
	}

	sampleWriter.Flush()
	testWriter.Flush()

	log.Printf("=== Done ===")
	log.Printf("Sample saved to: %s (%d lines)", samplePath, sampleCount)
	log.Printf("Test saved to: %s (%d lines)", testPath, testCount)

	// Показываем размеры файлов
	sampleInfo, _ := os.Stat(samplePath)
	testInfo, _ := os.Stat(testPath)
	log.Printf("Sample size: %.2f MB", float64(sampleInfo.Size())/1024/1024)
	log.Printf("Test size: %.2f MB", float64(testInfo.Size())/1024/1024)
}

// countLines быстро считает количество строк в файле
func countLines(path string) int {
	file, err := os.Open(path)
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
