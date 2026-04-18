// cmd/build-corpus/main.go
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

// Sentence — минимальная структура для извлечения текста
type Sentence struct {
	Text string `json:"text"`
}

func openFile(path string) (*os.File, *gzip.Reader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}

	if strings.HasSuffix(path, ".gz") {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			file.Close()
			return nil, nil, err
		}
		return file, gzReader, nil
	}

	return file, nil, nil
}

func processFile(filePath string, ch chan<- string, stats *int64) error {
	file, gzReader, err := openFile(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	var scanner *bufio.Scanner
	if gzReader != nil {
		defer gzReader.Close()
		scanner = bufio.NewScanner(gzReader)
	} else {
		scanner = bufio.NewScanner(file)
	}

	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var s Sentence
		if err := json.Unmarshal(line, &s); err != nil {
			continue
		}

		text := strings.TrimSpace(s.Text)
		if len(text) > 0 {
			ch <- text + "\n"
			atomic.AddInt64(stats, 1)
		}
	}

	return scanner.Err()
}

func main() {
	var (
		sentencesDir = flag.String("dir", "", "директория с JSONL файлами")
		outputFile   = flag.String("output", "corpus.txt", "выходной файл корпуса")
		workers      = flag.Int("workers", 32, "количество воркеров")
	)
	flag.Parse()

	if *sentencesDir == "" {
		log.Fatal("--dir is required")
	}

	log.Printf("=== Corpus Builder ===")
	log.Printf("Input dir: %s", *sentencesDir)
	log.Printf("Output file: %s", *outputFile)
	log.Printf("Workers: %d", *workers)

	// Находим все JSONL файлы
	files, err := filepath.Glob(filepath.Join(*sentencesDir, "*.jsonl"))
	if len(files) == 0 {
		files, err = filepath.Glob(filepath.Join(*sentencesDir, "*.jsonl.gz"))
	}
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d JSONL files", len(files))

	if len(files) == 0 {
		log.Fatal("No files found")
	}

	// Создаем выходной файл
	outFile, err := os.Create(*outputFile)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1024*1024) // 1MB buffer
	defer writer.Flush()

	// Канал для строк
	ch := make(chan string, 100000)
	var wg sync.WaitGroup
	var totalSentences int64

	// Воркер для записи (один, чтобы избежать гонок)
	go func() {
		for line := range ch {
			writer.WriteString(line)
		}
	}()

	// Прогресс
	go func() {
		var last int64
		for {
			select {
			default:
				current := atomic.LoadInt64(&totalSentences)
				if current != last {
					log.Printf("[PROGRESS] %d sentences extracted", current)
					last = current
				}
			}
		}
	}()

	// Запускаем воркеров для обработки файлов
	sem := make(chan struct{}, *workers)
	for _, file := range files {
		sem <- struct{}{}
		wg.Add(1)
		go func(f string) {
			defer func() { <-sem; wg.Done() }()
			if err := processFile(f, ch, &totalSentences); err != nil {
				log.Printf("ERROR processing %s: %v", f, err)
			}
		}(file)
	}

	wg.Wait()
	close(ch)
	writer.Flush()

	log.Printf("=== Done ===")
	log.Printf("Total sentences extracted: %d", totalSentences)
	log.Printf("Corpus saved to: %s", *outputFile)
}
