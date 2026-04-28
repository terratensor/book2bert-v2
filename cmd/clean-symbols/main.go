package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// КОНФИГУРАЦИЯ
// ============================================================================

type CleanStats struct {
	Processed        int64
	TotalSentences   int64
	CleanedSentences int64
	SymbolsRemoved   int64
}

// ============================================================================
// ОЧИСТКА СИМВОЛОВ
// ============================================================================

// cleanProblematicSymbols удаляет все проблемные символы из текста
func cleanProblematicSymbols(text string) (string, int) {
	var cleaned strings.Builder
	removed := 0

	for _, r := range text {
		keep := true

		switch {
		// 1. Неразрывные и специальные пробелы → обычный пробел
		case r == '\u00A0' || r == '\u2007' || r == '\u202F' ||
			r == '\u205F' || r == '\u3000' || r == '\u1680':
			cleaned.WriteRune(' ')
			removed++
			continue

		// 2. Управляющие символы C0 (кроме \n, \r, \t)
		case r < 0x20 && r != '\n' && r != '\r' && r != '\t':
			keep = false

		// 3. Soft hyphen, zero-width, BOM, невидимые
		case r == '\u00AD' || // soft hyphen
			r == '\u200B' || // zero-width space
			r == '\u200C' || // zero-width non-joiner
			r == '\u200D' || // zero-width joiner
			r == '\u2060' || // word joiner
			r == '\uFEFF' || // BOM / zero-width no-break space
			r == '\u200E' || // left-to-right mark
			r == '\u200F' || // right-to-left mark
			r == '\uFFFD' || // replacement character
			r == '\uFFFC': // object replacement character
			keep = false

		// 4. Направляющие символы (directional formatting)
		case (r >= 0x202A && r <= 0x202E) ||
			(r >= 0x2066 && r <= 0x206F):
			keep = false

		// 5. C1 controls (U+007F-U+009F)
		case r >= 0x7F && r <= 0x9F:
			keep = false

		// 6. Приватные зоны Unicode
		case (r >= 0xE000 && r <= 0xF8FF) ||
			(r >= 0xF0000 && r <= 0xFFFFF) ||
			(r >= 0x100000 && r <= 0x10FFFF):
			keep = false

		// 7. OCR-лигатуры → разложение на буквы
		case r == '\uFB01': // ﬁ → fi
			cleaned.WriteString("fi")
			removed++
			continue
		case r == '\uFB02': // ﬂ → fl
			cleaned.WriteString("fl")
			removed++
			continue
		case r == '\uFB00': // ﬀ → ff
			cleaned.WriteString("ff")
			removed++
			continue
		case r == '\u0133': // ĳ → ij
			cleaned.WriteString("ij")
			removed++
			continue
		}

		if keep {
			cleaned.WriteRune(r)
		} else {
			removed++
		}
	}

	return cleaned.String(), removed
}

// ============================================================================
// ОБРАБОТКА ФАЙЛА
// ============================================================================

func processFile(inputPath, outputPath string, stats *CleanStats) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return err
	}
	defer inputFile.Close()

	// Создаём временный файл для безопасной записи
	tmpPath := outputPath + ".tmp"
	if err := os.MkdirAll(filepath.Dir(tmpPath), 0755); err != nil {
		return err
	}

	tmpFile, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	scanner := bufio.NewScanner(inputFile)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)
	writer := bufio.NewWriter(tmpFile)

	var sentences []map[string]interface{}
	var localTotal, localCleaned, localRemoved int64

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

		localTotal++
		cleaned, removed := cleanProblematicSymbols(text)

		if removed > 0 {
			sent["text"] = cleaned
			localCleaned++
			localRemoved += int64(removed)
		}

		sentences = append(sentences, sent)
	}

	if err := scanner.Err(); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return err
	}

	// Записываем результат
	for _, sent := range sentences {
		data, err := json.Marshal(sent)
		if err != nil {
			continue
		}
		writer.Write(data)
		writer.Write([]byte("\n"))
	}
	writer.Flush()
	tmpFile.Close()

	// Атомарно заменяем старый файл новым
	if err := os.Rename(tmpPath, outputPath); err != nil {
		os.Remove(tmpPath)
		return err
	}

	atomic.AddInt64(&stats.Processed, 1)
	atomic.AddInt64(&stats.TotalSentences, localTotal)
	atomic.AddInt64(&stats.CleanedSentences, localCleaned)
	atomic.AddInt64(&stats.SymbolsRemoved, localRemoved)

	return nil
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	var (
		inputDir  = flag.String("input", "", "входная директория с JSONL файлами")
		outputDir = flag.String("output", "", "выходная директория (если не указана = input)")
		workers   = flag.Int("workers", 32, "количество воркеров")
	)
	flag.Parse()

	if *inputDir == "" {
		log.Fatal("--input is required")
	}

	if *outputDir == "" {
		*outputDir = *inputDir
	}

	log.Printf("=== Symbol Cleaner ===")
	log.Printf("Input dir:  %s", *inputDir)
	log.Printf("Output dir: %s", *outputDir)
	log.Printf("Workers:    %d", *workers)

	// Находим все JSONL файлы
	files, err := filepath.Glob(filepath.Join(*inputDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d files", len(files))

	if len(files) == 0 {
		log.Fatal("No JSONL files found")
	}

	stats := &CleanStats{}
	startTime := time.Now()

	// Канал задач
	tasks := make(chan struct {
		input  string
		output string
	}, len(files))

	for _, f := range files {
		base := filepath.Base(f)
		tasks <- struct {
			input  string
			output string
		}{f, filepath.Join(*outputDir, base)}
	}
	close(tasks)

	// Прогресс
	var processed int64
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			proc := atomic.LoadInt64(&processed)
			if proc >= int64(len(files)) {
				return
			}
			elapsed := time.Since(startTime)
			speed := float64(proc) / elapsed.Seconds()
			percent := float64(proc) / float64(len(files)) * 100
			log.Printf("[PROGRESS] %d/%d files (%.1f%%), speed: %.1f files/sec, elapsed: %v",
				proc, len(files), percent, speed, elapsed.Round(time.Second))
		}
	}()

	// Воркеры
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range tasks {
				if err := processFile(task.input, task.output, stats); err != nil {
					log.Printf("ERROR processing %s: %v", task.input, err)
				}
				atomic.AddInt64(&processed, 1)
			}
		}()
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	// Вывод статистики
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("SYMBOL CLEANING COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total files:        %d\n", len(files))
	fmt.Printf("Total sentences:    %d\n", stats.TotalSentences)
	fmt.Printf("Cleaned sentences:  %d (%.2f%%)\n",
		stats.CleanedSentences,
		float64(stats.CleanedSentences)/float64(stats.TotalSentences+1)*100)
	fmt.Printf("Symbols removed:    %d\n", stats.SymbolsRemoved)
	fmt.Printf("Time:               %v\n", totalTime.Round(time.Second))
	fmt.Println(strings.Repeat("=", 60))
}
