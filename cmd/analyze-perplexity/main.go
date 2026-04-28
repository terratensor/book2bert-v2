package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

type PerplexityClient struct {
	baseURL string
	client  *http.Client
}

func NewPerplexityClient(baseURL string) *PerplexityClient {
	return &PerplexityClient{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

type PerplexityResult struct {
	Perplexity float64 `json:"perplexity"`
	IsGarbage  bool    `json:"is_garbage"`
	TextLength int     `json:"text_length"`
	Error      string  `json:"error,omitempty"`
}

func (c *PerplexityClient) AnalyzeBatch(texts []string) ([]PerplexityResult, error) {
	body := struct {
		Texts []string `json:"texts"`
	}{Texts: texts}

	jsonBody, _ := json.Marshal(body)
	resp, err := c.client.Post(c.baseURL+"/perplexity_batch", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Results []PerplexityResult `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Results, nil
}

// analyzeFileList обрабатывает список файлов из текстового файла
func analyzeFileList(
	listFile string,
	client *PerplexityClient,
	outputDir string,
	batchSize int,
) error {
	// Читаем список файлов
	f, err := os.Open(listFile)
	if err != nil {
		return err
	}
	defer f.Close()

	var files []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		file := strings.TrimSpace(scanner.Text())
		if file != "" {
			files = append(files, file)
		}
	}

	log.Printf("Loaded %d files from %s", len(files), listFile)

	// Создаем выходные файлы
	os.MkdirAll(outputDir, 0755)

	keepFile, _ := os.Create(filepath.Join(outputDir, "books_to_keep.txt"))
	defer keepFile.Close()
	removeFile, _ := os.Create(filepath.Join(outputDir, "books_to_remove.txt"))
	defer removeFile.Close()
	reviewFile, _ := os.Create(filepath.Join(outputDir, "books_to_review.jsonl"))
	defer reviewFile.Close()

	keepWriter := bufio.NewWriter(keepFile)
	removeWriter := bufio.NewWriter(removeFile)
	reviewWriter := bufio.NewWriter(reviewFile)
	defer keepWriter.Flush()
	defer removeWriter.Flush()
	defer reviewWriter.Flush()

	var processed int64
	var keptBooks int64
	var removedBooks int64
	var reviewBooks int64

	// Пороги для принятия решений
	const (
		avgPerplexityThreshold = 500.0 // Средняя перплексия > 500 → мусор
		highPerplexityRatio    = 0.5   // >50% предложений с высокой перплексией → мусор
		garbageThreshold       = 0.7   // >70% предложений помечены как garbage → удалить
		keepThreshold          = 0.1   // <10% garbage → оставить
	)

	for _, filePath := range files {
		// Читаем все предложения из файла
		sentences, err := readSentencesFromJSONL(filePath)
		if err != nil {
			log.Printf("ERROR reading %s: %v", filePath, err)
			continue
		}

		if len(sentences) == 0 {
			removeWriter.WriteString(filePath + "\n")
			atomic.AddInt64(&removedBooks, 1)
			atomic.AddInt64(&processed, 1)
			continue
		}

		// Анализируем батчами
		var allResults []PerplexityResult
		for i := 0; i < len(sentences); i += batchSize {
			end := i + batchSize
			if end > len(sentences) {
				end = len(sentences)
			}
			batch := sentences[i:end]

			results, err := client.AnalyzeBatch(batch)
			if err != nil {
				log.Printf("ERROR analyzing batch in %s: %v", filePath, err)
				continue
			}
			allResults = append(allResults, results...)
		}

		// Вычисляем статистику по книге
		var totalPerplexity float64
		var highPerplexityCount int
		var garbageCount int

		for _, r := range allResults {
			totalPerplexity += r.Perplexity
			if r.Perplexity > avgPerplexityThreshold {
				highPerplexityCount++
			}
			if r.IsGarbage {
				garbageCount++
			}
		}

		avgPerplexity := totalPerplexity / float64(len(allResults))
		garbageRatio := float64(garbageCount) / float64(len(allResults))
		highRatio := float64(highPerplexityCount) / float64(len(allResults))

		// Принимаем решение
		bookInfo := map[string]interface{}{
			"file":                  filePath,
			"sentences":             len(sentences),
			"avg_perplexity":        avgPerplexity,
			"garbage_ratio":         garbageRatio,
			"high_perplexity_ratio": highRatio,
		}

		if len(sentences) == 0 || garbageRatio >= garbageThreshold {
			// Удалить
			removeWriter.WriteString(filePath + "\n")
			atomic.AddInt64(&removedBooks, 1)
			bookInfo["decision"] = "remove"
		} else if garbageRatio <= keepThreshold && avgPerplexity < avgPerplexityThreshold {
			// Оставить
			keepWriter.WriteString(filePath + "\n")
			atomic.AddInt64(&keptBooks, 1)
			bookInfo["decision"] = "keep"
		} else {
			// На проверку
			data, _ := json.Marshal(bookInfo)
			reviewWriter.Write(data)
			reviewWriter.Write([]byte("\n"))
			atomic.AddInt64(&reviewBooks, 1)
			bookInfo["decision"] = "review"
		}

		processed := atomic.AddInt64(&processed, 1)
		if processed%100 == 0 {
			log.Printf("Processed %d/%d files, kept: %d, removed: %d, review: %d",
				processed, len(files), keptBooks, removedBooks, reviewBooks)
		}
	}

	log.Printf("=== Done ===")
	log.Printf("Total files: %d", len(files))
	log.Printf("Kept: %d", keptBooks)
	log.Printf("Removed: %d", removedBooks)
	log.Printf("Need review: %d", reviewBooks)
	log.Printf("Results saved to %s", outputDir)

	return nil
}

// readSentencesFromJSONL читает все предложения из JSONL файла
func readSentencesFromJSONL(filePath string) ([]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var sentences []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		var sent struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &sent); err != nil {
			continue
		}
		if sent.Text != "" {
			sentences = append(sentences, sent.Text)
		}
	}

	return sentences, scanner.Err()
}

// sampleJSONL оставляем для обратной совместимости
func sampleJSONL(dir string, sampleRatio float64) ([]string, error) {
	// ... (старый код, можно оставить)
	return nil, nil
}

func main() {
	var (
		fileList   = flag.String("file-list", "", "файл со списком JSONL для анализа")
		inputDir   = flag.String("input", "", "директория с JSONL (если не указан file-list)")
		outputDir  = flag.String("output", "data/analysis/perplexity", "выходная директория")
		serviceURL = flag.String("service", "http://localhost:8093", "URL сервиса перплексии")
		sampleRate = flag.Float64("sample", 0.01, "доля сэмпла (если используется input)")
		batchSize  = flag.Int("batch", 100, "размер батча")
	)
	flag.Parse()

	log.Printf("=== Perplexity Analyzer ===")
	log.Printf("Output: %s", *outputDir)
	log.Printf("Service: %s", *serviceURL)

	// Проверяем сервис
	client := NewPerplexityClient(*serviceURL)
	resp, err := http.Get(*serviceURL + "/health")
	if err != nil {
		log.Fatalf("Service not available: %v", err)
	}
	resp.Body.Close()
	log.Printf("Service OK")

	if *fileList != "" {
		// Режим анализа списка файлов
		if err := analyzeFileList(*fileList, client, *outputDir, *batchSize); err != nil {
			log.Fatalf("Analysis error: %v", err)
		}
	} else if *inputDir != "" {
		// Режим сэмплирования
		log.Printf("Sampling from %s (%.2f%%)", *inputDir, *sampleRate*100)
		// ... (старый код сэмплирования)
	} else {
		log.Fatal("Either --file-list or --input is required")
	}
}
