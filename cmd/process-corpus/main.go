package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/terratensor/book2bert/v2/pkg/adapters/filerepo"
	segmenterAdapter "github.com/terratensor/book2bert/v2/pkg/adapters/segmenter"
	"github.com/terratensor/book2bert/v2/pkg/core/book"
	"github.com/terratensor/book2bert/v2/pkg/core/segmenter"
	"github.com/terratensor/book2bert/v2/pkg/textutils"
)

var (
	corpusDir    = flag.String("corpus", "", "директория с txt/txt.gz файлами")
	outputDir    = flag.String("output", "data/processed/sentences", "директория для сохранения предложений")
	segmenterURL = flag.String("segmenter", "http://localhost:8090", "URL сервиса сегментации")
	workers      = flag.Int("workers", 16, "количество параллельных воркеров (горутин)")
	extensions   = flag.String("extensions", ".txt,.txt.gz", "расширения файлов для обработки (через запятую)")
)

// FileTask задача для воркера
type FileTask struct {
	Path     string
	Filename string
}

// ProcessStats статистика обработки
type ProcessStats struct {
	Total     int64
	Processed int64
	Errors    int64
	Skipped   int64
	Sentences int64
}

// BookMeta метаданные книги (отдельный файл, не дублируется в предложениях)
type BookMeta struct {
	BookID     string `json:"book_id"`
	Title      string `json:"title,omitempty"`
	Author     string `json:"author,omitempty"`
	Genre      string `json:"genre,omitempty"`
	SourceFile string `json:"source_file"`
}

// parseMetadataFromFilename извлекает жанр, автора и название из имени файла
func parseMetadataFromFilename(filename string) (genre, author, title string) {
	basename := strings.TrimSuffix(filename, ".txt.gz")
	basename = strings.TrimSuffix(basename, ".txt")

	// Паттерн 1: {жанр}_{автор} — {название}
	if idx := strings.Index(basename, " — "); idx != -1 {
		parts := strings.SplitN(basename, " — ", 2)
		if len(parts) == 2 {
			left := parts[0]
			right := parts[1]

			if underscoreIdx := strings.Index(left, "_"); underscoreIdx != -1 {
				genre = left[:underscoreIdx]
				author = left[underscoreIdx+1:]
			} else {
				author = left
			}
			title = right
			return
		}
	}

	// Паттерн 2: militera формат "Автор — Название, год"
	if idx := strings.Index(basename, " — "); idx != -1 {
		parts := strings.SplitN(basename, " — ", 2)
		if len(parts) == 2 {
			author = parts[0]
			title = parts[1]
			// Убираем год в конце (если есть)
			title = regexp.MustCompile(`,\s*\d{4}$`).ReplaceAllString(title, "")
			title = regexp.MustCompile(`\s*\(\d{4}\)$`).ReplaceAllString(title, "")
			return
		}
	}

	// Паттерн 3: просто имя файла
	author = "Unknown"
	title = basename
	return
}

// readGZFile читает содержимое .txt.gz файла
func readGZFile(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return "", err
	}
	defer gzReader.Close()

	data, err := io.ReadAll(gzReader)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// readTXTFile читает содержимое .txt файла
func readTXTFile(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// cleanSpecialChars удаляет служебные символы и восстанавливает переносы слов
func cleanSpecialChars(text string) string {
	// 1. Заменяем неразрывные пробелы на обычные
	text = strings.ReplaceAll(text, "\u00A0", " ") // &nbsp;
	text = strings.ReplaceAll(text, "\u2007", " ") // Figure space
	text = strings.ReplaceAll(text, "\u202F", " ") // Narrow no-break space

	// 2. Удаляем carriage return (оставляем только \n)
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	// 3. Восстанавливаем переносы слов: "внима- \n тельно" → "внимательно"
	//    (дефис + перенос строки + пробелы + продолжение)
	reHyphen := regexp.MustCompile(`(\p{L}+)-\s*\n\s*(\p{L}+)`)
	text = reHyphen.ReplaceAllString(text, "$1$2")

	// 4. Удаляем табуляции и другие управляющие символы (но сохраняем \n)
	reControl := regexp.MustCompile(`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`)
	text = reControl.ReplaceAllString(text, "")

	// 5. Удаляем лишние пробелы (множественные пробелы → один)
	reSpaces := regexp.MustCompile(`[ \t]+`)
	text = reSpaces.ReplaceAllString(text, " ")

	return text
}

// processFile обрабатывает один файл
func processFile(ctx context.Context, task FileTask, seg segmenter.Segmenter, repo book.Repository, stats *ProcessStats, cjkLogFile *os.File, metaChan chan<- BookMeta) {
	defer atomic.AddInt64(&stats.Processed, 1)

	// 1. Читаем файл (поддерживается .txt и .txt.gz)
	var content string
	var err error

	if strings.HasSuffix(task.Path, ".gz") {
		content, err = readGZFile(task.Path)
	} else {
		content, err = readTXTFile(task.Path)
	}

	if err != nil {
		log.Printf("[ERROR] read file %s: %v", task.Filename, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	// 2. Конвертируем кодировку (Windows-1251 → UTF-8)
	text, err := textutils.ToUTF8([]byte(content))
	if err != nil {
		text = content
	}

	// 3. Нормализация текста (удаление BOM, лишних переносов и т.д.)
	text = textutils.NormalizeText(text)

	// 4. Очистка от служебных символов и восстановление переносов
	text = cleanSpecialChars(text)

	// 5. Очистка от OCR-мусора (ISBN, URL, списки и т.д.)
	text = textutils.CleanText(text)

	// 6. Удаление не-русского текста (оставляем только кириллицу)
	text = textutils.FilterNonRussian(text)

	// 7. Проверка на CJK/тайские символы (логируем и удаляем)
	hadCJK := textutils.HasCJKThai(text)
	text = textutils.FilterCJKThai(text)

	// 8. Логируем файлы, содержащие CJK
	if hadCJK {
		cjkLogFile.WriteString(fmt.Sprintf("%s\t%s\t%s\t%s\n",
			task.Filename, "", "", ""))
	}

	// 9. Пропускаем пустые файлы
	if len(strings.TrimSpace(text)) == 0 {
		log.Printf("[SKIP] %s: empty after filtering (had CJK: %v)", task.Filename, hadCJK)
		atomic.AddInt64(&stats.Skipped, 1)
		return
	}

	// 10. Извлекаем метаданные из имени файла
	genre, author, title := parseMetadataFromFilename(task.Filename)
	if title == "" {
		title = strings.TrimSuffix(task.Filename, ".txt.gz")
		title = strings.TrimSuffix(title, ".txt")
	}

	// 11. Генерируем уникальный ID для книги
	bookID := uuid.New().String()

	// 12. Отправляем метаданные в отдельный файл (не дублируются в предложениях)
	metaChan <- BookMeta{
		BookID:     bookID,
		Title:      title,
		Author:     author,
		Genre:      genre,
		SourceFile: task.Path,
	}

	// 13. Отправляем ВЕСЬ текст в сегментатор (razdel)
	//     Сегментатор сам разбивает текст на предложения,
	//     учитывая точки, восклицательные знаки, аббревиатуры и т.д.
	sentences, err := seg.Segment(ctx, text)
	if err != nil {
		log.Printf("[ERROR] segment %s: %v", task.Filename, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	// 14. Сохраняем предложения с сохранением порядка (position)
	bookSentences := make([]book.Sentence, len(sentences))
	for i, sentence := range sentences {
		// Очищаем каждое предложение от лишних пробелов
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			continue
		}
		bookSentences[i] = book.Sentence{
			BookID:    bookID,
			Text:      sentence,
			Position:  i,
			CreatedAt: time.Now(),
		}
	}

	// 15. Сохраняем предложения в JSONL файл (один файл на книгу)
	if err := repo.SaveSentences(ctx, bookSentences); err != nil {
		log.Printf("[ERROR] save sentences %s: %v", task.Filename, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	// 16. Обновляем статистику
	atomic.AddInt64(&stats.Sentences, int64(len(sentences)))
	log.Printf("[OK] %s | genre=%s author=%s sentences=%d | CJK: %v",
		task.Filename, genre, author, len(sentences), hadCJK)
}

// collectFiles собирает все файлы с нужными расширениями
func collectFiles(rootDir string, extensions []string) ([]FileTask, error) {
	var tasks []FileTask

	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		// Проверяем расширение
		for _, ext := range extensions {
			if strings.HasSuffix(path, ext) {
				tasks = append(tasks, FileTask{
					Path:     path,
					Filename: filepath.Base(path),
				})
				break
			}
		}
		return nil
	})

	return tasks, err
}

func main() {
	flag.Parse()

	if *corpusDir == "" {
		log.Fatal("--corpus is required")
	}

	// Парсим расширения
	extList := strings.Split(*extensions, ",")
	for i, ext := range extList {
		extList[i] = strings.TrimSpace(ext)
	}

	log.Printf("=== Corpus Processor v2 (with full filtering) ===")
	log.Printf("Corpus dir: %s", *corpusDir)
	log.Printf("Output dir: %s", *outputDir)
	log.Printf("Segmenter URL: %s", *segmenterURL)
	log.Printf("Workers: %d", *workers)
	log.Printf("Extensions: %v", extList)

	// Создаём выходную директорию
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Создаём CJK лог файл
	cjkLogPath := filepath.Join(*outputDir, "cjk_filtered.log")
	cjkLogFile, err := os.Create(cjkLogPath)
	if err != nil {
		log.Fatalf("create CJK log: %v", err)
	}
	defer cjkLogFile.Close()
	cjkLogFile.WriteString("filename\ttitle\tauthor\tgenre\n")

	// Создаём файл для метаданных книг (отдельно от предложений)
	metaFilePath := filepath.Join(*outputDir, "books_meta.jsonl")
	metaFile, err := os.Create(metaFilePath)
	if err != nil {
		log.Fatalf("create meta file: %v", err)
	}
	defer metaFile.Close()
	metaWriter := bufio.NewWriter(metaFile)
	defer metaWriter.Flush()

	// Канал для метаданных (асинхронная запись)
	metaChan := make(chan BookMeta, 100)
	go func() {
		for meta := range metaChan {
			data, err := json.Marshal(meta)
			if err != nil {
				log.Printf("ERROR marshalling meta: %v", err)
				continue
			}
			metaWriter.Write(data)
			metaWriter.Write([]byte("\n"))
		}
	}()
	defer close(metaChan)

	// Собираем файлы
	log.Printf("Collecting files...")
	tasks, err := collectFiles(*corpusDir, extList)
	if err != nil {
		log.Fatalf("collect files: %v", err)
	}
	log.Printf("Found %d files", len(tasks))

	if len(tasks) == 0 {
		log.Fatal("No files found")
	}

	// Создаем репозиторий для сохранения предложений
	repo, err := filerepo.NewJSONLRepository(*outputDir)
	if err != nil {
		log.Fatalf("create repository: %v", err)
	}
	defer repo.Close()

	// Создаем клиент сегментатора (один на всех)
	seg := segmenterAdapter.NewHTTPClient(*segmenterURL, 120*time.Second)

	// Статистика
	var stats ProcessStats
	stats.Total = int64(len(tasks))

	// Создаем очередь задач
	taskQueue := make(chan FileTask, len(tasks))

	// Контекст с отменой
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Запускаем воркеров
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			log.Printf("[Worker %d] started", workerID)
			for task := range taskQueue {
				select {
				case <-ctx.Done():
					return
				default:
					processFile(ctx, task, seg, repo, &stats, cjkLogFile, metaChan)
				}
			}
			log.Printf("[Worker %d] finished", workerID)
		}(i)
	}

	// Отправляем задачи
	for _, task := range tasks {
		taskQueue <- task
	}
	close(taskQueue)

	// Ждем завершения воркеров
	wg.Wait()

	// Выводим статистику
	log.Printf("\n=== Summary ===")
	log.Printf("Total files: %d", stats.Total)
	log.Printf("Processed: %d", stats.Processed)
	log.Printf("Errors: %d", stats.Errors)
	log.Printf("Skipped: %d", stats.Skipped)
	log.Printf("Total sentences: %d", stats.Sentences)
	log.Printf("CJK log saved to: %s", cjkLogPath)
	log.Printf("Metadata saved to: %s", metaFilePath)
	log.Println("Done!")
}
