package filerepo

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/terratensor/book2bert-v2/pkg/core/book"
)

// JSONLRepository репозиторий для хранения предложений в JSONL формате
type JSONLRepository struct {
	outputDir string
	mu        sync.Mutex
	files     map[string]*os.File
	writers   map[string]*bufio.Writer
}

// NewJSONLRepository создает новый репозиторий
func NewJSONLRepository(outputDir string) (*JSONLRepository, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("create output dir: %w", err)
	}

	return &JSONLRepository{
		outputDir: outputDir,
		files:     make(map[string]*os.File),
		writers:   make(map[string]*bufio.Writer),
	}, nil
}

// SaveBook сохраняет книгу (пока просто логируем, в будущем можно сохранять метаданные отдельно)
func (r *JSONLRepository) SaveBook(ctx context.Context, b *book.Book) error {
	// Для начала просто пропускаем, метаданные будут в каждом предложении
	return nil
}

// SaveSentences сохраняет предложения книги
func (r *JSONLRepository) SaveSentences(ctx context.Context, sentences []book.Sentence) error {
	if len(sentences) == 0 {
		return nil
	}

	bookID := sentences[0].BookID
	filePath := filepath.Join(r.outputDir, bookID+".jsonl")

	r.mu.Lock()
	defer r.mu.Unlock()

	// Открываем файл для книги (если еще не открыт)
	f, exists := r.files[bookID]
	if !exists {
		var err error
		f, err = os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return fmt.Errorf("open file for book %s: %w", bookID, err)
		}
		r.files[bookID] = f
		r.writers[bookID] = bufio.NewWriter(f)
	}

	writer := r.writers[bookID]

	// Записываем каждое предложение
	for _, s := range sentences {
		data, err := json.Marshal(s)
		if err != nil {
			return fmt.Errorf("marshal sentence: %w", err)
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return fmt.Errorf("write sentence: %w", err)
		}
	}

	return writer.Flush()
}

// Close закрывает все открытые файлы
func (r *JSONLRepository) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var lastErr error
	for _, f := range r.files {
		if err := f.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}
