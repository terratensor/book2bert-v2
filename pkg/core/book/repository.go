package book

import "context"

// Repository порт для хранения книг и предложений
type Repository interface {
	// SaveBook сохраняет книгу (метаданные)
	SaveBook(ctx context.Context, book *Book) error

	// SaveSentences сохраняет предложения книги
	SaveSentences(ctx context.Context, sentences []Sentence) error

	// LoadBook загружает книгу по ID
	// LoadBook(ctx context.Context, id string) (*Book, error)

	// LoadSentences загружает все предложения книги
	// LoadSentences(ctx context.Context, bookID string) ([]Sentence, error)

	// Close закрывает репозиторий
	Close() error
}
