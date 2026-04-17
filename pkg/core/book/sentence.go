package book

import "time"

// Sentence предложение с метаданными
type Sentence struct {
	BookID    string    `json:"book_id"`
	Text      string    `json:"text"`
	Position  int       `json:"position"` // порядковый номер в книге
	CreatedAt time.Time `json:"created_at"`
}
