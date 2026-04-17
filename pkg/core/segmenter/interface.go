package segmenter

import "context"

// Segmenter порт для сегментации текста на предложения
type Segmenter interface {
	// Segment сегментирует один текст
	Segment(ctx context.Context, text string) ([]string, error)

	// SegmentBatch сегментирует несколько текстов за один вызов (оптимизация)
	SegmentBatch(ctx context.Context, texts []string) ([][]string, error)
}
