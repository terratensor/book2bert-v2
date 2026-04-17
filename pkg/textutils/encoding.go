package textutils

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	"github.com/saintfish/chardet"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/transform"
)

// DetectEncoding определяет кодировку текста
// Возвращает название кодировки и уверенность (0-100)
func DetectEncoding(text []byte) (string, int) {
	detector := chardet.NewTextDetector()
	result, err := detector.DetectBest(text)
	if err != nil {
		return "utf-8", 0
	}
	return result.Charset, result.Confidence
}

// ToUTF8 конвертирует текст в UTF-8
// Поддерживает Windows-1251, UTF-8, KOI8-R
func ToUTF8(text []byte) (string, error) {
	// Если текст уже валидный UTF-8, возвращаем как есть
	if utf8.Valid(text) {
		return string(text), nil
	}

	// Пробуем определить кодировку
	encoding, confidence := DetectEncoding(text)

	// Если уверенность низкая (< 70%), пробуем Windows-1251 как самую вероятную
	if confidence < 70 {
		encoding = "windows-1251"
	}

	// Конвертируем в UTF-8
	var decoder transform.Transformer

	switch strings.ToLower(encoding) {
	case "windows-1251", "cp1251":
		decoder = charmap.Windows1251.NewDecoder()
	case "koi8-r", "koi8r":
		decoder = charmap.KOI8R.NewDecoder()
	case "iso-8859-1", "latin1":
		decoder = charmap.ISO8859_1.NewDecoder()
	default:
		// Неизвестная кодировка, пробуем Windows-1251 как fallback
		decoder = charmap.Windows1251.NewDecoder()
	}

	reader := transform.NewReader(bytes.NewReader(text), decoder)
	result, err := io.ReadAll(reader)
	if err != nil {
		return "", fmt.Errorf("decode to UTF-8: %w", err)
	}

	return string(result), nil
}

// NormalizeText нормализует текст: удаляет BOM, лишние пробелы
// ВНИМАНИЕ: переносы строк уже нормализованы в CleanText, здесь только BOM и пробелы
func NormalizeText(text string) string {
	// Удаляем BOM (Byte Order Mark) если есть
	text = strings.TrimPrefix(text, "\uFEFF")
	text = strings.TrimPrefix(text, "\uFFFE")

	// Удаляем лишние пробелы в начале и конце строк
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimSpace(line)
	}
	text = strings.Join(lines, "\n")

	return text
}

// ReadFileWithEncoding читает файл и возвращает текст в UTF-8
func ReadFileWithEncoding(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	// Конвертируем в UTF-8
	utf8Text, err := ToUTF8(data)
	if err != nil {
		return "", fmt.Errorf("convert to UTF-8: %w", err)
	}

	// Нормализуем текст
	utf8Text = NormalizeText(utf8Text)

	return utf8Text, nil
}
