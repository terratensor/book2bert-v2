// pkg/textutils/clean.go

package textutils

import (
	"regexp"
	"strings"
	"unicode"
)

// CleanText очищает текст от OCR-артефактов
func CleanText(text string) string {
	// 1. Заменяем неразрывные пробелы на обычные
	text = strings.ReplaceAll(text, "\u00A0", " ")
	text = strings.ReplaceAll(text, "\u2007", " ")
	text = strings.ReplaceAll(text, "\u202F", " ")

	// 2. Нормализуем переносы строк
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	// 3. Удаляем управляющие символы (кроме \n)
	text = regexp.MustCompile(`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`).ReplaceAllString(text, "")

	// 4. Восстанавливаем переносы слов "внима- \n тельно" → "внимательно"
	text = fixHyphenatedWords(text)

	// 5. Удаляем строки с метаданными (ISBN, УДК, ББК)
	text = regexp.MustCompile(`(?m)^.*\b(?:ISBN|УДК|ББК|ISSN|DOI)\b.*$`).ReplaceAllString(text, "")

	// 6. Удаляем нумерацию (строки, состоящие только из цифр и точки)
	text = regexp.MustCompile(`(?m)^\s*\d+\.?\s*$`).ReplaceAllString(text, "")

	// Удаляем колонцифры
	text = regexp.MustCompile(`(?m)^\s*\d+\s*$`).ReplaceAllString(text, "")
	text = regexp.MustCompile(`(?m)^\s*\d+[~'`+"`"+`"]?\s*$`).ReplaceAllString(text, "")
	text = regexp.MustCompile(`(?m)^[\*\-\=]{3,}$`).ReplaceAllString(text, "")

	// 7. Удаляем оглавления (цифры в конце строки)
	text = regexp.MustCompile(`([А-Яа-яЁё])(\d+)(?:\s|$)`).ReplaceAllString(text, "$1")

	// 8. Удаляем строки-таблицы (много цифр и знаков)
	lines := strings.Split(text, "\n")
	var cleanedLines []string
	for _, line := range lines {
		if isTableLine(line) {
			continue
		}
		if isBibliographicLine(line) {
			continue
		}
		if isGarbageLine(line) {
			continue
		}
		cleanedLines = append(cleanedLines, line)
	}
	text = strings.Join(cleanedLines, "\n")

	// 9. Удаляем специальные символы
	specialChars := regexp.MustCompile(`[▲►▼◄■□▪▫●○◦★☆♦✓✗→←↑↓]`)
	text = specialChars.ReplaceAllString(text, "")

	// 10. Удаляем повторяющиеся символы
	text = regexp.MustCompile(`(?m)^[=\-*_]{10,}$`).ReplaceAllString(text, "")

	// 11. Удаляем ПУСТЫЕ СТРОКИ
	lines = strings.Split(text, "\n")
	var nonEmptyLines []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			nonEmptyLines = append(nonEmptyLines, line)
		}
	}
	text = strings.Join(nonEmptyLines, "\n")

	// 12. Нормализуем пробелы
	text = regexp.MustCompile(`[ \t]+`).ReplaceAllString(text, " ")

	// 13. Удаляем пробелы перед знаками препинания
	text = regexp.MustCompile(`\s+([.,!?;:])`).ReplaceAllString(text, "$1")

	// 14. Удаление Word/XML тегов (ТОЛЬКО ЭТО ДОБАВЛЕНО!)
	text = cleanWordXMLTags(text)

	return strings.TrimSpace(text)
}

// cleanWordXMLTags удаляет теги Word/XML, но НЕ трогает остальной текст
func cleanWordXMLTags(text string) string {
	// Полные Word XML теги: </w:tab>, <w:ind w:firstLine="360"/>, и т.д.
	wordXMLRegex := regexp.MustCompile(`</?w:[^>]+/?>`)
	text = wordXMLRegex.ReplaceAllString(text, "")

	// Остатки тегов после поломанной очистки
	brokenTagRegex := regexp.MustCompile(`<w:[^\s>]*`)
	text = brokenTagRegex.ReplaceAllString(text, "")

	// Атрибуты тегов
	tagAttrRegex := regexp.MustCompile(`\bw:[a-z]+="[^"]*"`)
	text = tagAttrRegex.ReplaceAllString(text, "")

	// Одиночные закрывающие скобки от тегов
	text = regexp.MustCompile(`\s*/?>`).ReplaceAllString(text, "")

	// Убираем пустые строки, которые могли остаться после удаления тегов
	lines := strings.Split(text, "\n")
	var cleanedLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !regexp.MustCompile(`^[\s<>/]+$`).MatchString(trimmed) {
			cleanedLines = append(cleanedLines, line)
		}
	}
	return strings.Join(cleanedLines, "\n")
}

// fixHyphenatedWords исправляет переносы слов
func fixHyphenatedWords(text string) string {
	re := regexp.MustCompile(`(\p{L}+)-\s*\n\s*(\p{L}+)`)
	text = re.ReplaceAllString(text, "$1-$2")
	return text
}

// isTableLine проверяет, является ли строка таблицей
func isTableLine(line string) bool {
	if len(line) < 20 {
		return false
	}
	digitCount := 0
	punctCount := 0
	for _, r := range line {
		if unicode.IsDigit(r) {
			digitCount++
		} else if unicode.IsPunct(r) || r == ',' || r == '.' || r == ';' {
			punctCount++
		}
	}
	total := len([]rune(line))
	if total == 0 {
		return false
	}
	ratio := float64(digitCount+punctCount) / float64(total)
	return ratio > 0.5
}

// isBibliographicLine проверяет строки с библиографическими ссылками
func isBibliographicLine(line string) bool {
	patterns := []string{
		`[А-ЯЁ]\s+[ѴІ]\.\s*[А-ЯЁ]?\d+`,
		`[А-ЯЁ]{2,3}\s+[А-ЯЁ]?\.\d+`,
		`[А-ЯЁ]\.\s*\d+\.\d+`,
	}
	for _, pattern := range patterns {
		if matched, _ := regexp.MatchString(pattern, line); matched {
			return true
		}
	}
	return false
}

// isGarbageLine проверяет строки с битыми символами
func isGarbageLine(line string) bool {
	if len([]rune(line)) < 10 {
		return false
	}
	letterCount := 0
	for _, r := range line {
		if unicode.IsLetter(r) {
			letterCount++
		}
	}
	total := len([]rune(line))
	if total == 0 {
		return false
	}
	ratio := float64(letterCount) / float64(total)
	return ratio < 0.3
}

// IsAcceptableChar проверяет, можно ли оставить символ
func IsAcceptableChar(r rune) bool {
	switch {
	case r == ' ' || r == '\n' || r == '\r' || r == '\t':
		return true
	case r >= 0x20 && r <= 0x7E:
		return true
	case r >= 0x0400 && r <= 0x052F:
		return true
	case r >= 0x2E00 && r <= 0x2E7F:
		return true
	case r == '—' || r == '–' || r == '…':
		return true
	default:
		return false
	}
}

// FilterNonRussian удаляет символы, не относящиеся к русскому/английскому
func FilterNonRussian(text string) string {
	var result []rune
	for _, r := range text {
		if IsAcceptableChar(r) {
			result = append(result, r)
		}
	}
	return string(result)
}
