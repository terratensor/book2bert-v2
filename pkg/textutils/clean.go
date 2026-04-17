package textutils

import (
	"regexp"
	"strings"
	"unicode"
)

// CleanText очищает текст от OCR-артефактов
func CleanText(text string) string {
	// 1. Удаляем управляющие символы
	text = regexp.MustCompile(`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`).ReplaceAllString(text, "")

	// 2. Восстанавливаем переносы слов "внима- \n тельно" → "внимательно"
	text = fixHyphenatedWords(text)

	// 3. Удаляем строки с метаданными (ISBN, УДК, ББК)
	text = regexp.MustCompile(`(?m)^.*\b(?:ISBN|УДК|ББК|ISSN|DOI|УДК|ББК)\b.*$`).ReplaceAllString(text, "")

	// 4. Удаляем нумерацию (строки, состоящие только из цифр и точки)
	text = regexp.MustCompile(`(?m)^\s*\d+\.?\s*$`).ReplaceAllString(text, "")

	// 5. Удаляем оглавления (цифры в конце строки)
	text = regexp.MustCompile(`([А-Яа-яЁё])(\d+)(?:\s|$)`).ReplaceAllString(text, "$1")

	// 6. Удаляем строки-таблицы (много цифр и знаков)
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

	// 7. Удаляем специальные символы
	specialChars := regexp.MustCompile(`[▲►▼◄■□▪▫●○◦★☆♦✓✗→←↑↓]`)
	text = specialChars.ReplaceAllString(text, "")

	// 8. Удаляем повторяющиеся символы
	text = regexp.MustCompile(`(?m)^[=\-*_]{10,}$`).ReplaceAllString(text, "")

	// 9. Нормализуем пробелы (один пробел вместо многих)
	text = regexp.MustCompile(`[ \t]+`).ReplaceAllString(text, " ")

	// 10. Удаляем пробелы перед знаками препинания
	text = regexp.MustCompile(`\s+([.,!?;:])`).ReplaceAllString(text, "$1")

	return strings.TrimSpace(text)
}

// fixHyphenatedWords исправляет переносы слов: "внима- \n тельно" → "внимательно"
func fixHyphenatedWords(text string) string {
	re := regexp.MustCompile(`(\p{L}+)-\s*\n\s*(\p{L}+)`)
	text = re.ReplaceAllString(text, "$1$2")
	return text
}

// isTableLine проверяет, является ли строка таблицей (>50% цифр и пунктуации)
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

// isGarbageLine проверяет строки с битыми символам (<30% букв)
func isGarbageLine(line string) bool {
	if len(line) < 10 {
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
	// Разрешённые диапазоны
	switch {
	case r == ' ' || r == '\n' || r == '\r' || r == '\t':
		return true
	case r >= 0x20 && r <= 0x7E: // ASCII
		return true
	case r >= 0x0400 && r <= 0x052F: // Кириллица + расширенная
		return true
	case r >= 0x2E00 && r <= 0x2E7F: // Дополнительная пунктуация
		return true
	case r == '—' || r == '–' || r == '…': // Тире, многоточие
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
