// pkg/textutils/clean.go

package textutils

import (
	"regexp"
	"strings"
	"unicode"
)

// CleanText очищает текст от OCR-артефактов, Word/XML тегов и прочего мусора
func CleanText(text string) string {
	// ========================================================================
	// ШАГ 0: Удаление Word/XML тегов (НОВОЕ!)
	// ========================================================================

	// Полные Word XML теги: </w:tab>, <w:ind w:firstLine="360"/>, и т.д.
	wordXMLRegex := regexp.MustCompile(`</?w:[^>]+/?>`)
	text = wordXMLRegex.ReplaceAllString(text, "")

	// Остатки тегов после поломанной очистки: <w:tab, <w:ind, и т.д.
	brokenTagRegex := regexp.MustCompile(`<w:[^\s>]*`)
	text = brokenTagRegex.ReplaceAllString(text, "")

	// Атрибуты тегов: w:pos="596", w:val="left", и т.д.
	tagAttrRegex := regexp.MustCompile(`\bw:[a-z]+="[^"]*"`)
	text = tagAttrRegex.ReplaceAllString(text, "")

	// Одиночные закрывающие скобки от тегов: />, >
	text = regexp.MustCompile(`\s*/?>`).ReplaceAllString(text, "")

	// ========================================================================
	// ШАГ 1: Неразрывные пробелы и специальные пробелы
	// ========================================================================

	nbspRunes := []string{
		"\u00A0", // &nbsp;
		"\u2007", // Figure space
		"\u202F", // Narrow no-break space
		"\u2060", // Word joiner
		"\uFEFF", // Zero-width no-break space (BOM)
		"\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
		"\u2008", "\u2009", "\u200A", "\u205F", "\u3000",
	}
	for _, r := range nbspRunes {
		text = strings.ReplaceAll(text, r, " ")
	}

	// ========================================================================
	// ШАГ 2: Zero-width и невидимые символы
	// ========================================================================

	zeroWidthRunes := []string{
		"\u200B",                                         // Zero-width space
		"\u200C",                                         // Zero-width non-joiner
		"\u200D",                                         // Zero-width joiner
		"\u00AD",                                         // Soft hyphen
		"\u034F",                                         // Combining grapheme joiner
		"\u061C",                                         // Arabic letter mark
		"\u180E",                                         // Mongolian vowel separator
		"\uFEFF",                                         // BOM
		"\u202A", "\u202B", "\u202C", "\u202D", "\u202E", // Directional formatting
		"\u2061", "\u2062", "\u2063", "\u2064", // Invisible operators
		"\u2066", "\u2067", "\u2068", "\u2069", // Directional isolates
		"\u206A", "\u206B", "\u206C", "\u206D", "\u206E", "\u206F", // Deprecated
	}
	for _, r := range zeroWidthRunes {
		text = strings.ReplaceAll(text, r, "")
	}

	// ========================================================================
	// ШАГ 3: Нормализация переносов строк
	// ========================================================================

	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	// ========================================================================
	// ШАГ 4: Удаление управляющих символов (C0, C1, DEL)
	// ========================================================================

	text = regexp.MustCompile(`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`).ReplaceAllString(text, "")

	// ========================================================================
	// ШАГ 5: Восстановление переносов слов (сохраняем дефис)
	// ========================================================================

	text = fixHyphenatedWords(text)

	// ========================================================================
	// ШАГ 6: Удаление строк с метаданными (ISBN, УДК, ББК)
	// ========================================================================

	text = regexp.MustCompile(`(?m)^.*\b(?:ISBN|УДК|ББК|ISSN|DOI)\b.*$`).ReplaceAllString(text, "")

	// ========================================================================
	// ШАГ 7: Удаление колонцифр и разделителей
	// ========================================================================

	text = regexp.MustCompile(`(?m)^\s*\d+\.?\s*$`).ReplaceAllString(text, "")
	text = regexp.MustCompile(`(?m)^\s*\d+\s*$`).ReplaceAllString(text, "")
	text = regexp.MustCompile(`(?m)^\s*\d+[~'`+"`"+`"]?\s*$`).ReplaceAllString(text, "")
	text = regexp.MustCompile(`(?m)^[\*\-\=]{3,}$`).ReplaceAllString(text, "")

	// ========================================================================
	// ШАГ 8: Удаление цифр в конце слов (оглавления)
	// ========================================================================

	text = regexp.MustCompile(`([А-Яа-яЁё])(\d+)(?:\s|$)`).ReplaceAllString(text, "$1")

	// ========================================================================
	// ШАГ 9: Удаление строк-таблиц и мусорных строк
	// ========================================================================

	lines := strings.Split(text, "\n")
	var cleanedLines []string
	for _, line := range lines {
		// Пропускаем табличные строки
		if isTableLine(line) {
			continue
		}
		// Пропускаем библиографические строки
		if isBibliographicLine(line) {
			continue
		}
		// Пропускаем мусорные строки
		if isGarbageLine(line) {
			continue
		}
		// Пропускаем строки с XML/Word мусором (НОВОЕ!)
		if isXMLGarbageLine(line) {
			continue
		}
		// Пропускаем строки с OCR-артефактами (НОВОЕ!)
		if isOCRGarbageLine(line) {
			continue
		}
		cleanedLines = append(cleanedLines, line)
	}
	text = strings.Join(cleanedLines, "\n")

	// ========================================================================
	// ШАГ 10: Удаление специальных символов-маркеров
	// ========================================================================

	specialChars := regexp.MustCompile(`[▲►▼◄■□▪▫●○◦★☆♦✓✗→←↑↓]`)
	text = specialChars.ReplaceAllString(text, "")

	// ========================================================================
	// ШАГ 11: Удаление пустых строк
	// ========================================================================

	lines = strings.Split(text, "\n")
	var nonEmptyLines []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			nonEmptyLines = append(nonEmptyLines, line)
		}
	}
	text = strings.Join(nonEmptyLines, "\n")

	// ========================================================================
	// ШАГ 12: Нормализация пробелов
	// ========================================================================

	text = regexp.MustCompile(`[ \t]+`).ReplaceAllString(text, " ")
	text = regexp.MustCompile(`\s+([.,!?;:])`).ReplaceAllString(text, "$1")

	// ========================================================================
	// ШАГ 13: Специфичные замены
	// ========================================================================

	// text = strings.ReplaceAll(text, "…", "...")
	// text = strings.ReplaceAll(text, "–", "-")
	// text = strings.ReplaceAll(text, "—", "-")
	// text = strings.ReplaceAll(text, "―", "-")
	// text = strings.ReplaceAll(text, "−", "-")

	return strings.TrimSpace(text)
}

// isXMLGarbageLine проверяет, состоит ли строка в основном из XML/Word мусора
func isXMLGarbageLine(line string) bool {
	// Удаляем все буквы и цифры
	nonText := regexp.MustCompile(`[^\p{L}\p{N}]`).ReplaceAllString(line, "")
	textContent := regexp.MustCompile(`[\p{L}\p{N}]`).ReplaceAllString(line, "")

	if len(textContent) == 0 {
		return true
	}

	// Если не-текстовых символов больше 80% — это мусор
	if float64(len(nonText))/float64(len(line)) > 0.8 {
		return true
	}

	// Проверка на паттерны XML тегов
	xmlPatterns := []string{
		`</?w:`, `<w:`, `/>`, `</`, `xml`, `XML`,
	}
	for _, p := range xmlPatterns {
		if strings.Contains(line, p) {
			// Если есть XML и мало букв — точно мусор
			letterRatio := float64(len(regexp.MustCompile(`\p{L}`).FindAllString(line, -1))) / float64(len([]rune(line)))
			if letterRatio < 0.3 {
				return true
			}
		}
	}

	return false
}

// isOCRGarbageLine проверяет строки с OCR-артефактами
func isOCRGarbageLine(line string) bool {
	// 1. Разреженные слова: "б а р а б а н щ и к"
	if regexp.MustCompile(`\b\w( \w){4,}\b`).MatchString(line) {
		return true
	}

	// 2. Одиночные буквы, разделенные пробелами, в начале предложения
	if regexp.MustCompile(`^[А-ЯЁ]\s+[а-яё]\s+[а-яё]`).MatchString(line) {
		return true
	}

	// 3. Перемешаны кириллица и латиница в одном "слове"
	if regexp.MustCompile(`\b[а-яё]+[a-z]+[а-яё]*\b`).MatchString(strings.ToLower(line)) {
		return true
	}

	// 4. Слишком много одиночных букв
	words := strings.Fields(line)
	if len(words) > 5 {
		singleLetters := 0
		for _, w := range words {
			if len(w) == 1 && unicode.IsLetter([]rune(w)[0]) {
				singleLetters++
			}
		}
		if float64(singleLetters)/float64(len(words)) > 0.5 {
			return true
		}
	}

	// 5. Много не-буквенных символов подряд
	if regexp.MustCompile(`[^а-яё\s]{5,}`).MatchString(strings.ToLower(line)) {
		return true
	}

	return false
}

// fixHyphenatedWords исправляет переносы слов
func fixHyphenatedWords(text string) string {
	re := regexp.MustCompile(`(\p{L}+)-\s*\n\s*(\p{L}+)`)
	text = re.ReplaceAllString(text, "$1-$2")
	return text
}

// isTableLine проверяет, является ли строка таблицей
func isTableLine(line string) bool {
	if len([]rune(line)) < 20 {
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
