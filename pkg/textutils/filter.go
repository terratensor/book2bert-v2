package textutils

// CJKRanges диапазоны CJK символов
var CJKRanges = [][2]int{
	{0x4E00, 0x9FFF},   // CJK Unified Ideographs
	{0x3400, 0x4DBF},   // Extension A
	{0x20000, 0x2A6DF}, // Extension B
	{0x2A700, 0x2B73F}, // Extension C
	{0x2B740, 0x2B81F}, // Extension D
	{0x2B820, 0x2CEAF}, // Extension E
	{0x2CEB0, 0x2EBEF}, // Extension F
	{0x30000, 0x3134F}, // Extension G
	{0x31350, 0x323AF}, // Extension H
	{0x2E80, 0x2EFF},   // CJK Radicals
	{0x2F00, 0x2FDF},   // Kangxi Radicals
	{0x2FF0, 0x2FFF},   // Ideographic Description
	{0x3000, 0x303F},   // CJK Symbols
	{0x31C0, 0x31EF},   // CJK Strokes
	{0x3200, 0x32FF},   // Enclosed CJK
	{0x3300, 0x33FF},   // CJK Compatibility
	{0xF900, 0xFAFF},   // Compatibility Ideographs
	{0xFE30, 0xFE4F},   // Compatibility Forms
}

// IsCJK проверяет, является ли символ CJK
func IsCJK(r rune) bool {
	code := int(r)
	for _, rng := range CJKRanges {
		if code >= rng[0] && code <= rng[1] {
			return true
		}
	}
	return false
}

// IsThai проверяет, является ли символ тайским
func IsThai(r rune) bool {
	code := int(r)
	return code >= 0x0E00 && code <= 0x0E7F
}

// FilterCJKThai удаляет CJK и тайские символы из строки
func FilterCJKThai(s string) string {
	var result []rune
	for _, r := range s {
		if !IsCJK(r) && !IsThai(r) {
			result = append(result, r)
		}
	}
	return string(result)
}

// HasCJKThai проверяет, содержит ли строка CJK или тайские символы
func HasCJKThai(s string) bool {
	for _, r := range s {
		if IsCJK(r) || IsThai(r) {
			return true
		}
	}
	return false
}
