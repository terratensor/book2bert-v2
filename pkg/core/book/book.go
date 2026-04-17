package book

// Book доменная модель книги
type Book struct {
	ID       string
	Title    string
	Author   string
	Genre    string
	Source   string // путь к исходному файлу
	Text     string // полный текст (или по частям)
	Metadata map[string]interface{}
}
