package segmenter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/terratensor/book2bert/v2/pkg/core/segmenter"
)

// HTTPClient адаптер для вызова HTTP-сервиса сегментации
type HTTPClient struct {
	client  *http.Client
	baseURL string
}

// NewHTTPClient создает новый HTTP-клиент для сегментатора
func NewHTTPClient(baseURL string, timeout time.Duration) segmenter.Segmenter {
	return &HTTPClient{
		client: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConnsPerHost: 100,
				MaxIdleConns:        100,
			},
		},
		baseURL: baseURL,
	}
}

// Segment реализует интерфейс Segmenter
func (c *HTTPClient) Segment(ctx context.Context, text string) ([]string, error) {
	reqBody := struct {
		Text string `json:"text"`
	}{Text: text}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/segment", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Sentences []string `json:"sentences"`
		Error     string   `json:"error,omitempty"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if result.Error != "" {
		return nil, fmt.Errorf("service error: %s", result.Error)
	}

	return result.Sentences, nil
}

// SegmentBatch реализует пакетную сегментацию
func (c *HTTPClient) SegmentBatch(ctx context.Context, texts []string) ([][]string, error) {
	reqBody := struct {
		Texts []string `json:"texts"`
	}{Texts: texts}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal batch request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/segment_batch", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create batch request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("batch http request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read batch response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("batch status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Results [][]string `json:"results"`
		Error   string     `json:"error,omitempty"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal batch response: %w", err)
	}

	if result.Error != "" {
		return nil, fmt.Errorf("batch service error: %s", result.Error)
	}

	return result.Results, nil
}
