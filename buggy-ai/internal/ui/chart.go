package ui

import (
	"fmt"
	"strings"
)

type DataPoint struct {
	Label string
	Value float64
}

type Chart struct {
	Title  string
	Points []DataPoint
	Width  int
}

func NewChart(title string) *Chart {
	return &Chart{
		Title: title,
		Width: 50,
	}
}

func (c *Chart) AddPoint(label string, value float64) {
	c.Points = append(c.Points, DataPoint{Label: label, Value: value})
}

func (c *Chart) Render() string {
	if len(c.Points) == 0 {
		return "No data"
	}

	maxVal := 0.0
	for _, p := range c.Points {
		if p.Value > maxVal {
			maxVal = p.Value
		}
	}

	if maxVal == 0 {
		maxVal = 1
	}

	scale := float64(c.Width-15) / maxVal

	var result strings.Builder
	result.WriteString(c.Title + "\n")
	for _, p := range c.Points {
		barLen := int(p.Value * scale)
		bar := strings.Repeat("█", barLen)
		label := p.Label
		if len(label) > 10 {
			label = label[:10]
		}
		result.WriteString(fmt.Sprintf("%-10s │%s %.1f\n", label, bar, p.Value))
	}

	return result.String()
}

type Table struct {
	Headers []string
	Rows    [][]string
}

func NewTable() *Table {
	return &Table{}
}

func (t *Table) AddRow(row []string) {
	t.Rows = append(t.Rows, row)
}

func (t *Table) Render() string {
	if len(t.Headers) == 0 {
		return ""
	}

	colWidths := make([]int, len(t.Headers))
	for i, h := range t.Headers {
		colWidths[i] = len(h)
	}

	for _, row := range t.Rows {
		for i, cell := range row {
			if len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	var result strings.Builder

	result.WriteString("┌")
	for i, w := range colWidths {
		result.WriteString(strings.Repeat("─", w+2))
		if i < len(colWidths)-1 {
			result.WriteString("┬")
		}
	}
	result.WriteString("┐\n")

	result.WriteString("│")
	for i, h := range t.Headers {
		result.WriteString(" " + fmt.Sprintf("%-*s", colWidths[i], h) + " │")
	}
	result.WriteString("\n")

	result.WriteString("├")
	for i, w := range colWidths {
		result.WriteString(strings.Repeat("─", w+2))
		if i < len(colWidths)-1 {
			result.WriteString("┼")
		}
	}
	result.WriteString("┤\n")

	for _, row := range t.Rows {
		result.WriteString("│")
		for i, cell := range row {
			result.WriteString(" " + fmt.Sprintf("%-*s", colWidths[i], cell) + " │")
		}
		result.WriteString("\n")
	}

	result.WriteString("└")
	for i, w := range colWidths {
		result.WriteString(strings.Repeat("─", w+2))
		if i < len(colWidths)-1 {
			result.WriteString("┴")
		}
	}
	result.WriteString("┘\n")

	return result.String()
}

type ProgressBar struct {
	Total   int
	Current int
	Width   int
}

func NewProgressBar(total int) *ProgressBar {
	return &ProgressBar{Total: total, Width: 40}
}

func (p *ProgressBar) Render() string {
	if p.Total == 0 {
		return ""
	}

	percent := float64(p.Current) / float64(p.Total) * 100
	filled := int(float64(p.Width) * percent / 100)
	bar := strings.Repeat("█", filled) + strings.Repeat("░", p.Width-filled)

	return fmt.Sprintf("[%s] %d/%d (%.1f%%)", bar, p.Current, p.Total, percent)
}

func Sparkline(data []float64) string {
	if len(data) == 0 {
		return ""
	}

	min, max := data[0], data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	if max == min {
		max = min + 1
	}

	chars := []string{"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"}
	var result strings.Builder

	for _, v := range data {
		idx := int((v - min) / (max - min) * float64(len(chars)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(chars) {
			idx = len(chars) - 1
		}
		result.WriteString(chars[idx])
	}

	return result.String()
}
