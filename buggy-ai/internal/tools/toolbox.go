package tools

import "fmt"

type Tool interface {
	Name() string
	Category() string
	Description() string
	Execute(args map[string]interface{}) (interface{}, error)
}

type Toolbox struct {
	tools map[string]Tool
}

func New() *Toolbox {
	return &Toolbox{tools: make(map[string]Tool)}
}

func (t *Toolbox) Register(tool Tool) {
	t.tools[tool.Name()] = tool
}

func (t *Toolbox) Get(name string) (Tool, bool) {
	tool, ok := t.tools[name]
	return tool, ok
}

func (t *Toolbox) Execute(name string, args map[string]interface{}) (interface{}, error) {
	tool, ok := t.tools[name]
	if !ok {
		return nil, fmt.Errorf("tool not found: %s", name)
	}
	return tool.Execute(args)
}

func (t *Toolbox) ListAll() []string {
	var list []string
	for _, tool := range t.tools {
		list = append(list, fmt.Sprintf("[%s] %s - %s", tool.Category(), tool.Name(), tool.Description()))
	}
	return list
}
