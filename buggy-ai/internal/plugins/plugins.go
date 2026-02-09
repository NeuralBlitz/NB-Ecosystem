package plugins

type Plugin interface {
	GetName() string
	Execute(input interface{}) (interface{}, error)
}

type PluginManager struct {
	plugins map[string]Plugin
}

func New() *PluginManager {
	return &PluginManager{plugins: make(map[string]Plugin)}
}

func (m *PluginManager) Register(plugin Plugin) {
	m.plugins[plugin.GetName()] = plugin
}

func (m *PluginManager) Get(name string) (Plugin, bool) {
	p, ok := m.plugins[name]
	return p, ok
}

func (m *PluginManager) Execute(name string, input interface{}) (interface{}, error) {
	p, ok := m.plugins[name]
	if !ok {
		return nil, nil
	}
	return p.Execute(input)
}

type BuiltInPlugin struct {
	PluginName string
}

func (p *BuiltInPlugin) GetName() string { return p.PluginName }
func (p *BuiltInPlugin) Execute(input interface{}) (interface{}, error) {
	return input, nil
}
