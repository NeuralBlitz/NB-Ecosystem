package ai

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Provider interface {
	Chat(messages []Message) (string, error)
	Stream(messages []Message, callback func(chunk string)) error
	GetModel() string
}

func NewProvider(providerType string) Provider {
	switch providerType {
	case "anthropic":
		return &AnthropicProvider{}
	case "gemini":
		return &GeminiProvider{}
	case "ollama":
		return &OllamaProvider{}
	case "local":
		return &LocalProvider{}
	default:
		return &OpenAIProvider{}
	}
}

type OpenAIProvider struct{}

func (p *OpenAIProvider) Chat(messages []Message) (string, error) {
	return "Buggy AI Response (OpenAI)", nil
}

func (p *OpenAIProvider) Stream(messages []Message, callback func(chunk string)) error {
	callback("Buggy ")
	callback("AI ")
	callback("Response")
	return nil
}

func (p *OpenAIProvider) GetModel() string {
	return "gpt-4"
}

type AnthropicProvider struct{}

func (p *AnthropicProvider) Chat(messages []Message) (string, error) {
	return "Buggy AI Response (Anthropic)", nil
}

func (p *AnthropicProvider) Stream(messages []Message, callback func(chunk string)) error {
	callback("Claude ")
	callback("Response")
	return nil
}

func (p *AnthropicProvider) GetModel() string {
	return "claude-3-opus"
}

type GeminiProvider struct{}

func (p *GeminiProvider) Chat(messages []Message) (string, error) {
	return "Buggy AI Response (Gemini)", nil
}

func (p *GeminiProvider) Stream(messages []Message, callback func(chunk string)) error {
	callback("Gemini ")
	callback("Response")
	return nil
}

func (p *GeminiProvider) GetModel() string {
	return "gemini-pro"
}

type OllamaProvider struct{}

func (p *OllamaProvider) Chat(messages []Message) (string, error) {
	return "Buggy AI Response (Ollama)", nil
}

func (p *OllamaProvider) Stream(messages []Message, callback func(chunk string)) error {
	callback("Ollama ")
	callback("Response")
	return nil
}

func (p *OllamaProvider) GetModel() string {
	return "llama2"
}

type LocalProvider struct{}

func (p *LocalProvider) Chat(messages []Message) (string, error) {
	return "Buggy AI Response (Local)", nil
}

func (p *LocalProvider) Stream(messages []Message, callback func(chunk string)) error {
	callback("Local ")
	callback("Model")
	return nil
}

func (p *LocalProvider) GetModel() string {
	return "local-model"
}
