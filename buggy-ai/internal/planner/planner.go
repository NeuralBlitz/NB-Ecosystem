package planner

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

type Planner struct {
	config    PlannerConfig
	goals     map[string]*Goal
	knowledge KnowledgeBase
	mu        sync.RWMutex
}

type PlannerConfig struct {
	MaxDepth            int
	MaxBranches         int
	Timeout             time.Duration
	UseRAG              bool
	ConfidenceThreshold float64
}

type Goal struct {
	ID          string
	Description string
	State       GoalState
	Steps       []*PlanStep
	RootCause   string
	Solution    string
	Confidence  float64
	CreatedAt   time.Time
	CompletedAt time.Time
	SubGoals    []*Goal
	Context     map[string]interface{}
}

type GoalState string

const (
	GoalPending   GoalState = "pending"
	GoalRunning   GoalState = "running"
	GoalCompleted GoalState = "completed"
	GoalFailed    GoalState = "failed"
	GoalPaused    GoalState = "paused"
)

type PlanStep struct {
	ID            string
	Description   string
	Action        string
	Preconditions []string
	Effects       []string
	State         StepState
	Result        interface{}
	SubGoal       *Goal
	Dependencies  []string
}

type StepState string

const (
	StepPending StepState = "pending"
	StepReady   StepState = "ready"
	StepRunning StepState = "running"
	StepDone    StepState = "done"
	StepFailed  StepState = "failed"
	StepSkipped StepState = "skipped"
)

type KnowledgeBase struct {
	Facts      []Fact
	Rules      []Rule
	Documents  []Document
	Embeddings map[string][]float32
	mu         sync.RWMutex
}

type Fact struct {
	Subject    string
	Predicate  string
	Object     string
	Confidence float64
	Source     string
	Timestamp  time.Time
}

type Rule struct {
	Head      string
	Body      []string
	Weight    float64
	Triggered int
}

type Document struct {
	ID        string
	Content   string
	Embedding []float32
	Metadata  map[string]string
	CreatedAt time.Time
}

func NewPlanner(cfg PlannerConfig) *Planner {
	return &Planner{
		config: cfg,
		goals:  make(map[string]*Goal),
		knowledge: KnowledgeBase{
			Embeddings: make(map[string][]float32),
		},
	}
}

func (p *Planner) CreateGoal(description string, context map[string]interface{}) *Goal {
	goal := &Goal{
		ID:          fmt.Sprintf("goal-%d", time.Now().UnixNano()),
		Description: description,
		State:       GoalPending,
		Steps:       []*PlanStep{},
		Context:     context,
		CreatedAt:   time.Now(),
	}

	p.mu.Lock()
	p.goals[goal.ID] = goal
	p.mu.Unlock()

	return goal
}

func (p *Planner) Decompose(goal *Goal) error {
	p.mu.Lock()
	goal.State = GoalRunning
	p.mu.Unlock()

	steps := []*PlanStep{
		{
			ID:          fmt.Sprintf("step-%d-1", time.Now().UnixNano()),
			Description: fmt.Sprintf("Analyze: %s", goal.Description),
			Action:      "analyze",
			State:       StepPending,
		},
		{
			ID:          fmt.Sprintf("step-%d-2", time.Now().UnixNano()),
			Description: "Gather relevant information",
			Action:      "gather",
			State:       StepPending,
		},
		{
			ID:          fmt.Sprintf("step-%d-3", time.Now().UnixNano()),
			Description: "Generate solution approach",
			Action:      "generate",
			State:       StepPending,
		},
		{
			ID:          fmt.Sprintf("step-%d-4", time.Now().UnixNano()),
			Description: "Execute solution",
			Action:      "execute",
			State:       StepPending,
		},
		{
			ID:          fmt.Sprintf("step-%d-5", time.Now().UnixNano()),
			Description: "Verify results",
			Action:      "verify",
			State:       StepPending,
		},
	}

	p.mu.Lock()
	goal.Steps = steps
	p.mu.Unlock()

	return nil
}

func (p *Planner) Execute(goal *Goal) error {
	for _, step := range goal.Steps {
		if step.State == StepDone {
			continue
		}

		step.State = StepRunning

		err := p.executeStep(step, goal)
		if err != nil {
			step.State = StepFailed
			goal.State = GoalFailed
			return err
		}

		step.State = StepDone
	}

	goal.State = GoalCompleted
	goal.CompletedAt = time.Now()

	return nil
}

func (p *Planner) executeStep(step *PlanStep, goal *Goal) error {
	switch step.Action {
	case "analyze":
		p.analyze(goal)
	case "gather":
		p.gather(goal)
	case "generate":
		p.generate(goal)
	case "execute":
		p.execute(goal)
	case "verify":
		p.verify(goal)
	}
	return nil
}

func (p *Planner) analyze(goal *Goal) {
	goal.RootCause = p.findRootCause(goal.Description)
	p.knowledge.AddFact(Fact{
		Subject:    goal.ID,
		Predicate:  "analyzed",
		Object:     goal.RootCause,
		Confidence: 0.8,
	})
}

func (p *Planner) gather(goal *Goal) {
	goal.Solution = fmt.Sprintf("Solution approach for: %s", goal.Description)
	p.knowledge.AddFact(Fact{
		Subject:    goal.ID,
		Predicate:  "gathered",
		Object:     goal.Solution,
		Confidence: 0.7,
	})
}

func (p *Planner) generate(goal *Goal) {
	goal.Solution = fmt.Sprintf("Generated solution: Step-by-step plan for %s", goal.Description)
}

func (p *Planner) execute(goal *Goal) {
	goal.Solution = fmt.Sprintf("Executed: %s - Complete", goal.Description)
}

func (p *Planner) verify(goal *Goal) {
	goal.Confidence = 0.95
}

func (p *Planner) findRootCause(query string) string {
	return fmt.Sprintf("Root cause analysis of: %s", query)
}

func (p *Planner) QueryRAG(query string) []Document {
	p.knowledge.mu.RLock()
	defer p.knowledge.mu.RUnlock()
	return p.knowledge.Documents
}

func (p *Planner) AddDocument(doc Document) {
	p.knowledge.mu.Lock()
	p.knowledge.Documents = append(p.knowledge.Documents, doc)
	p.knowledge.Embeddings[doc.ID] = doc.Embedding
	p.mu.Unlock()
}

func (k *KnowledgeBase) AddFact(fact Fact) {
	k.mu.Lock()
	k.Facts = append(k.Facts, fact)
	k.mu.Unlock()
}

func (k *KnowledgeBase) Query(query string) []Fact {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.Facts
}

func (p *Planner) GetGoalStats() GoalStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var pending, running, completed, failed int
	for _, goal := range p.goals {
		switch goal.State {
		case GoalPending:
			pending++
		case GoalRunning:
			running++
		case GoalCompleted:
			completed++
		case GoalFailed:
			failed++
		}
	}

	return GoalStats{
		Total:     len(p.goals),
		Pending:   pending,
		Running:   running,
		Completed: completed,
		Failed:    failed,
	}
}

type GoalStats struct {
	Total     int
	Pending   int
	Running   int
	Completed int
	Failed    int
}

type TreeNode struct {
	Value    string
	Children []*TreeNode
	Score    float64
}

func (p *Planner) BuildDecisionTree(goal string) *TreeNode {
	return &TreeNode{
		Value: goal,
		Children: []*TreeNode{
			{Value: "Option A", Score: 0.8},
			{Value: "Option B", Score: 0.7},
			{Value: "Option C", Score: 0.6},
		},
	}
}

func (p *Planner) EvaluatePath(node *TreeNode) float64 {
	maxScore := 0.0
	for _, child := range node.Children {
		score := p.evaluatePathRecursive(child)
		if score > maxScore {
			maxScore = score
		}
	}
	return maxScore
}

func (p *Planner) evaluatePathRecursive(node *TreeNode) float64 {
	if len(node.Children) == 0 {
		return node.Score
	}

	maxScore := 0.0
	for _, child := range node.Children {
		score := p.evaluatePathRecursive(child)
		if score > maxScore {
			maxScore = score
		}
	}
	return node.Score + maxScore*0.5
}

type RAGEngine struct {
	index     map[string][]int
	documents []Document
	mu        sync.RWMutex
}

func NewRAGEngine() *RAGEngine {
	return &RAGEngine{
		index: make(map[string][]int),
	}
}

func (r *RAGEngine) Index(doc Document) {
	r.mu.Lock()
	docID := len(r.documents)
	r.documents = append(r.documents, doc)

	for _, word := range strings.Fields(strings.ToLower(doc.Content)) {
		if len(word) > 3 {
			r.index[word] = append(r.index[word], docID)
		}
	}
	r.mu.Unlock()
}

func (r *RAGEngine) Search(query string, topK int) []Document {
	r.mu.RLock()
	defer r.mu.RUnlock()

	scores := make(map[int]float64)
	queryWords := strings.Fields(strings.ToLower(query))

	for _, word := range queryWords {
		if docIDs, ok := r.index[word]; ok {
			for _, docID := range docIDs {
				scores[docID] += 1.0
			}
		}
	}

	type scoredDoc struct {
		doc   Document
		score float64
	}

	var results []scoredDoc
	for docID, score := range scores {
		results = append(results, scoredDoc{doc: r.documents[docID], score: score})
	}

	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	var docs []Document
	for i := 0; i < min(topK, len(results)); i++ {
		docs = append(docs, results[i].doc)
	}

	return docs
}

func min(a, b int) int {
	return int(math.Min(float64(a), float64(b)))
}
