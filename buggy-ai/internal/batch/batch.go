package batch

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type Processor struct {
	config     ProcessorConfig
	queues     map[string]*Queue
	processors int
	mu         sync.RWMutex
}

type ProcessorConfig struct {
	MaxQueueSize    int
	MaxRetries      int
	RetryDelay      time.Duration
	DeadLetterQueue string
	RateLimit       int
	WindowSize      time.Duration
}

type Queue struct {
	Name     string
	Items    []QueueItem
	Priority int
	mu       sync.RWMutex
	cond     *sync.Cond
}

type QueueItem struct {
	ID         string
	Payload    interface{}
	Priority   int
	Retries    int
	MaxRetries int
	CreatedAt  time.Time
	LastTry    time.Time
}

type ProcessorStats struct {
	Processed  int64
	Failed     int64
	Retried    int64
	InQueue    int
	RatePerSec float64
}

func NewProcessor(cfg ProcessorConfig) *Processor {
	return &Processor{
		config: cfg,
		queues: make(map[string]*Queue),
	}
}

func (p *Processor) Enqueue(queueName string, item QueueItem) error {
	p.mu.Lock()
	queue, ok := p.queues[queueName]
	if !ok {
		queue = &Queue{
			Name:  queueName,
			Items: []QueueItem{},
			cond:  sync.NewCond(&sync.Mutex{}),
		}
		p.queues[queueName] = queue
	}
	p.mu.Unlock()

	queue.mu.Lock()
	if len(queue.Items) >= p.config.MaxQueueSize {
		queue.mu.Unlock()
		return fmt.Errorf("queue is full")
	}
	queue.Items = append(queue.Items, item)
	queue.cond.Signal()
	queue.mu.Unlock()

	return nil
}

func (p *Processor) Dequeue(queueName string) *QueueItem {
	p.mu.RLock()
	queue, ok := p.queues[queueName]
	p.mu.RUnlock()

	if !ok {
		return nil
	}

	queue.mu.Lock()
	defer queue.mu.Unlock()

	if len(queue.Items) == 0 {
		return nil
	}

	item := queue.Items[0]
	queue.Items = queue.Items[1:]
	return &item
}

func (p *Processor) Process(ctx context.Context, queueName string, handler func(item QueueItem) error) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			item := p.Dequeue(queueName)
			if item == nil {
				time.Sleep(100 * time.Millisecond)
				continue
			}

			err := handler(*item)
			if err != nil {
				item.Retries++
				if item.Retries < item.MaxRetries {
					time.Sleep(p.config.RetryDelay)
					p.Enqueue(queueName, *item)
				} else {
					p.SendToDeadLetter(*item, err)
				}
			}
		}
	}
}

func (p *Processor) SendToDeadLetter(item QueueItem, err error) {
	p.mu.RLock()
	dlq := p.queues[p.config.DeadLetterQueue]
	p.mu.RUnlock()

	if dlq != nil {
		dlq.mu.Lock()
		dlq.Items = append(dlq.Items, QueueItem{
			ID:        item.ID + "-dlq",
			Payload:   map[string]interface{}{"original": item.Payload, "error": err.Error()},
			Priority:  -1,
			CreatedAt: time.Now(),
		})
		dlq.mu.Unlock()
	}
}

func (p *Processor) GetStats(queueName string) ProcessorStats {
	p.mu.RLock()
	queue, ok := p.queues[queueName]
	p.mu.RUnlock()

	stats := ProcessorStats{
		InQueue: 0,
	}

	if ok {
		queue.mu.RLock()
		stats.InQueue = len(queue.Items)
		queue.mu.RUnlock()
	}

	return stats
}

type RateLimiter struct {
	limit    int
	window   time.Duration
	requests []time.Time
	mu       sync.Mutex
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		limit:    limit,
		window:   window,
		requests: make([]time.Time, 0),
	}
}

func (r *RateLimiter) Allow() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-r.window)

	var valid []time.Time
	for _, t := range r.requests {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	r.requests = valid

	if len(r.requests) >= r.limit {
		return false
	}

	r.requests = append(r.requests, now)
	return true
}

type BatchProcessor struct {
	items     []BatchItem
	batchSize int
	interval  time.Duration
	handler   BatchHandler
	mu        sync.Mutex
}

type BatchItem struct {
	ID       string
	Payload  interface{}
	Metadata map[string]interface{}
}

type BatchHandler func(items []BatchItem) ([]BatchResult, error)

type BatchResult struct {
	ID      string
	Success bool
	Error   string
	Output  interface{}
}

func NewBatchProcessor(batchSize int, interval time.Duration, handler BatchHandler) *BatchProcessor {
	return &BatchProcessor{
		items:     make([]BatchItem, 0),
		batchSize: batchSize,
		interval:  interval,
		handler:   handler,
	}
}

func (b *BatchProcessor) Add(item BatchItem) {
	b.mu.Lock()
	b.items = append(b.items, item)
	b.mu.Unlock()
}

func (b *BatchProcessor) Process() ([]BatchResult, error) {
	b.mu.Lock()
	if len(b.items) == 0 {
		b.mu.Unlock()
		return nil, nil
	}

	batch := b.items[:min(b.batchSize, len(b.items))]
	b.items = b.items[len(batch):]
	b.mu.Unlock()

	return b.handler(batch)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type TaskScheduler struct {
	tasks    map[string]*ScheduledTask
	mu       sync.RWMutex
	interval time.Duration
}

type ScheduledTask struct {
	ID        string
	CronExpr  string
	NextRun   time.Time
	Handler   func() error
	LastRun   time.Time
	LastError error
}

func NewTaskScheduler(interval time.Duration) *TaskScheduler {
	return &TaskScheduler{
		tasks:    make(map[string]*ScheduledTask),
		interval: interval,
	}
}

func (s *TaskScheduler) Schedule(id string, cronExpr string, handler func() error) error {
	s.mu.Lock()
	s.tasks[id] = &ScheduledTask{
		ID:       id,
		CronExpr: cronExpr,
		Handler:  handler,
		NextRun:  time.Now().Add(time.Minute),
	}
	s.mu.Unlock()
	return nil
}

func (s *TaskScheduler) GetDueTasks() []*ScheduledTask {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var due []*ScheduledTask
	now := time.Now()
	for _, task := range s.tasks {
		if !task.NextRun.After(now) {
			due = append(due, task)
		}
	}
	return due
}

func (s *TaskScheduler) RunTask(id string) error {
	s.mu.RLock()
	task, ok := s.tasks[id]
	s.mu.RUnlock()

	if !ok {
		return fmt.Errorf("task not found: %s", id)
	}

	err := task.Handler()
	task.LastRun = time.Now()
	task.LastError = err
	return err
}
