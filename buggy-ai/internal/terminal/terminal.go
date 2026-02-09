package terminal

import (
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"
)

type Terminal struct {
	ID     string
	TTY    string
	PID    int
	State  TerminalState
	Screen *Screen
	mu     sync.RWMutex
}

type TerminalState string

const (
	TerminalActive TerminalState = "active"
	TerminalIdle   TerminalState = "idle"
	TerminalClosed TerminalState = "closed"
)

type Screen struct {
	Width   int
	Height  int
	CursorX int
	CursorY int
	Content [][]rune
}

type TerminalManager struct {
	terminals map[string]*Terminal
	active    *Terminal
	mu        sync.RWMutex
}

func NewTerminalManager() *TerminalManager {
	return &TerminalManager{
		terminals: make(map[string]*Terminal),
	}
}

func (tm *TerminalManager) OpenPTY(name string) (*Terminal, error) {
	term := &Terminal{
		ID:    name,
		TTY:   "/dev/ptmx",
		PID:   0,
		State: TerminalActive,
		Screen: &Screen{
			Width:  80,
			Height: 24,
		},
	}

	tm.mu.Lock()
	tm.terminals[name] = term
	tm.active = term
	tm.mu.Unlock()

	return term, nil
}

func (t *Terminal) Write(data []byte) (int, error) {
	return os.Stdout.Write(data)
}

func (t *Terminal) Resize(width, height int) error {
	t.mu.Lock()
	t.Screen.Width = width
	t.Screen.Height = height
	t.mu.Unlock()
	return nil
}

func (tm *TerminalManager) ListTerminals() []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	terms := make([]string, 0, len(tm.terminals))
	for id := range tm.terminals {
		terms = append(terms, id)
	}
	return terms
}

func (tm *TerminalManager) GetTerminal(id string) (*Terminal, bool) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	t, ok := tm.terminals[id]
	return t, ok
}

func (tm *TerminalManager) Broadcast(message string) error {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	for _, term := range tm.terminals {
		term.Write([]byte(message))
	}
	return nil
}

type Session struct {
	ID        string
	Command   string
	StartTime time.Time
	Output    []byte
	Status    int
	PID       int
}

type SessionManager struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*Session),
	}
}

func (sm *SessionManager) Start(name, command string) (*Session, error) {
	cmd := exec.Command("bash", "-c", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}

	session := &Session{
		ID:        name,
		Command:   command,
		StartTime: time.Now(),
		Output:    output,
		PID:       0,
	}

	sm.mu.Lock()
	sm.sessions[name] = session
	sm.mu.Unlock()

	return session, nil
}

func (sm *SessionManager) Get(name string) (*Session, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	s, ok := sm.sessions[name]
	return s, ok
}

func (sm *SessionManager) Kill(name string) error {
	sm.mu.RLock()
	session, ok := sm.sessions[name]
	sm.mu.RUnlock()

	if !ok {
		return fmt.Errorf("session not found: %s", name)
	}

	return exec.Command("kill", fmt.Sprintf("%d", session.PID)).Run()
}

func (sm *SessionManager) List() []*Session {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	sessions := make([]*Session, 0, len(sm.sessions))
	for _, s := range sm.sessions {
		sessions = append(sessions, s)
	}
	return sessions
}

type Multiplexer struct {
	windows map[string]*Window
	current *Window
	mu      sync.RWMutex
}

type Window struct {
	ID      string
	Title   string
	Command string
	Session *Session
	Pane    *Pane
	Active  bool
}

type Pane struct {
	X, Y, Width, Height int
}

func NewMultiplexer() *Multiplexer {
	return &Multiplexer{
		windows: make(map[string]*Window),
	}
}

func (m *Multiplexer) NewWindow(id, title, command string) *Window {
	window := &Window{
		ID:      id,
		Title:   title,
		Command: command,
		Pane: &Pane{
			X:      0,
			Y:      0,
			Width:  80,
			Height: 24,
		},
		Active: true,
	}

	m.mu.Lock()
	m.windows[id] = window
	m.current = window
	m.mu.Unlock()

	return window
}

func (m *Multiplexer) Switch(id string) error {
	m.mu.RLock()
	window, ok := m.windows[id]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("window not found: %s", id)
	}

	m.mu.Lock()
	if m.current != nil {
		m.current.Active = false
	}
	window.Active = true
	m.current = window
	m.mu.Unlock()

	return nil
}

func (m *Multiplexer) SplitHorizontal(id, newID string) error {
	m.mu.RLock()
	window, ok := m.windows[id]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("window not found: %s", id)
	}

	window.Pane.Height /= 2

	newWindow := &Window{
		ID:    newID,
		Title: "New Pane",
		Pane: &Pane{
			X:      window.Pane.X,
			Y:      window.Pane.Y + window.Pane.Height,
			Width:  window.Pane.Width,
			Height: window.Pane.Height,
		},
	}

	m.mu.Lock()
	m.windows[newID] = newWindow
	m.mu.Unlock()

	return nil
}

func (m *Multiplexer) ListWindows() []*Window {
	m.mu.RLock()
	defer m.mu.RUnlock()

	windows := make([]*Window, 0, len(m.windows))
	for _, w := range m.windows {
		windows = append(windows, w)
	}
	return windows
}

func (m *Multiplexer) Layout() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var layout string
	for id, window := range m.windows {
		status := " "
		if window.Active {
			status = "*"
		}
		layout += fmt.Sprintf("[%s] %s (%dx%d) ", status, id, window.Pane.Width, window.Pane.Height)
	}
	return layout
}

type BackgroundJob struct {
	ID        string
	Command   string
	StartTime time.Time
	PID       int
	Status    string
	Output    []byte
	done      chan bool
}

func NewBackgroundJob(id, command string) *BackgroundJob {
	return &BackgroundJob{
		ID:      id,
		Command: command,
		done:    make(chan bool),
	}
}

func (j *BackgroundJob) Start() {
	go func() {
		cmd := exec.Command("bash", "-c", j.Command)
		output, err := cmd.CombinedOutput()
		j.Output = output
		if err != nil {
			j.Status = "failed"
		} else {
			j.Status = "completed"
		}
		j.done <- true
	}()
}

func (j *BackgroundJob) Wait() {
	<-j.done
}

func (j *BackgroundJob) IsDone() bool {
	select {
	case <-j.done:
		return true
	default:
		return false
	}
}

type JobManager struct {
	jobs map[string]*BackgroundJob
	mu   sync.RWMutex
}

func NewJobManager() *JobManager {
	return &JobManager{
		jobs: make(map[string]*BackgroundJob),
	}
}

func (jm *JobManager) Start(id, command string) *BackgroundJob {
	job := NewBackgroundJob(id, command)
	jm.mu.Lock()
	jm.jobs[id] = job
	jm.mu.Unlock()
	job.Start()
	return job
}

func (jm *JobManager) Get(id string) (*BackgroundJob, bool) {
	jm.mu.RLock()
	defer jm.mu.RUnlock()
	j, ok := jm.jobs[id]
	return j, ok
}

func (jm *JobManager) List() []*BackgroundJob {
	jm.mu.RLock()
	defer jm.mu.RUnlock()

	jobs := make([]*BackgroundJob, 0, len(jm.jobs))
	for _, j := range jm.jobs {
		jobs = append(jobs, j)
	}
	return jobs
}

func (jm *JobManager) Kill(id string) error {
	jm.mu.RLock()
	job, ok := jm.jobs[id]
	jm.mu.RUnlock()

	if !ok {
		return fmt.Errorf("job not found: %s", id)
	}

	return exec.Command("kill", fmt.Sprintf("%d", job.PID)).Run()
}
