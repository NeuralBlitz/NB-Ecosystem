package cluster

import (
	"fmt"
	"sync"
	"time"
)

type Cluster struct {
	config    ClusterConfig
	nodes     map[NodeID]*Node
	shards    map[int]*Shard
	discovery Discovery
	consensus Consensus
	mu        sync.RWMutex
}

type ClusterConfig struct {
	NodeID            string
	Role              NodeRole
	MasterURL         string
	ShardCount        int
	ReplicationFactor int
	HeartbeatInterval time.Duration
}

type NodeID string

type Node struct {
	ID            NodeID
	Host          string
	Port          int
	Role          NodeRole
	State         NodeState
	Capacity      NodeCapacity
	Metrics       NodeMetrics
	LastHeartbeat time.Time
	mu            sync.RWMutex
}

type NodeRole string

const (
	RoleMaster NodeRole = "master"
	RoleWorker NodeRole = "worker"
	RoleEdge   NodeRole = "edge"
)

type NodeState string

const (
	NodeOnline  NodeState = "online"
	NodeOffline NodeState = "offline"
	NodeBusy    NodeState = "busy"
)

type NodeCapacity struct {
	CPUs      int
	MemoryMB  int
	StorageMB int
	TasksMax  int
	AgentsMax int
}

type NodeMetrics struct {
	CPUUsage     float64
	MemoryUsage  float64
	TasksActive  int
	AgentsActive int
	UptimeSec    int64
}

type Shard struct {
	ID       int
	Owner    NodeID
	Replicas []NodeID
	Data     map[string]interface{}
	mu       sync.RWMutex
}

type Discovery interface {
	RegisterNode(node *Node) error
	DeregisterNode(nodeID NodeID) error
	ListNodes() ([]*Node, error)
}

type DiscoveryConfig struct {
	Type   string
	Prefix string
	TTL    int
}

type DiscoveryService struct {
	nodes map[NodeID]*Node
	mu    sync.RWMutex
}

func NewDiscoveryService() *DiscoveryService {
	return &DiscoveryService{
		nodes: make(map[NodeID]*Node),
	}
}

func (d *DiscoveryService) RegisterNode(node *Node) error {
	d.mu.Lock()
	d.nodes[node.ID] = node
	d.mu.Unlock()
	return nil
}

func (d *DiscoveryService) DeregisterNode(nodeID NodeID) error {
	d.mu.Lock()
	delete(d.nodes, nodeID)
	d.mu.Unlock()
	return nil
}

func (d *DiscoveryService) ListNodes() ([]*Node, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	nodes := make([]*Node, 0, len(d.nodes))
	for _, node := range d.nodes {
		nodes = append(nodes, node)
	}
	return nodes, nil
}

type Consensus interface {
	Propose(value interface{}) error
	Commit() error
	GetValue() interface{}
}

type RaftConsensus struct {
	mu          sync.RWMutex
	term        int
	log         []interface{}
	commitIndex int
}

func NewRaftConsensus() *RaftConsensus {
	return &RaftConsensus{
		log: make([]interface{}, 0),
	}
}

func (r *RaftConsensus) Propose(value interface{}) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.log = append(r.log, value)
	return nil
}

func (r *RaftConsensus) Commit() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.commitIndex = len(r.log)
	return nil
}

func (r *RaftConsensus) GetValue() interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.commitIndex > 0 {
		return r.log[r.commitIndex-1]
	}
	return nil
}

func NewCluster(cfg ClusterConfig) *Cluster {
	return &Cluster{
		config:    cfg,
		nodes:     make(map[NodeID]*Node),
		shards:    make(map[int]*Shard),
		discovery: NewDiscoveryService(),
		consensus: NewRaftConsensus(),
	}
}

func (c *Cluster) AddNode(node *Node) error {
	c.mu.Lock()
	c.nodes[node.ID] = node
	c.mu.Unlock()

	return c.discovery.RegisterNode(node)
}

func (c *Cluster) RemoveNode(nodeID NodeID) error {
	c.mu.Lock()
	delete(c.nodes, nodeID)
	c.mu.Unlock()

	return c.discovery.DeregisterNode(nodeID)
}

func (c *Cluster) GetNode(nodeID NodeID) (*Node, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	node, ok := c.nodes[nodeID]
	return node, ok
}

func (c *Cluster) ListNodes() []*Node {
	c.mu.RLock()
	defer c.mu.RUnlock()

	nodes := make([]*Node, 0, len(c.nodes))
	for _, node := range c.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

func (c *Cluster) ShardData(key string, data interface{}) error {
	shardID := c.getShardID(key)

	c.mu.Lock()
	if shard, ok := c.shards[shardID]; ok {
		shard.mu.Lock()
		shard.Data[key] = data
		shard.mu.Unlock()
	} else {
		c.shards[shardID] = &Shard{
			ID:    shardID,
			Owner: "",
			Data:  map[string]interface{}{key: data},
		}
	}
	c.mu.Unlock()

	return c.consensus.Propose(map[string]interface{}{
		"shard": shardID,
		"key":   key,
		"value": data,
	})
}

func (c *Cluster) GetShardData(key string) (interface{}, bool) {
	shardID := c.getShardID(key)

	c.mu.RLock()
	shard, ok := c.shards[shardID]
	c.mu.RUnlock()

	if !ok {
		return nil, false
	}

	shard.mu.RLock()
	defer shard.mu.RUnlock()
	value, ok := shard.Data[key]
	return value, ok
}

func (c *Cluster) getShardID(key string) int {
	hash := 0
	for _, c := range key {
		hash += int(c)
	}
	return hash % c.config.ShardCount
}

func (c *Cluster) ReassignShards() error {
	c.mu.Lock()
	nodes := c.ListNodes()
	c.mu.Unlock()

	if len(nodes) == 0 {
		return fmt.Errorf("no nodes available")
	}

	c.mu.Lock()
	for shardID, shard := range c.shards {
		node := nodes[shardID%len(nodes)]
		shard.Owner = node.ID
	}
	c.mu.Unlock()

	return c.consensus.Commit()
}

func (c *Cluster) Heartbeat(nodeID NodeID) error {
	c.mu.RLock()
	node, ok := c.nodes[nodeID]
	c.mu.RUnlock()

	if !ok {
		return fmt.Errorf("node not found: %s", nodeID)
	}

	node.mu.Lock()
	node.LastHeartbeat = time.Now()
	node.mu.Unlock()

	return nil
}

func (c *Cluster) GetStats() ClusterStats {
	c.mu.RLock()
	nodes := c.ListNodes()
	c.mu.RUnlock()

	online := 0
	offline := 0

	for _, node := range nodes {
		node.mu.RLock()
		if node.State == NodeOnline {
			online++
		} else {
			offline++
		}
		node.mu.RUnlock()
	}

	return ClusterStats{
		TotalNodes:   len(nodes),
		OnlineNodes:  online,
		OfflineNodes: offline,
		TotalShards:  len(c.shards),
	}
}

type ClusterStats struct {
	TotalNodes   int
	OnlineNodes  int
	OfflineNodes int
	TotalShards  int
}

func (c *Cluster) Start() error {
	go c.heartbeatLoop()
	return nil
}

func (c *Cluster) heartbeatLoop() {
	ticker := time.NewTicker(c.config.HeartbeatInterval)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.RLock()
		for nodeID := range c.nodes {
			c.Heartbeat(nodeID)
		}
		c.mu.RUnlock()
	}
}
