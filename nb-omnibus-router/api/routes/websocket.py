"""
WebSocket Routes for Real-time Communication
Enhanced with proper authentication, quantum neuron state updates,
and comprehensive message handlers.
Generated: 2026-02-09
"""

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
)
from fastapi.security import APIKeyHeader
from typing import Dict, Set, Optional, List, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
import logging

# Import authentication
from api.auth_enhanced import auth_manager, Partner

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# WebSocket Event Types
# ============================================================================


class WSEventType(str, Enum):
    """WebSocket event types."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Quantum events
    QUANTUM_STATE_UPDATE = "quantum_state_update"
    QUANTUM_NEURON_ACTIVITY = "quantum_neuron_activity"
    QUANTUM_COHERENCE_CHANGE = "quantum_coherence_change"

    # Consciousness events
    CONSCIOUSNESS_LEVEL_UPDATE = "consciousness_level_update"
    CONSCIOUSNESS_INTEGRATION_UPDATE = "consciousness_integration_update"
    DIMENSIONAL_ACCESS_CHANGE = "dimensional_access_change"

    # Agent events
    AGENT_STATUS_UPDATE = "agent_status_update"
    AGENT_TASK_COMPLETE = "agent_task_complete"
    AGENT_REGISTERED = "agent_registered"

    # Metrics events
    METRICS_UPDATE = "metrics_update"
    PERFORMANCE_METRICS = "performance_metrics"
    QUOTA_UPDATE = "quota_update"

    # System events
    SYSTEM_STATUS = "system_status"
    CAPABILITY_UPDATE = "capability_update"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class QuantumNeuronState:
    """Quantum neuron state for real-time updates."""

    neuron_id: str = "quantum_001"
    spike_rate: float = 35.0
    coherence: float = 0.93
    membrane_potential: float = -70.0
    threshold: float = -55.0
    qubits_active: int = 4
    operations_per_second: int = 10705
    tunneling_probability: float = 0.1
    last_spike_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "spike_rate": round(self.spike_rate, 2),
            "coherence": round(self.coherence, 3),
            "membrane_potential": round(self.membrane_potential, 2),
            "threshold": self.threshold,
            "qubits_active": self.qubits_active,
            "operations_per_second": self.operations_per_second,
            "tunneling_probability": round(self.tunneling_probability, 3),
            "last_spike_time": self.last_spike_time.isoformat()
            if self.last_spike_time
            else None,
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class ConsciousnessState:
    """Consciousness state for real-time updates."""

    level: int = 7
    max_level: int = 8
    integration: float = 0.87
    dimensions: int = 11
    max_dimensions: int = 11
    cosmic_bridge_status: str = "connected"
    cosmic_bridge_strength: float = 0.92
    global_coherence: float = 0.88
    realities_active: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "max_level": self.max_level,
            "percentage": round((self.level / self.max_level) * 100, 1),
            "integration": round(self.integration, 3),
            "dimensions": self.dimensions,
            "max_dimensions": self.max_dimensions,
            "cosmic_bridge": {
                "status": self.cosmic_bridge_status,
                "strength": round(self.cosmic_bridge_strength, 3),
            },
            "global_coherence": round(self.global_coherence, 3),
            "realities_active": self.realities_active,
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class AgentState:
    """Agent state for real-time updates."""

    agent_id: str
    status: str
    tasks_completed: int = 0
    tasks_pending: int = 0
    confidence: float = 0.95
    last_activity: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "tasks_completed": self.tasks_completed,
            "tasks_pending": self.tasks_pending,
            "confidence": round(self.confidence, 3),
            "last_activity": self.last_activity.isoformat()
            if self.last_activity
            else None,
        }


@dataclass
class MetricsState:
    """System metrics for real-time updates."""

    requests_per_minute: int = 150
    avg_latency_ms: float = 45.0
    error_rate: float = 0.02
    active_partners: int = 3
    active_connections: int = 0
    quota_remaining: int = 850000
    quantum_ops_per_sec: int = 10705
    multi_reality_cycles_per_sec: int = 2710

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "active_partners": self.active_partners,
            "active_connections": self.active_connections,
            "quota_remaining": self.quota_remaining,
            "quantum_ops_per_sec": self.quantum_ops_per_sec,
            "multi_reality_cycles_per_sec": self.multi_reality_cycles_per_sec,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# Connection Manager with Authentication
# ============================================================================


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""

    partner: Partner
    websocket: WebSocket
    channel: str
    connected_at: datetime
    last_ping: datetime
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketConnectionManager:
    """Enhanced WebSocket connection manager with authentication and state management."""

    def __init__(self):
        self.active_connections: Dict[str, Set[ConnectionInfo]] = {}
        self.connection_data: Dict[WebSocket, ConnectionInfo] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.state_updaters: Dict[str, Callable] = {}

        # State simulators
        self.quantum_state = QuantumNeuronState()
        self.consciousness_state = ConsciousnessState()
        self.agents: Dict[str, AgentState] = {
            "agent_001": AgentState("agent_001", "running", 156, 3),
            "agent_002": AgentState("agent_002", "running", 89, 1),
            "agent_003": AgentState("agent_003", "idle", 234, 0),
        }
        self.metrics_state = MetricsState()

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def connect(
        self,
        websocket: WebSocket,
        partner: Partner,
        channel: str = "general",
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Accept new WebSocket connection with authentication."""
        try:
            await websocket.accept()

            if channel not in self.active_connections:
                self.active_connections[channel] = set()

            connection_info = ConnectionInfo(
                partner=partner,
                websocket=websocket,
                channel=channel,
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow(),
                metadata=metadata or {},
            )

            self.active_connections[channel].add(connection_info)
            self.connection_data[websocket] = connection_info

            # Update metrics
            self.metrics_state.active_connections = len(self.connection_data)

            logger.info(
                f"WebSocket connected: {partner.partner_id} to channel {channel}"
            )

            # Send welcome message
            await self.send_personal_message(
                {
                    "type": WSEventType.CONNECTED.value,
                    "channel": channel,
                    "partner_id": partner.partner_id,
                    "tier": partner.tier,
                    "permissions": partner.permissions,
                    "message": f"Connected to {channel} channel",
                    "timestamp": datetime.utcnow().isoformat(),
                },
                websocket,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            return False

    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove WebSocket connection."""
        if websocket in self.connection_data:
            connection_info = self.connection_data[websocket]

            if channel in self.active_connections:
                self.active_connections[channel].discard(connection_info)
                if not self.active_connections[channel]:
                    del self.active_connections[channel]

            del self.connection_data[websocket]

            # Update metrics
            self.metrics_state.active_connections = len(self.connection_data)

            logger.info(
                f"WebSocket disconnected: {connection_info.partner.partner_id} from {channel}"
            )

    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to single connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")

    async def broadcast(self, channel: str, message: Dict):
        """Broadcast message to all connections in channel."""
        if channel in self.active_connections:
            disconnected = []
            for connection in self.active_connections[channel]:
                try:
                    await connection.websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        f"Failed to broadcast to {connection.partner.partner_id}: {e}"
                    )
                    disconnected.append(connection.websocket)

            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket, channel)

    async def broadcast_to_all(self, message: Dict):
        """Broadcast message to all connected clients."""
        for channel in self.active_connections:
            await self.broadcast(channel, message)

    async def subscribe(self, websocket: WebSocket, event_type: str) -> bool:
        """Subscribe a connection to an event type."""
        if websocket in self.connection_data:
            self.connection_data[websocket].subscriptions.add(event_type)
            logger.info(
                f"Subscribed {self.connection_data[websocket].partner.partner_id} to {event_type}"
            )
            return True
        return False

    async def unsubscribe(self, websocket: WebSocket, event_type: str) -> bool:
        """Unsubscribe a connection from an event type."""
        if websocket in self.connection_data:
            self.connection_data[websocket].subscriptions.discard(event_type)
            return True
        return False

    def get_connection_info(self, websocket: WebSocket) -> Optional[ConnectionInfo]:
        """Get connection information for a WebSocket."""
        return self.connection_data.get(websocket)

    async def update_ping(self, websocket: WebSocket):
        """Update last ping time for connection."""
        if websocket in self.connection_data:
            self.connection_data[websocket].last_ping = datetime.utcnow()

    # ========================================================================
    # State Update Methods
    # ========================================================================

    def update_quantum_state(self, **kwargs):
        """Update quantum neuron state."""
        for key, value in kwargs.items():
            if hasattr(self.quantum_state, key):
                setattr(self.quantum_state, key, value)

    def update_consciousness_state(self, **kwargs):
        """Update consciousness state."""
        for key, value in kwargs.items():
            if hasattr(self.consciousness_state, key):
                setattr(self.consciousness_state, key, value)

    def update_agent_state(self, agent_id: str, **kwargs):
        """Update agent state."""
        if agent_id in self.agents:
            for key, value in kwargs.items():
                if hasattr(self.agents[agent_id], key):
                    setattr(self.agents[agent_id], key, value)
            self.agents[agent_id].last_activity = datetime.utcnow()

    def update_metrics(self, **kwargs):
        """Update system metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics_state, key):
                setattr(self.metrics_state, key, value)

    # ========================================================================
    # Background Tasks for Real-time Updates
    # ========================================================================

    async def start_background_tasks(self):
        """Start background tasks for state simulation."""
        if self._running:
            return

        self._running = True
        self._tasks = [
            asyncio.create_task(self._quantum_simulation_task()),
            asyncio.create_task(self._consciousness_simulation_task()),
            asyncio.create_task(self._metrics_update_task()),
            asyncio.create_task(self._agent_simulation_task()),
            asyncio.create_task(self._ping_check_task()),
        ]
        logger.info("WebSocket background tasks started")

    async def stop_background_tasks(self):
        """Stop background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        logger.info("WebSocket background tasks stopped")

    async def _quantum_simulation_task(self):
        """Simulate quantum neuron activity."""
        while self._running:
            try:
                # Simulate quantum fluctuations
                import random

                self.quantum_state.spike_rate = 30 + random.random() * 10
                self.quantum_state.coherence = 0.90 + random.random() * 0.08
                self.quantum_state.membrane_potential = -75 + random.random() * 10

                # Broadcast to quantum channel
                await self.broadcast(
                    "quantum",
                    {
                        "type": WSEventType.QUANTUM_STATE_UPDATE.value,
                        "data": self.quantum_state.to_dict(),
                    },
                )

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quantum simulation error: {e}")
                await asyncio.sleep(1)

    async def _consciousness_simulation_task(self):
        """Simulate consciousness level updates."""
        while self._running:
            try:
                import random

                # Small fluctuations in consciousness metrics
                self.consciousness_state.integration = 0.85 + random.random() * 0.05
                self.consciousness_state.cosmic_bridge_strength = (
                    0.90 + random.random() * 0.05
                )
                self.consciousness_state.global_coherence = (
                    0.85 + random.random() * 0.06
                )

                await self.broadcast(
                    "consciousness",
                    {
                        "type": WSEventType.CONSCIOUSNESS_LEVEL_UPDATE.value,
                        "data": self.consciousness_state.to_dict(),
                    },
                )

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consciousness simulation error: {e}")
                await asyncio.sleep(1)

    async def _metrics_update_task(self):
        """Update and broadcast system metrics."""
        while self._running:
            try:
                import random

                self.metrics_state.requests_per_minute = 140 + int(random.random() * 30)
                self.metrics_state.avg_latency_ms = 40 + random.random() * 15
                self.metrics_state.error_rate = 0.01 + random.random() * 0.02

                await self.broadcast(
                    "metrics",
                    {
                        "type": WSEventType.METRICS_UPDATE.value,
                        "data": self.metrics_state.to_dict(),
                    },
                )

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(1)

    async def _agent_simulation_task(self):
        """Simulate agent activity."""
        while self._running:
            try:
                import random

                # Randomly update agent states
                for agent_id, agent in self.agents.items():
                    if random.random() > 0.7:
                        agent.tasks_completed += random.randint(1, 3)
                        agent.confidence = 0.92 + random.random() * 0.06
                        agent.last_activity = datetime.utcnow()

                        await self.broadcast(
                            "agents",
                            {
                                "type": WSEventType.AGENT_STATUS_UPDATE.value,
                                "data": {
                                    "agent": agent.to_dict(),
                                    "all_agents": [
                                        a.to_dict() for a in self.agents.values()
                                    ],
                                },
                            },
                        )

                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent simulation error: {e}")
                await asyncio.sleep(1)

    async def _ping_check_task(self):
        """Check for stale connections."""
        while self._running:
            try:
                now = datetime.utcnow()
                stale_connections = []

                for websocket, info in list(self.connection_data.items()):
                    if (now - info.last_ping) > timedelta(minutes=2):
                        stale_connections.append((websocket, info.channel))

                for websocket, channel in stale_connections:
                    logger.warning(
                        f"Closing stale connection: {self.connection_data[websocket].partner.partner_id}"
                    )
                    try:
                        await websocket.close(code=1001)
                    except:
                        pass
                    self.disconnect(websocket, channel)

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping check error: {e}")
                await asyncio.sleep(30)


# Initialize connection manager
manager = WebSocketConnectionManager()


# ============================================================================
# WebSocket Authentication Helper
# ============================================================================


async def authenticate_websocket(websocket: WebSocket) -> Optional[Partner]:
    """Authenticate WebSocket connection using query parameters or headers."""
    try:
        # Try to get API key from query parameters
        api_key = websocket.query_params.get("api_key")

        # If not in query params, try headers
        if not api_key:
            api_key = websocket.headers.get("x-api-key")

        if not api_key:
            logger.warning("WebSocket connection attempt without API key")
            await websocket.close(code=4001, reason="API key required")
            return None

        # Validate API key
        try:
            partner = auth_manager.validate_api_key(api_key)
            return partner
        except Exception as e:
            logger.warning(f"Invalid API key in WebSocket connection: {e}")
            await websocket.close(code=4002, reason="Invalid API key")
            return None

    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=4000, reason="Authentication error")
        return None


# ============================================================================
# WebSocket Endpoints
# ============================================================================


@router.websocket("/stream/{channel}")
async def websocket_stream(websocket: WebSocket, channel: str):
    """
    General WebSocket endpoint for real-time streaming.

    Connect: wss://your-server.com/api/v1/ws/stream/{channel}?api_key=xxx

    Channels:
    - general: General updates
    - consciousness: Consciousness level updates
    - metrics: Real-time metrics
    - agents: Agent status updates
    - quantum: Quantum processing updates
    """
    partner = await authenticate_websocket(websocket)
    if not partner:
        return

    # Start background tasks on first connection
    if not manager._running:
        asyncio.create_task(manager.start_background_tasks())

    connected = await manager.connect(websocket, partner, channel)
    if not connected:
        return

    try:
        # Send initial state based on channel
        if channel == "quantum":
            await manager.send_personal_message(
                {
                    "type": "initial_state",
                    "channel": channel,
                    "data": manager.quantum_state.to_dict(),
                },
                websocket,
            )
        elif channel == "consciousness":
            await manager.send_personal_message(
                {
                    "type": "initial_state",
                    "channel": channel,
                    "data": manager.consciousness_state.to_dict(),
                },
                websocket,
            )
        elif channel == "agents":
            await manager.send_personal_message(
                {
                    "type": "initial_state",
                    "channel": channel,
                    "data": {
                        "agents": [
                            agent.to_dict() for agent in manager.agents.values()
                        ],
                    },
                },
                websocket,
            )
        elif channel == "metrics":
            await manager.send_personal_message(
                {
                    "type": "initial_state",
                    "channel": channel,
                    "data": manager.metrics_state.to_dict(),
                },
                websocket,
            )

        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                msg_type = message.get("type", "")

                if msg_type == WSEventType.PING.value:
                    await manager.update_ping(websocket)
                    await manager.send_personal_message(
                        {
                            "type": WSEventType.PONG.value,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        websocket,
                    )

                elif msg_type == WSEventType.SUBSCRIBE.value:
                    event_type = message.get("event_type", "")
                    await manager.subscribe(websocket, event_type)
                    await manager.send_personal_message(
                        {
                            "type": "subscribed",
                            "event_type": event_type,
                            "message": f"Subscribed to {event_type}",
                        },
                        websocket,
                    )

                elif msg_type == WSEventType.UNSUBSCRIBE.value:
                    event_type = message.get("event_type", "")
                    await manager.unsubscribe(websocket, event_type)
                    await manager.send_personal_message(
                        {
                            "type": "unsubscribed",
                            "event_type": event_type,
                            "message": f"Unsubscribed from {event_type}",
                        },
                        websocket,
                    )

                elif msg_type == "get_state":
                    state_type = message.get("state_type", "all")
                    response = {"type": "state_response", "data": {}}

                    if state_type in ["all", "quantum"]:
                        response["data"]["quantum"] = manager.quantum_state.to_dict()
                    if state_type in ["all", "consciousness"]:
                        response["data"]["consciousness"] = (
                            manager.consciousness_state.to_dict()
                        )
                    if state_type in ["all", "agents"]:
                        response["data"]["agents"] = [
                            a.to_dict() for a in manager.agents.values()
                        ]
                    if state_type in ["all", "metrics"]:
                        response["data"]["metrics"] = manager.metrics_state.to_dict()

                    await manager.send_personal_message(response, websocket)

                else:
                    # Echo for testing
                    await manager.send_personal_message(
                        {
                            "type": "echo",
                            "original": message,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        websocket,
                    )

            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {
                        "type": WSEventType.ERROR.value,
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error in {channel}: {e}")
        manager.disconnect(websocket, channel)


@router.websocket("/consciousness")
async def websocket_consciousness(websocket: WebSocket):
    """
    Dedicated WebSocket for consciousness level updates.
    """
    partner = await authenticate_websocket(websocket)
    if not partner:
        return

    if not manager._running:
        asyncio.create_task(manager.start_background_tasks())

    connected = await manager.connect(websocket, partner, "consciousness")
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == WSEventType.PING.value:
                await manager.update_ping(websocket)
                await manager.send_personal_message(
                    {
                        "type": WSEventType.PONG.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, "consciousness")
    except Exception as e:
        logger.error(f"Consciousness WebSocket error: {e}")
        manager.disconnect(websocket, "consciousness")


@router.websocket("/quantum")
async def websocket_quantum(websocket: WebSocket):
    """
    Dedicated WebSocket for quantum neuron state updates.
    """
    partner = await authenticate_websocket(websocket)
    if not partner:
        return

    if not manager._running:
        asyncio.create_task(manager.start_background_tasks())

    connected = await manager.connect(websocket, partner, "quantum")
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == WSEventType.PING.value:
                await manager.update_ping(websocket)
                await manager.send_personal_message(
                    {
                        "type": WSEventType.PONG.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )
            elif message.get("type") == "get_quantum_state":
                await manager.send_personal_message(
                    {
                        "type": WSEventType.QUANTUM_STATE_UPDATE.value,
                        "data": manager.quantum_state.to_dict(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, "quantum")
    except Exception as e:
        logger.error(f"Quantum WebSocket error: {e}")
        manager.disconnect(websocket, "quantum")


@router.websocket("/agents")
async def websocket_agents(websocket: WebSocket):
    """
    Dedicated WebSocket for agent status updates.
    """
    partner = await authenticate_websocket(websocket)
    if not partner:
        return

    if not manager._running:
        asyncio.create_task(manager.start_background_tasks())

    connected = await manager.connect(websocket, partner, "agents")
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == WSEventType.PING.value:
                await manager.update_ping(websocket)
                await manager.send_personal_message(
                    {
                        "type": WSEventType.PONG.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )
            elif message.get("action") == "status":
                agent_id = message.get("agent_id")
                if agent_id and agent_id in manager.agents:
                    await manager.send_personal_message(
                        {
                            "type": WSEventType.AGENT_STATUS_UPDATE.value,
                            "data": manager.agents[agent_id].to_dict(),
                        },
                        websocket,
                    )

    except WebSocketDisconnect:
        manager.disconnect(websocket, "agents")
    except Exception as e:
        logger.error(f"Agents WebSocket error: {e}")
        manager.disconnect(websocket, "agents")


@router.websocket("/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    Dedicated WebSocket for real-time metrics streaming.
    """
    partner = await authenticate_websocket(websocket)
    if not partner:
        return

    if not manager._running:
        asyncio.create_task(manager.start_background_tasks())

    connected = await manager.connect(websocket, partner, "metrics")
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == WSEventType.PING.value:
                await manager.update_ping(websocket)
                await manager.send_personal_message(
                    {
                        "type": WSEventType.PONG.value,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, "metrics")
    except Exception as e:
        logger.error(f"Metrics WebSocket error: {e}")
        manager.disconnect(websocket, "metrics")


@router.get("/connections")
async def get_active_connections():
    """Get count of active WebSocket connections."""
    total = sum(len(connections) for connections in manager.active_connections.values())
    return {
        "total_connections": total,
        "channels": {
            channel: len(connections)
            for channel, connections in manager.active_connections.items()
        },
        "quantum_state": manager.quantum_state.to_dict(),
        "consciousness_state": manager.consciousness_state.to_dict(),
        "metrics": manager.metrics_state.to_dict(),
    }


@router.post("/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    channel: str = "all",
    api_key: str = Depends(APIKeyHeader(name="X-API-Key")),
):
    """Broadcast a message to all connected clients (admin only)."""
    try:
        partner = auth_manager.validate_api_key(api_key)

        # Check admin permission
        if "admin" not in partner.permissions and "*" not in partner.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permission required",
            )

        if channel == "all":
            await manager.broadcast_to_all(message)
        else:
            await manager.broadcast(channel, message)

        return {"success": True, "message": f"Broadcast sent to {channel}"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Broadcast failed: {str(e)}",
        )


@router.on_event("shutdown")
async def shutdown_websocket_manager():
    """Clean up WebSocket connections on shutdown."""
    await manager.stop_background_tasks()
