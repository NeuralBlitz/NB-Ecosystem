"""
ID Generation and Hashing Utilities Module
NeuralBlitz Core - ID and Hash Generation Helpers

Provides:
- Unique ID generation
- Hash computation
- Timestamp-based IDs
- Random generation
"""

import hashlib
import uuid
import secrets
from typing import Optional
from datetime import datetime


def generate_uuid() -> str:
    """Generate a UUID4 string.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def generate_short_uuid(length: int = 16) -> str:
    """Generate a shortened UUID.

    Args:
        length: Length of the ID

    Returns:
        Shortened UUID string
    """
    return uuid.uuid4().hex[:length]


def generate_timestamp_id(prefix: str = "", suffix: str = "") -> str:
    """Generate ID based on current timestamp.

    Args:
        prefix: Optional prefix
        suffix: Optional suffix

    Returns:
        Timestamp-based ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}{timestamp}{suffix}"


def generate_hash_id(
    data: str, algorithm: str = "sha256", length: Optional[int] = 16, prefix: str = ""
) -> str:
    """Generate hash-based ID from data.

    Args:
        data: Input data to hash
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
        length: Length of hash to use (None for full hash)
        prefix: Optional prefix

    Returns:
        Hash-based ID
    """
    if algorithm == "sha256":
        hash_obj = hashlib.sha256(data.encode())
    elif algorithm == "sha512":
        hash_obj = hashlib.sha512(data.encode())
    elif algorithm == "md5":
        hash_obj = hashlib.md5(data.encode())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hash_str = hash_obj.hexdigest()
    if length:
        hash_str = hash_str[:length]

    return f"{prefix}{hash_str}"


def generate_secure_token(length: int = 32, prefix: str = "nb_") -> str:
    """Generate cryptographically secure random token.

    Args:
        length: Length of token (in hex characters)
        prefix: Token prefix

    Returns:
        Secure token string
    """
    return f"{prefix}{secrets.token_hex(length)}"


def generate_api_key() -> str:
    """Generate API key for partners.

    Returns:
        API key string
    """
    return generate_secure_token(32, "nb_")


def compute_sha256(data: str) -> str:
    """Compute SHA256 hash of data.

    Args:
        data: Data to hash

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(data.encode()).hexdigest()


def compute_sha512(data: str) -> str:
    """Compute SHA512 hash of data.

    Args:
        data: Data to hash

    Returns:
        SHA512 hex digest
    """
    return hashlib.sha512(data.encode()).hexdigest()


def compute_content_hash(
    content: str, timestamp: Optional[float] = None, extra_fields: Optional[dict] = None
) -> str:
    """Compute hash for content with optional timestamp.

    Args:
        content: Content to hash
        timestamp: Optional timestamp
        extra_fields: Optional extra fields to include

    Returns:
        Computed hash
    """
    hash_input = content

    if timestamp is not None:
        hash_input = f"{hash_input}:{timestamp}"

    if extra_fields:
        for key, value in sorted(extra_fields.items()):
            hash_input = f"{hash_input}:{key}={value}"

    return compute_sha256(hash_input)


def generate_dag_node_hash(
    index: int,
    timestamp: float,
    operation: str,
    actor_id: str,
    payload_hash: str,
    parent_hash: str,
) -> str:
    """Compute hash for DAG node.

    Args:
        index: Node index
        timestamp: Node timestamp
        operation: Operation string
        actor_id: Actor ID
        payload_hash: Payload hash
        parent_hash: Parent hash

    Returns:
        Computed node hash
    """
    content = f"{index}:{timestamp}:{operation}:{actor_id}:{payload_hash}:{parent_hash}"
    return compute_sha256(content)


def generate_proof_id(
    theorem_name: str, length: int = 8, prefix: str = "VPROOF#"
) -> str:
    """Generate proof ID from theorem name.

    Args:
        theorem_name: Theorem name
        length: Hash length
        prefix: ID prefix

    Returns:
        Proof ID
    """
    return f"{prefix}{generate_hash_id(theorem_name, length=length)}"


def generate_trace_id(
    command_verb: str, timestamp: datetime, command_hash: int, prefix: str = "TRC-v20-"
) -> str:
    """Generate trace ID for command.

    Args:
        command_verb: Command verb
        timestamp: Command timestamp
        command_hash: Command hash
        prefix: ID prefix

    Returns:
        Trace ID
    """
    hash_input = f"{command_verb}:{timestamp.isoformat()}:{command_hash}"
    return f"{prefix}{generate_hash_id(hash_input, length=16)}"


def generate_sandbox_id(prefix: str = "SBX-QEC-") -> str:
    """Generate sandbox ID with timestamp.

    Args:
        prefix: ID prefix

    Returns:
        Sandbox ID
    """
    timestamp_str = str(datetime.now())
    return f"{prefix}{generate_hash_id(timestamp_str, length=12)}"


def generate_correlate_ref(
    timestamp: datetime, sandbox_id: str, length: int = 16
) -> str:
    """Generate correlate reference ID.

    Args:
        timestamp: Correlation timestamp
        sandbox_id: Sandbox ID
        length: Hash length

    Returns:
        Correlate reference ID
    """
    content = f"{timestamp.isoformat()}:{sandbox_id}"
    return generate_hash_id(content, length=length)


def generate_braid_id(seed: str, length: int = 16) -> str:
    """Generate braid ID.

    Args:
        seed: Seed string
        length: ID length

    Returns:
        Braid ID
    """
    return generate_hash_id(seed, length=length)


def generate_transfer_id(
    braid_id: str, destination: str, timestamp: Optional[datetime] = None
) -> str:
    """Generate transfer ID.

    Args:
        braid_id: Braid ID
        destination: Destination instance
        timestamp: Optional timestamp

    Returns:
        Transfer ID
    """
    ts = timestamp or datetime.now()
    content = f"{braid_id}:{destination}:{ts}"
    return generate_hash_id(content, length=16)


# Compatibility aliases
generate_id = generate_uuid
hash_data = compute_sha256
hash_sha512 = compute_sha512
