"""Private Information Retrieval (PIR) Base Classes.

PIR allows a client to retrieve an item from a database held by a server,
without the server learning which item was retrieved.

This module provides:
- Abstract base classes for PIR protocols
- Common data structures for PIR operations
- Utility functions for PIR computations

Key concepts:
- Query privacy: Server cannot learn which item is being queried
- Computational PIR: Security based on computational assumptions
- Information-theoretic PIR: Security even against unbounded adversaries
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Generic, TypeVar
from enum import Enum
import numpy as np
import hashlib
import time


class PIRScheme(Enum):
    """Types of PIR schemes."""
    SINGLE_SERVER_HE = "single_server_he"  # Homomorphic encryption based
    SINGLE_SERVER_OT = "single_server_ot"  # Oblivious transfer based
    MULTI_SERVER_XOR = "multi_server_xor"  # XOR-based multi-server
    MULTI_SERVER_SHAMIR = "multi_server_shamir"  # Shamir secret sharing
    HYBRID = "hybrid"  # Combination of schemes


class PIRSecurityLevel(Enum):
    """Security levels for PIR protocols."""
    COMPUTATIONAL = "computational"  # Secure against polynomial-time adversaries
    INFORMATION_THEORETIC = "information_theoretic"  # Secure against unbounded adversaries
    STATISTICAL = "statistical"  # Statistically close to information-theoretic


@dataclass
class PIRParameters:
    """Parameters for PIR protocol.

    Attributes:
        database_size: Number of items in the database
        item_size: Size of each item in bytes
        security_parameter: Security parameter (e.g., key size)
        num_servers: Number of servers (for multi-server PIR)
        scheme: PIR scheme to use
    """
    database_size: int
    item_size: int
    security_parameter: int = 128
    num_servers: int = 1
    scheme: PIRScheme = PIRScheme.SINGLE_SERVER_HE


@dataclass
class PIRQuery:
    """Encrypted/encoded PIR query.

    The query hides which index the client wants to retrieve.

    Attributes:
        encoded_query: The encoded/encrypted query data
        query_id: Unique identifier for this query
        metadata: Additional query metadata
    """
    encoded_query: Any  # Could be encrypted vector, shares, etc.
    query_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.query_id:
            self.query_id = hashlib.sha256(
                str(time.time()).encode()
            ).hexdigest()[:16]


@dataclass
class PIRResponse:
    """Server's response to a PIR query.

    Attributes:
        encoded_response: The encoded/encrypted response
        server_id: ID of the responding server
        computation_time: Time taken to compute response
        metadata: Additional response metadata
    """
    encoded_response: Any
    server_id: str = "server_0"
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIRResult:
    """Final decoded result of PIR retrieval.

    Attributes:
        item: The retrieved item
        index: The index that was queried
        success: Whether retrieval was successful
        total_time: Total time for the PIR operation
        communication_cost: Total bytes transferred
        metadata: Additional result metadata
    """
    item: Any
    index: int
    success: bool = True
    total_time: float = 0.0
    communication_cost: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIRClient(ABC):
    """Abstract base class for PIR clients.

    The client generates queries that hide which item is being requested,
    and decodes the server's response to obtain the desired item.
    """

    def __init__(self, params: PIRParameters):
        """Initialize PIR client.

        Args:
            params: PIR protocol parameters
        """
        self.params = params
        self._query_count = 0

    @abstractmethod
    def setup(self) -> None:
        """Perform any necessary setup/key generation."""
        pass

    @abstractmethod
    def generate_query(self, index: int) -> PIRQuery:
        """Generate a PIR query for the given index.

        Args:
            index: The index of the item to retrieve (0 to n-1)

        Returns:
            PIRQuery that hides the requested index
        """
        pass

    @abstractmethod
    def decode_response(
        self,
        query: PIRQuery,
        responses: List[PIRResponse],
    ) -> PIRResult:
        """Decode server response(s) to obtain the item.

        Args:
            query: The original query
            responses: Response(s) from server(s)

        Returns:
            The retrieved item
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "query_count": self._query_count,
            "scheme": self.params.scheme.value,
            "database_size": self.params.database_size,
        }


class PIRServer(ABC):
    """Abstract base class for PIR servers.

    The server processes PIR queries without learning which item
    is being retrieved.
    """

    def __init__(self, server_id: str = "server_0"):
        """Initialize PIR server.

        Args:
            server_id: Unique identifier for this server
        """
        self.server_id = server_id
        self._database: Optional[List[Any]] = None
        self._query_count = 0

    @abstractmethod
    def setup(self, database: List[Any]) -> None:
        """Setup the server with a database.

        Args:
            database: List of items to serve
        """
        pass

    @abstractmethod
    def process_query(self, query: PIRQuery) -> PIRResponse:
        """Process a PIR query and generate response.

        Args:
            query: The PIR query from client

        Returns:
            PIRResponse that encodes the answer
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "server_id": self.server_id,
            "query_count": self._query_count,
            "database_size": len(self._database) if self._database else 0,
        }


class PIRProtocol(ABC):
    """Abstract base class for complete PIR protocols.

    Combines client and server(s) into a complete protocol.
    """

    def __init__(self, params: PIRParameters):
        """Initialize PIR protocol.

        Args:
            params: Protocol parameters
        """
        self.params = params
        self._client: Optional[PIRClient] = None
        self._servers: List[PIRServer] = []

    @abstractmethod
    def setup(self, database: List[Any]) -> None:
        """Setup the complete protocol.

        Args:
            database: The database to serve
        """
        pass

    @abstractmethod
    def retrieve(self, index: int) -> PIRResult:
        """Retrieve an item by index.

        Args:
            index: Index of item to retrieve

        Returns:
            The retrieved item
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            "params": {
                "scheme": self.params.scheme.value,
                "database_size": self.params.database_size,
                "num_servers": self.params.num_servers,
            }
        }

        if self._client:
            stats["client"] = self._client.get_stats()

        if self._servers:
            stats["servers"] = [s.get_stats() for s in self._servers]

        return stats


# Utility functions

def bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to list of bits.

    Args:
        data: Bytes to convert

    Returns:
        List of bits (0 or 1)
    """
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert list of bits to bytes.

    Args:
        bits: List of bits

    Returns:
        Bytes representation
    """
    # Pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append(0)

    result = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return bytes(result)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings.

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        XOR of a and b
    """
    return bytes(x ^ y for x, y in zip(a, b))


def serialize_array(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes.

    Args:
        arr: Array to serialize

    Returns:
        Bytes representation
    """
    return arr.tobytes()


def deserialize_array(data: bytes, dtype=np.float32, shape=None) -> np.ndarray:
    """Deserialize bytes to numpy array.

    Args:
        data: Bytes to deserialize
        dtype: Array dtype
        shape: Optional shape to reshape to

    Returns:
        Numpy array
    """
    arr = np.frombuffer(data, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def pad_to_size(data: bytes, size: int) -> bytes:
    """Pad data to specified size.

    Args:
        data: Data to pad
        size: Target size

    Returns:
        Padded data
    """
    if len(data) >= size:
        return data[:size]
    return data + b'\x00' * (size - len(data))


def generate_random_mask(size: int) -> bytes:
    """Generate random mask for encryption.

    Args:
        size: Size of mask in bytes

    Returns:
        Random bytes
    """
    return np.random.bytes(size)
