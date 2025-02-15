"""Multi-Server Private Information Retrieval.

Implements information-theoretic PIR using multiple non-colluding servers.
Based on the Chor-Goldreich-Kushilevitz-Sudan (CGKS) scheme.

Key idea:
- Client secret-shares the selection vector among k servers
- Each server computes XOR of selected database items
- Client XORs all responses to recover the item

Security: Information-theoretic (secure even against unbounded adversaries)
Assumption: At least one server is honest (non-colluding)

Trade-off:
- Single-server PIR: No trust assumptions, but computationally expensive
- Multi-server PIR: Efficient, but requires trust in server non-collusion
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import secrets
import time
import hashlib

from ragshield.pir.base import (
    PIRClient,
    PIRServer,
    PIRProtocol,
    PIRParameters,
    PIRQuery,
    PIRResponse,
    PIRResult,
    PIRScheme,
    PIRSecurityLevel,
    xor_bytes,
    pad_to_size,
)


# ============================================================================
# XOR-based Secret Sharing
# ============================================================================

class XORSecretSharing:
    """XOR-based secret sharing for binary data.

    Splits a secret into k shares such that XORing all shares
    recovers the original secret. Any k-1 shares reveal nothing.
    """

    @staticmethod
    def share(secret: bytes, num_shares: int) -> List[bytes]:
        """Split secret into shares.

        Args:
            secret: Secret bytes to share
            num_shares: Number of shares to create

        Returns:
            List of shares
        """
        if num_shares < 2:
            raise ValueError("Need at least 2 shares")

        shares = []

        # Generate k-1 random shares
        for _ in range(num_shares - 1):
            share = secrets.token_bytes(len(secret))
            shares.append(share)

        # Last share is XOR of secret with all other shares
        last_share = secret
        for share in shares:
            last_share = xor_bytes(last_share, share)
        shares.append(last_share)

        return shares

    @staticmethod
    def reconstruct(shares: List[bytes]) -> bytes:
        """Reconstruct secret from shares.

        Args:
            shares: All shares

        Returns:
            Original secret
        """
        if not shares:
            raise ValueError("No shares provided")

        result = shares[0]
        for share in shares[1:]:
            result = xor_bytes(result, share)

        return result


# ============================================================================
# Multi-Server PIR Implementation
# ============================================================================

@dataclass
class SelectionVector:
    """Binary selection vector for PIR query."""
    bits: np.ndarray  # Binary array of length n

    def to_bytes(self) -> bytes:
        """Convert to bytes."""
        # Pack bits into bytes
        n_bytes = (len(self.bits) + 7) // 8
        result = bytearray(n_bytes)

        for i, bit in enumerate(self.bits):
            if bit:
                result[i // 8] |= (1 << (7 - i % 8))

        return bytes(result)

    @classmethod
    def from_bytes(cls, data: bytes, length: int) -> 'SelectionVector':
        """Create from bytes."""
        bits = np.zeros(length, dtype=np.uint8)

        for i in range(min(length, len(data) * 8)):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            if byte_idx < len(data):
                bits[i] = (data[byte_idx] >> bit_idx) & 1

        return cls(bits=bits)

    @classmethod
    def unit_vector(cls, index: int, length: int) -> 'SelectionVector':
        """Create unit vector with 1 at index."""
        bits = np.zeros(length, dtype=np.uint8)
        if 0 <= index < length:
            bits[index] = 1
        return cls(bits=bits)


class MultiServerPIRClient(PIRClient):
    """Client for multi-server XOR-based PIR.

    Splits the selection vector into shares, sends one share
    to each server, and XORs the responses.
    """

    def __init__(self, params: PIRParameters):
        """Initialize client.

        Args:
            params: PIR parameters (num_servers must be >= 2)
        """
        super().__init__(params)
        if params.num_servers < 2:
            raise ValueError("Multi-server PIR requires at least 2 servers")

        self.num_servers = params.num_servers

    def setup(self) -> None:
        """No setup needed for XOR-based PIR."""
        pass

    def generate_query(self, index: int) -> PIRQuery:
        """Generate query shares for all servers.

        Creates a unit selection vector and splits it into shares.

        Args:
            index: Index to retrieve

        Returns:
            PIRQuery containing all server shares
        """
        if index < 0 or index >= self.params.database_size:
            raise ValueError(f"Index {index} out of range")

        # Create unit selection vector
        selection = SelectionVector.unit_vector(index, self.params.database_size)
        selection_bytes = selection.to_bytes()

        # Split into shares
        shares = XORSecretSharing.share(selection_bytes, self.num_servers)

        self._query_count += 1

        return PIRQuery(
            encoded_query={
                "shares": shares,
                "target_index": index,
            },
            metadata={
                "num_servers": self.num_servers,
                "database_size": self.params.database_size,
            },
        )

    def decode_response(
        self,
        query: PIRQuery,
        responses: List[PIRResponse],
    ) -> PIRResult:
        """Decode responses by XORing all server responses.

        Args:
            query: Original query
            responses: Responses from all servers

        Returns:
            Retrieved item
        """
        if len(responses) != self.num_servers:
            raise ValueError(f"Expected {self.num_servers} responses, got {len(responses)}")

        # XOR all responses
        response_bytes = [r.encoded_response for r in responses]
        result_bytes = XORSecretSharing.reconstruct(response_bytes)

        # Convert bytes to item
        item = int.from_bytes(result_bytes[:8], 'big')

        total_time = sum(r.computation_time for r in responses)

        return PIRResult(
            item=item,
            index=query.encoded_query["target_index"],
            success=True,
            total_time=total_time,
            communication_cost=sum(len(r.encoded_response) for r in responses),
            metadata={
                "num_servers": len(responses),
            },
        )


class MultiServerPIRServer(PIRServer):
    """Server for multi-server XOR-based PIR.

    Each server holds a copy of the database and processes
    its share of the query.
    """

    def __init__(self, server_id: str, item_size: int = 64):
        """Initialize server.

        Args:
            server_id: Unique server identifier
            item_size: Size of each item in bytes
        """
        super().__init__(server_id)
        self.item_size = item_size
        self._database_bytes: Optional[List[bytes]] = None

    def setup(self, database: List[Any]) -> None:
        """Setup server with database.

        Args:
            database: Database items
        """
        self._database = database
        self._database_bytes = []

        for item in database:
            if isinstance(item, bytes):
                item_bytes = pad_to_size(item, self.item_size)
            elif isinstance(item, (int, np.integer)):
                item_bytes = int(item).to_bytes(self.item_size, 'big', signed=False)
            elif isinstance(item, np.ndarray):
                item_bytes = pad_to_size(item.tobytes(), self.item_size)
            else:
                # Hash for other types
                h = hashlib.sha256(str(item).encode()).digest()
                item_bytes = pad_to_size(h, self.item_size)

            self._database_bytes.append(item_bytes)

    def process_query(self, query: PIRQuery) -> PIRResponse:
        """Process query share and return XOR of selected items.

        For selection share s, compute XOR of all items where s[i] = 1.

        Args:
            query: Query share for this server

        Returns:
            XOR of selected items
        """
        start_time = time.time()

        share = query.encoded_query
        n = len(self._database_bytes)

        # Parse selection share
        selection = SelectionVector.from_bytes(share, n)

        # XOR items where selection bit is 1
        result = bytes(self.item_size)
        for i, bit in enumerate(selection.bits):
            if bit and i < len(self._database_bytes):
                result = xor_bytes(result, self._database_bytes[i])

        self._query_count += 1

        return PIRResponse(
            encoded_response=result,
            server_id=self.server_id,
            computation_time=time.time() - start_time,
            metadata={
                "selected_count": int(np.sum(selection.bits)),
            },
        )


class MultiServerPIR(PIRProtocol):
    """Complete multi-server PIR protocol.

    Uses k servers (default 2) that must not collude.
    Provides information-theoretic security.

    Example:
        >>> database = [100, 200, 300, 400, 500]
        >>> pir = MultiServerPIR(database, num_servers=2)
        >>> result = pir.retrieve(2)
        >>> print(result.item)  # Should be 300
    """

    def __init__(
        self,
        database: Optional[List[Any]] = None,
        num_servers: int = 2,
        item_size: int = 64,
    ):
        """Initialize multi-server PIR.

        Args:
            database: Optional database to setup
            num_servers: Number of servers (default 2)
            item_size: Size of each item in bytes
        """
        params = PIRParameters(
            database_size=len(database) if database else 0,
            item_size=item_size,
            scheme=PIRScheme.MULTI_SERVER_XOR,
            num_servers=num_servers,
        )
        super().__init__(params)

        self.item_size = item_size
        self._client = MultiServerPIRClient(params)
        self._servers = [
            MultiServerPIRServer(f"server_{i}", item_size)
            for i in range(num_servers)
        ]

        if database:
            self.setup(database)

    def setup(self, database: List[Any]) -> None:
        """Setup all servers with the database.

        Args:
            database: The database (replicated to all servers)
        """
        self.params.database_size = len(database)
        self._client.params.database_size = len(database)

        # Each server gets a copy of the database
        for server in self._servers:
            server.setup(database)

        self._client.setup()

    def retrieve(self, index: int) -> PIRResult:
        """Retrieve an item by index.

        No single server learns which index was queried.

        Args:
            index: Index to retrieve

        Returns:
            PIRResult with the item
        """
        start_time = time.time()

        # Client generates query shares
        query = self._client.generate_query(index)

        # Send each share to its server
        responses = []
        for i, server in enumerate(self._servers):
            server_query = PIRQuery(
                encoded_query=query.encoded_query["shares"][i],
                query_id=query.query_id,
            )
            response = server.process_query(server_query)
            responses.append(response)

        # Decode combined response
        result = self._client.decode_response(query, responses)
        result.total_time = time.time() - start_time

        return result

    @property
    def security_level(self) -> PIRSecurityLevel:
        """Get security level of this protocol."""
        return PIRSecurityLevel.INFORMATION_THEORETIC


# ============================================================================
# Shamir Secret Sharing based PIR (threshold security)
# ============================================================================

class ShamirSecretSharing:
    """Shamir's (t, n) secret sharing over a finite field.

    Allows reconstruction with any t shares out of n.
    More flexible than XOR but computationally more expensive.
    """

    # Use a prime field for simplicity
    PRIME = 2**61 - 1  # A Mersenne prime

    @classmethod
    def share(
        cls,
        secret: int,
        num_shares: int,
        threshold: int,
    ) -> List[Tuple[int, int]]:
        """Split secret using Shamir's scheme.

        Args:
            secret: Secret integer to share
            num_shares: Number of shares to create
            threshold: Minimum shares needed to reconstruct

        Returns:
            List of (x, y) share pairs
        """
        if threshold > num_shares:
            raise ValueError("Threshold cannot exceed num_shares")

        secret = secret % cls.PRIME

        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
        coefficients = [secret]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(cls.PRIME))

        # Evaluate polynomial at points 1, 2, ..., n
        shares = []
        for x in range(1, num_shares + 1):
            y = cls._evaluate_polynomial(coefficients, x)
            shares.append((x, y))

        return shares

    @classmethod
    def reconstruct(cls, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret using Lagrange interpolation.

        Args:
            shares: List of (x, y) share pairs

        Returns:
            Original secret
        """
        if not shares:
            raise ValueError("No shares provided")

        # Lagrange interpolation at x=0
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % cls.PRIME
                    denominator = (denominator * (xi - xj)) % cls.PRIME

            # Modular inverse of denominator
            lagrange = (yi * numerator * pow(denominator, cls.PRIME - 2, cls.PRIME)) % cls.PRIME
            secret = (secret + lagrange) % cls.PRIME

        return secret

    @classmethod
    def _evaluate_polynomial(cls, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at x using Horner's method."""
        result = 0
        for coef in reversed(coefficients):
            result = (result * x + coef) % cls.PRIME
        return result


class ThresholdPIR(PIRProtocol):
    """Threshold PIR using Shamir secret sharing.

    Requires t out of n servers to respond correctly.
    Tolerates up to n-t server failures or corruptions.
    """

    def __init__(
        self,
        database: Optional[List[Any]] = None,
        num_servers: int = 3,
        threshold: int = 2,
    ):
        """Initialize threshold PIR.

        Args:
            database: Optional database
            num_servers: Total number of servers
            threshold: Minimum servers needed
        """
        if threshold > num_servers:
            raise ValueError("Threshold cannot exceed num_servers")

        params = PIRParameters(
            database_size=len(database) if database else 0,
            item_size=8,
            scheme=PIRScheme.MULTI_SERVER_SHAMIR,
            num_servers=num_servers,
        )
        super().__init__(params)

        self.threshold = threshold
        self._database: Optional[List[int]] = None

        if database:
            self.setup(database)

    def setup(self, database: List[Any]) -> None:
        """Setup with database.

        Args:
            database: Database items (converted to integers)
        """
        self._database = []
        for item in database:
            if isinstance(item, (int, np.integer)):
                self._database.append(int(item) % ShamirSecretSharing.PRIME)
            else:
                h = int(hashlib.sha256(str(item).encode()).hexdigest()[:15], 16)
                self._database.append(h % ShamirSecretSharing.PRIME)

        self.params.database_size = len(self._database)

    def retrieve(self, index: int) -> PIRResult:
        """Retrieve item using threshold secret sharing.

        Args:
            index: Index to retrieve

        Returns:
            PIRResult with the item
        """
        start_time = time.time()

        if self._database is None:
            raise RuntimeError("Database not setup")

        if index < 0 or index >= len(self._database):
            raise ValueError(f"Index {index} out of range")

        # For each position, create Shamir shares of selection bit
        # This is a simplified version - full implementation would
        # do this more efficiently

        # Create selection vector
        selection = [1 if i == index else 0 for i in range(len(self._database))]

        # Share each selection bit
        all_shares = []
        for bit in selection:
            shares = ShamirSecretSharing.share(bit, self.params.num_servers, self.threshold)
            all_shares.append(shares)

        # Each server computes weighted sum using its shares
        server_results = []
        for server_idx in range(self.params.num_servers):
            result = 0
            x_coord = server_idx + 1  # Share x-coordinate

            for i, db_item in enumerate(self._database):
                # Get this server's share of selection bit i
                _, selection_share = all_shares[i][server_idx]
                result = (result + selection_share * db_item) % ShamirSecretSharing.PRIME

            server_results.append((x_coord, result))

        # Reconstruct using threshold shares
        item = ShamirSecretSharing.reconstruct(server_results[:self.threshold])

        return PIRResult(
            item=item,
            index=index,
            success=True,
            total_time=time.time() - start_time,
            metadata={
                "num_servers": self.params.num_servers,
                "threshold": self.threshold,
            },
        )
