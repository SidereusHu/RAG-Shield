"""Single-Server Private Information Retrieval.

Implements computational PIR using additive homomorphic encryption.
The server cannot learn which item the client is requesting.

This implementation uses a simplified additive homomorphic scheme
for educational purposes. Production systems should use established
libraries like SEAL, HElib, or Paillier.

Key idea:
- Client encrypts a selection vector [0, 0, ..., 1, ..., 0]
- Server computes encrypted dot product with database
- Client decrypts to get the selected item

Security: Computational (based on hardness of discrete log in the
simplified scheme, or lattice problems in production schemes).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import hashlib
import secrets
import time

from ragshield.pir.base import (
    PIRClient,
    PIRServer,
    PIRProtocol,
    PIRParameters,
    PIRQuery,
    PIRResponse,
    PIRResult,
    PIRScheme,
    serialize_array,
    deserialize_array,
)


# ============================================================================
# Simplified Additive Homomorphic Encryption
# ============================================================================

@dataclass
class AdditiveHEPublicKey:
    """Public key for additive homomorphic encryption.

    This is a simplified scheme for demonstration.
    Production should use Paillier or lattice-based schemes.
    """
    n: int  # Modulus
    g: int  # Generator (for simplified scheme, g = n + 1)


@dataclass
class AdditiveHESecretKey:
    """Secret key for additive homomorphic encryption."""
    lambda_val: int  # Carmichael's lambda
    mu: int  # Modular inverse


@dataclass
class AdditiveHECiphertext:
    """Ciphertext in additive HE scheme."""
    value: int
    modulus: int

    def __add__(self, other: 'AdditiveHECiphertext') -> 'AdditiveHECiphertext':
        """Homomorphic addition of ciphertexts."""
        if self.modulus != other.modulus:
            raise ValueError("Modulus mismatch")
        return AdditiveHECiphertext(
            value=(self.value * other.value) % (self.modulus ** 2),
            modulus=self.modulus,
        )

    def scalar_mult(self, scalar: int) -> 'AdditiveHECiphertext':
        """Homomorphic scalar multiplication."""
        return AdditiveHECiphertext(
            value=pow(self.value, scalar, self.modulus ** 2),
            modulus=self.modulus,
        )


class SimplifiedPaillier:
    """Simplified Paillier-like additive homomorphic encryption.

    Properties:
    - Enc(m1) * Enc(m2) = Enc(m1 + m2) (additive homomorphic)
    - Enc(m)^k = Enc(k * m) (scalar multiplication)

    Note: This is a simplified implementation for educational purposes.
    Use a proper cryptographic library for production.
    """

    def __init__(self, key_bits: int = 512):
        """Initialize the encryption scheme.

        Args:
            key_bits: Bit length for key generation
        """
        self.key_bits = key_bits
        self.public_key: Optional[AdditiveHEPublicKey] = None
        self.secret_key: Optional[AdditiveHESecretKey] = None

    def generate_keys(self) -> Tuple[AdditiveHEPublicKey, AdditiveHESecretKey]:
        """Generate public and secret keys.

        For simplicity, we use a deterministic small modulus.
        Real Paillier uses large prime products.
        """
        # For demonstration, use smaller parameters
        # In production, p and q should be large primes
        p = self._generate_prime(self.key_bits // 2)
        q = self._generate_prime(self.key_bits // 2)

        n = p * q
        lambda_val = (p - 1) * (q - 1) // self._gcd(p - 1, q - 1)

        # g = n + 1 is a standard choice
        g = n + 1

        # mu = L(g^lambda mod n^2)^(-1) mod n
        # where L(x) = (x - 1) / n
        g_lambda = pow(g, lambda_val, n * n)
        l_val = (g_lambda - 1) // n
        mu = self._mod_inverse(l_val, n)

        self.public_key = AdditiveHEPublicKey(n=n, g=g)
        self.secret_key = AdditiveHESecretKey(lambda_val=lambda_val, mu=mu)

        return self.public_key, self.secret_key

    def encrypt(self, plaintext: int, public_key: AdditiveHEPublicKey) -> AdditiveHECiphertext:
        """Encrypt a plaintext integer.

        Args:
            plaintext: Integer to encrypt (must be < n)
            public_key: Public key

        Returns:
            Ciphertext
        """
        n = public_key.n
        n_sq = n * n

        # Ensure plaintext is in valid range
        plaintext = plaintext % n

        # Random r in Z*_n
        r = secrets.randbelow(n - 1) + 1
        while self._gcd(r, n) != 1:
            r = secrets.randbelow(n - 1) + 1

        # c = g^m * r^n mod n^2
        g_m = pow(public_key.g, plaintext, n_sq)
        r_n = pow(r, n, n_sq)
        c = (g_m * r_n) % n_sq

        return AdditiveHECiphertext(value=c, modulus=n)

    def decrypt(
        self,
        ciphertext: AdditiveHECiphertext,
        secret_key: AdditiveHESecretKey,
    ) -> int:
        """Decrypt a ciphertext.

        Args:
            ciphertext: Ciphertext to decrypt
            secret_key: Secret key

        Returns:
            Plaintext integer
        """
        n = ciphertext.modulus
        n_sq = n * n

        # m = L(c^lambda mod n^2) * mu mod n
        c_lambda = pow(ciphertext.value, secret_key.lambda_val, n_sq)
        l_val = (c_lambda - 1) // n
        plaintext = (l_val * secret_key.mu) % n

        return plaintext

    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime number.

        For demonstration, we use a simple method.
        Production should use proper prime generation.
        """
        # For small demos, use known primes
        if bits <= 16:
            primes = [65537, 65521, 65519, 65497, 65479]
            return secrets.choice(primes)

        # For larger, generate random odd and test
        while True:
            candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
            if self._is_prime(candidate):
                return candidate

    def _is_prime(self, n: int, k: int = 10) -> bool:
        """Miller-Rabin primality test."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False

        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        # Witness loop
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2
            x = pow(a, d, n)

            if x == 1 or x == n - 1:
                continue

            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False

        return True

    def _gcd(self, a: int, b: int) -> int:
        """Greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

    def _mod_inverse(self, a: int, m: int) -> int:
        """Modular multiplicative inverse using extended Euclidean algorithm."""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m


# ============================================================================
# Single-Server PIR Implementation
# ============================================================================

class SingleServerPIRClient(PIRClient):
    """Client for single-server PIR using homomorphic encryption.

    The client generates an encrypted selection vector where only
    the desired index has value 1 (encrypted), and all others are 0.
    """

    def __init__(self, params: PIRParameters, key_bits: int = 64):
        """Initialize PIR client.

        Args:
            params: PIR parameters
            key_bits: Bit size for HE keys (smaller for demo)
        """
        super().__init__(params)
        self.key_bits = key_bits
        self.he = SimplifiedPaillier(key_bits=key_bits)
        self.public_key: Optional[AdditiveHEPublicKey] = None
        self.secret_key: Optional[AdditiveHESecretKey] = None

    def setup(self) -> AdditiveHEPublicKey:
        """Generate keys and return public key.

        Returns:
            Public key to share with server
        """
        self.public_key, self.secret_key = self.he.generate_keys()
        return self.public_key

    def generate_query(self, index: int) -> PIRQuery:
        """Generate encrypted query for given index.

        Creates a vector [Enc(0), Enc(0), ..., Enc(1), ..., Enc(0)]
        where Enc(1) is at position `index`.

        Args:
            index: Index to retrieve (0 to n-1)

        Returns:
            Encrypted query
        """
        if index < 0 or index >= self.params.database_size:
            raise ValueError(f"Index {index} out of range [0, {self.params.database_size})")

        if self.public_key is None:
            raise RuntimeError("Must call setup() first")

        # Create encrypted selection vector
        encrypted_vector = []
        for i in range(self.params.database_size):
            if i == index:
                ct = self.he.encrypt(1, self.public_key)
            else:
                ct = self.he.encrypt(0, self.public_key)
            encrypted_vector.append(ct)

        self._query_count += 1

        return PIRQuery(
            encoded_query={
                "encrypted_vector": encrypted_vector,
                "public_key": self.public_key,
                "target_index": index,  # Stored locally, NOT sent to server
            },
            metadata={
                "database_size": self.params.database_size,
                "key_bits": self.key_bits,
            },
        )

    def decode_response(
        self,
        query: PIRQuery,
        responses: List[PIRResponse],
    ) -> PIRResult:
        """Decode server response to get the item.

        Args:
            query: Original query
            responses: Server response(s)

        Returns:
            Retrieved item
        """
        if not responses:
            raise ValueError("No response received")

        response = responses[0]
        encrypted_result = response.encoded_response

        # Decrypt the result
        if self.secret_key is None:
            raise RuntimeError("Secret key not available")

        decrypted = self.he.decrypt(encrypted_result, self.secret_key)

        metadata = response.metadata.copy()
        metadata["computation_time"] = response.computation_time

        return PIRResult(
            item=decrypted,
            index=query.encoded_query.get("target_index", -1),
            success=True,
            metadata=metadata,
        )


class SingleServerPIRServer(PIRServer):
    """Server for single-server PIR.

    Processes encrypted queries by computing homomorphic dot product
    of the encrypted selection vector with the database.
    """

    def __init__(self, server_id: str = "server_0"):
        """Initialize PIR server."""
        super().__init__(server_id)
        self._database_values: Optional[List[int]] = None

    def setup(self, database: List[Any]) -> None:
        """Setup server with database.

        Args:
            database: List of items (integers for this implementation)
        """
        # Convert to integers if needed
        self._database = database
        self._database_values = []

        for item in database:
            if isinstance(item, (int, np.integer)):
                self._database_values.append(int(item))
            elif isinstance(item, bytes):
                # Convert bytes to int
                self._database_values.append(int.from_bytes(item[:8], 'big'))
            elif isinstance(item, np.ndarray):
                # Use hash of array
                self._database_values.append(
                    int(hashlib.sha256(item.tobytes()).hexdigest()[:8], 16)
                )
            else:
                # Use hash of string representation
                self._database_values.append(
                    int(hashlib.sha256(str(item).encode()).hexdigest()[:8], 16)
                )

    def process_query(self, query: PIRQuery) -> PIRResponse:
        """Process PIR query homomorphically.

        Computes: sum_i (encrypted_vector[i] * database[i])
        which equals Enc(database[target_index]) due to selection vector.

        Args:
            query: Encrypted PIR query

        Returns:
            Encrypted response
        """
        start_time = time.time()

        encrypted_vector = query.encoded_query["encrypted_vector"]
        public_key = query.encoded_query["public_key"]

        if self._database_values is None:
            raise RuntimeError("Server not setup")

        # Compute homomorphic dot product
        # result = sum_i (ct[i] * db[i])
        # Using: ct^scalar = Enc(plaintext * scalar)
        # And: ct1 * ct2 = Enc(pt1 + pt2)

        result = None
        for i, (ct, db_val) in enumerate(zip(encrypted_vector, self._database_values)):
            # ct^db_val = Enc(selection[i] * db_val)
            term = ct.scalar_mult(db_val % public_key.n)

            if result is None:
                result = term
            else:
                # Homomorphic addition
                result = result + term

        self._query_count += 1
        computation_time = time.time() - start_time

        return PIRResponse(
            encoded_response=result,
            server_id=self.server_id,
            computation_time=computation_time,
            metadata={
                "database_size": len(self._database_values),
            },
        )


class SingleServerPIR(PIRProtocol):
    """Complete single-server PIR protocol.

    Combines client and server into an easy-to-use interface.

    Example:
        >>> database = [100, 200, 300, 400, 500]
        >>> pir = SingleServerPIR(database)
        >>> result = pir.retrieve(2)  # Get item at index 2
        >>> print(result.item)  # Should be 300
    """

    def __init__(
        self,
        database: Optional[List[Any]] = None,
        key_bits: int = 64,
    ):
        """Initialize single-server PIR.

        Args:
            database: Optional database to setup
            key_bits: Key size for homomorphic encryption
        """
        params = PIRParameters(
            database_size=len(database) if database else 0,
            item_size=8,  # Assuming integer items
            scheme=PIRScheme.SINGLE_SERVER_HE,
            num_servers=1,
        )
        super().__init__(params)

        self.key_bits = key_bits
        self._client = SingleServerPIRClient(params, key_bits=key_bits)
        self._server = SingleServerPIRServer()
        self._servers = [self._server]

        if database:
            self.setup(database)

    def setup(self, database: List[Any]) -> None:
        """Setup the PIR protocol with a database.

        Args:
            database: The database to serve
        """
        self.params.database_size = len(database)
        self._client.params.database_size = len(database)

        # Client generates keys
        public_key = self._client.setup()

        # Server sets up database
        self._server.setup(database)

    def retrieve(self, index: int) -> PIRResult:
        """Retrieve an item by index.

        The server will not learn which index was retrieved.

        Args:
            index: Index of item to retrieve

        Returns:
            PIRResult with the retrieved item
        """
        start_time = time.time()

        # Client generates query
        query = self._client.generate_query(index)

        # Remove target_index before sending to server (it's private!)
        server_query = PIRQuery(
            encoded_query={
                "encrypted_vector": query.encoded_query["encrypted_vector"],
                "public_key": query.encoded_query["public_key"],
            },
            query_id=query.query_id,
            metadata=query.metadata,
        )

        # Server processes query
        response = self._server.process_query(server_query)

        # Client decodes response
        result = self._client.decode_response(query, [response])
        result.total_time = time.time() - start_time

        return result


# ============================================================================
# Optimized Single-Server PIR (using matrix representation)
# ============================================================================

class MatrixPIRClient(PIRClient):
    """Optimized PIR client using matrix representation.

    Instead of O(n) ciphertexts, uses O(sqrt(n)) by representing
    the database as a sqrt(n) x sqrt(n) matrix.
    """

    def __init__(self, params: PIRParameters, key_bits: int = 64):
        """Initialize matrix PIR client.

        Args:
            params: PIR parameters
            key_bits: Key size
        """
        super().__init__(params)
        self.key_bits = key_bits
        self.he = SimplifiedPaillier(key_bits=key_bits)
        self.public_key: Optional[AdditiveHEPublicKey] = None
        self.secret_key: Optional[AdditiveHESecretKey] = None

        # Matrix dimensions
        n = params.database_size
        self.sqrt_n = int(np.ceil(np.sqrt(n)))
        self.rows = self.sqrt_n
        self.cols = self.sqrt_n

    def setup(self) -> AdditiveHEPublicKey:
        """Generate keys."""
        self.public_key, self.secret_key = self.he.generate_keys()
        return self.public_key

    def generate_query(self, index: int) -> PIRQuery:
        """Generate query using row selection.

        For index i, compute row = i // sqrt(n), col = i % sqrt(n).
        Send encrypted row selector, server returns encrypted column.

        Args:
            index: Index to retrieve

        Returns:
            Encrypted query (only O(sqrt(n)) ciphertexts)
        """
        if self.public_key is None:
            raise RuntimeError("Must call setup() first")

        row = index // self.cols
        col = index % self.cols

        # Encrypt row selector
        row_selector = []
        for r in range(self.rows):
            if r == row:
                ct = self.he.encrypt(1, self.public_key)
            else:
                ct = self.he.encrypt(0, self.public_key)
            row_selector.append(ct)

        self._query_count += 1

        return PIRQuery(
            encoded_query={
                "row_selector": row_selector,
                "public_key": self.public_key,
                "target_col": col,  # Private, not sent
                "target_index": index,  # Private
            },
            metadata={
                "rows": self.rows,
                "cols": self.cols,
            },
        )

    def decode_response(
        self,
        query: PIRQuery,
        responses: List[PIRResponse],
    ) -> PIRResult:
        """Decode response by selecting the correct column.

        Args:
            query: Original query
            responses: Server response

        Returns:
            Retrieved item
        """
        if not responses:
            raise ValueError("No response")

        response = responses[0]
        encrypted_cols = response.encoded_response

        # Decrypt all columns
        target_col = query.encoded_query["target_col"]

        if self.secret_key is None:
            raise RuntimeError("No secret key")

        # Decrypt the target column
        if target_col < len(encrypted_cols):
            decrypted = self.he.decrypt(encrypted_cols[target_col], self.secret_key)
        else:
            decrypted = 0

        return PIRResult(
            item=decrypted,
            index=query.encoded_query["target_index"],
            success=True,
            computation_time=response.computation_time,
        )


class MatrixPIRServer(PIRServer):
    """Optimized PIR server using matrix representation."""

    def __init__(self, server_id: str = "server_0"):
        """Initialize server."""
        super().__init__(server_id)
        self._matrix: Optional[np.ndarray] = None
        self.rows = 0
        self.cols = 0

    def setup(self, database: List[Any]) -> None:
        """Setup with database as matrix.

        Args:
            database: Database items
        """
        self._database = database
        n = len(database)
        sqrt_n = int(np.ceil(np.sqrt(n)))
        self.rows = sqrt_n
        self.cols = sqrt_n

        # Convert to integer values and reshape to matrix
        values = []
        for item in database:
            if isinstance(item, (int, np.integer)):
                values.append(int(item))
            else:
                values.append(int(hashlib.sha256(str(item).encode()).hexdigest()[:8], 16))

        # Pad to square matrix size
        while len(values) < self.rows * self.cols:
            values.append(0)

        self._matrix = np.array(values).reshape(self.rows, self.cols)

    def process_query(self, query: PIRQuery) -> PIRResponse:
        """Process query to produce encrypted columns.

        Computes encrypted dot product of row selector with each column.

        Args:
            query: Encrypted query with row selector

        Returns:
            Encrypted columns (O(sqrt(n)) ciphertexts)
        """
        start_time = time.time()

        row_selector = query.encoded_query["row_selector"]
        public_key = query.encoded_query["public_key"]

        if self._matrix is None:
            raise RuntimeError("Server not setup")

        # For each column, compute encrypted sum over rows
        encrypted_cols = []

        for col in range(self.cols):
            result = None
            for row in range(self.rows):
                val = int(self._matrix[row, col]) % public_key.n
                term = row_selector[row].scalar_mult(val)

                if result is None:
                    result = term
                else:
                    result = result + term

            encrypted_cols.append(result)

        self._query_count += 1

        return PIRResponse(
            encoded_response=encrypted_cols,
            server_id=self.server_id,
            computation_time=time.time() - start_time,
            metadata={
                "num_cols": len(encrypted_cols),
            },
        )
