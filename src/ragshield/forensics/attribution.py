"""Attack attribution for RAG systems.

Provides attribution analysis to identify attack sources and
link related attacks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib

from ragshield.core.document import Document
from ragshield.forensics.analyzer import AttackPatternAnalyzer, AttackFingerprint
from ragshield.forensics.provenance import ProvenanceTracker


class AttributionConfidence(Enum):
    """Confidence levels for attribution."""

    DEFINITE = "definite"  # >95% confidence
    HIGH = "high"  # 80-95% confidence
    MEDIUM = "medium"  # 50-80% confidence
    LOW = "low"  # 20-50% confidence
    UNCERTAIN = "uncertain"  # <20% confidence


@dataclass
class AttributionSource:
    """A potential source of an attack.

    Attributes:
        source_id: Unique identifier for the source
        source_type: Type of source (ip, user, api_key, etc.)
        identifier: Source identifier value
        confidence: Attribution confidence
        evidence: Supporting evidence
        related_docs: Documents attributed to this source
    """

    source_id: str
    source_type: str
    identifier: str
    confidence: AttributionConfidence
    evidence: List[str] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)


@dataclass
class AttackCampaign:
    """A coordinated attack campaign.

    Attributes:
        campaign_id: Unique campaign identifier
        name: Campaign name/description
        sources: Attributed sources
        documents: Documents in campaign
        fingerprints: Common fingerprints
        tactics: Identified tactics
        confidence: Campaign attribution confidence
    """

    campaign_id: str
    name: str
    sources: List[AttributionSource]
    documents: List[str]
    fingerprints: List[AttackFingerprint]
    tactics: List[str]
    confidence: AttributionConfidence


@dataclass
class AttributionReport:
    """Complete attribution report.

    Attributes:
        doc_id: Document being attributed
        sources: Potential sources
        campaigns: Related campaigns
        similar_attacks: Similar historical attacks
        confidence: Overall attribution confidence
        recommendations: Recommended actions
    """

    doc_id: str
    sources: List[AttributionSource]
    campaigns: List[AttackCampaign]
    similar_attacks: List[Tuple[str, float]]
    confidence: AttributionConfidence
    recommendations: List[str]


class AttackAttributor:
    """Attributes attacks to sources and campaigns.

    Uses fingerprinting, behavioral analysis, and provenance data
    to identify attack origins and link related attacks.
    """

    def __init__(
        self,
        analyzer: AttackPatternAnalyzer,
        provenance_tracker: Optional[ProvenanceTracker] = None,
    ):
        """Initialize attributor.

        Args:
            analyzer: Pattern analyzer for fingerprinting
            provenance_tracker: Optional provenance tracker
        """
        self.analyzer = analyzer
        self.provenance = provenance_tracker

        # Known sources database
        self._known_sources: Dict[str, AttributionSource] = {}

        # Campaign tracking
        self._campaigns: Dict[str, AttackCampaign] = {}

        # Fingerprint to source mapping
        self._fingerprint_sources: Dict[str, str] = {}

    def attribute(self, document: Document) -> AttributionReport:
        """Attribute an attack to its source.

        Args:
            document: Document to attribute

        Returns:
            Attribution report
        """
        # Analyze document
        analysis = self.analyzer.analyze(document)
        fingerprint = analysis.fingerprint

        # Find similar attacks
        similar = self.analyzer.find_similar_attacks(fingerprint, threshold=0.6)

        # Identify potential sources
        sources = self._identify_sources(document, fingerprint, similar)

        # Link to campaigns
        campaigns = self._identify_campaigns(document, fingerprint, sources)

        # Calculate overall confidence
        confidence = self._calculate_confidence(sources, campaigns, similar)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            document, sources, campaigns, confidence
        )

        return AttributionReport(
            doc_id=document.doc_id,
            sources=sources,
            campaigns=campaigns,
            similar_attacks=similar,
            confidence=confidence,
            recommendations=recommendations,
        )

    def register_source(
        self,
        source_type: str,
        identifier: str,
        doc_ids: List[str],
        evidence: Optional[List[str]] = None,
    ) -> AttributionSource:
        """Register a known attack source.

        Args:
            source_type: Type of source
            identifier: Source identifier
            doc_ids: Documents from this source
            evidence: Evidence of attribution

        Returns:
            Created source
        """
        source_id = self._generate_source_id(source_type, identifier)

        source = AttributionSource(
            source_id=source_id,
            source_type=source_type,
            identifier=identifier,
            confidence=AttributionConfidence.DEFINITE,
            evidence=evidence or [],
            related_docs=doc_ids,
        )

        self._known_sources[source_id] = source
        return source

    def create_campaign(
        self,
        name: str,
        doc_ids: List[str],
        sources: Optional[List[AttributionSource]] = None,
        tactics: Optional[List[str]] = None,
    ) -> AttackCampaign:
        """Create an attack campaign.

        Args:
            name: Campaign name
            doc_ids: Documents in campaign
            sources: Known sources
            tactics: Identified tactics

        Returns:
            Created campaign
        """
        campaign_id = self._generate_campaign_id(name, doc_ids)

        # Get fingerprints for documents
        fingerprints = []
        for doc_id in doc_ids:
            if doc_id in self.analyzer._fingerprint_db:
                fingerprints.append(self.analyzer._fingerprint_db[doc_id])

        campaign = AttackCampaign(
            campaign_id=campaign_id,
            name=name,
            sources=sources or [],
            documents=doc_ids,
            fingerprints=fingerprints,
            tactics=tactics or [],
            confidence=AttributionConfidence.HIGH if sources else AttributionConfidence.MEDIUM,
        )

        self._campaigns[campaign_id] = campaign
        return campaign

    def link_documents(
        self, doc_ids: List[str], similarity_threshold: float = 0.7
    ) -> List[List[str]]:
        """Link related documents into groups.

        Args:
            doc_ids: Documents to link
            similarity_threshold: Similarity threshold for linking

        Returns:
            Groups of linked documents
        """
        # Build similarity graph
        similarities: Dict[str, Dict[str, float]] = defaultdict(dict)

        for i, doc_id1 in enumerate(doc_ids):
            fp1 = self.analyzer._fingerprint_db.get(doc_id1)
            if not fp1:
                continue

            for doc_id2 in doc_ids[i + 1 :]:
                fp2 = self.analyzer._fingerprint_db.get(doc_id2)
                if fp2:
                    sim = fp1.similarity(fp2)
                    if sim >= similarity_threshold:
                        similarities[doc_id1][doc_id2] = sim
                        similarities[doc_id2][doc_id1] = sim

        # Connected components
        groups = []
        visited = set()

        for doc_id in doc_ids:
            if doc_id in visited:
                continue

            group = self._dfs_group(doc_id, similarities, visited)
            if group:
                groups.append(list(group))

        return groups

    def find_campaigns_by_source(self, source_id: str) -> List[AttackCampaign]:
        """Find campaigns linked to a source.

        Args:
            source_id: Source identifier

        Returns:
            List of campaigns
        """
        campaigns = []
        for campaign in self._campaigns.values():
            for source in campaign.sources:
                if source.source_id == source_id:
                    campaigns.append(campaign)
                    break
        return campaigns

    def get_source_statistics(self, source_id: str) -> Dict[str, Any]:
        """Get statistics for a source.

        Args:
            source_id: Source identifier

        Returns:
            Statistics dictionary
        """
        source = self._known_sources.get(source_id)
        if not source:
            return {}

        campaigns = self.find_campaigns_by_source(source_id)

        return {
            "source_id": source_id,
            "source_type": source.source_type,
            "document_count": len(source.related_docs),
            "campaign_count": len(campaigns),
            "confidence": source.confidence.value,
            "evidence_count": len(source.evidence),
        }

    def _identify_sources(
        self,
        document: Document,
        fingerprint: AttackFingerprint,
        similar: List[Tuple[str, float]],
    ) -> List[AttributionSource]:
        """Identify potential sources for an attack.

        Args:
            document: Document to attribute
            fingerprint: Document fingerprint
            similar: Similar attacks

        Returns:
            List of potential sources
        """
        sources = []

        # Check provenance for direct source info
        if self.provenance:
            chain = self.provenance.get_chain(document.doc_id)
            if chain and chain.metadata.get("source"):
                source_info = chain.metadata["source"]
                source = AttributionSource(
                    source_id=self._generate_source_id("provenance", source_info),
                    source_type="provenance",
                    identifier=source_info,
                    confidence=AttributionConfidence.HIGH,
                    evidence=["Provenance chain indicates origin"],
                    related_docs=[document.doc_id],
                )
                sources.append(source)

        # Check similar attacks for source hints
        for similar_doc_id, similarity in similar:
            # Look up source of similar document
            for source in self._known_sources.values():
                if similar_doc_id in source.related_docs:
                    # Same source likely
                    confidence = self._similarity_to_confidence(similarity)
                    new_source = AttributionSource(
                        source_id=source.source_id,
                        source_type=source.source_type,
                        identifier=source.identifier,
                        confidence=confidence,
                        evidence=[
                            f"Similar to {similar_doc_id} (similarity: {similarity:.2f})"
                        ],
                        related_docs=[document.doc_id],
                    )
                    sources.append(new_source)

        # Check fingerprint database
        fp_hash = fingerprint.structural_hash[:16]
        if fp_hash in self._fingerprint_sources:
            source_id = self._fingerprint_sources[fp_hash]
            source = self._known_sources.get(source_id)
            if source:
                sources.append(
                    AttributionSource(
                        source_id=source.source_id,
                        source_type=source.source_type,
                        identifier=source.identifier,
                        confidence=AttributionConfidence.MEDIUM,
                        evidence=["Fingerprint matches known source"],
                        related_docs=[document.doc_id],
                    )
                )

        return sources

    def _identify_campaigns(
        self,
        document: Document,
        fingerprint: AttackFingerprint,
        sources: List[AttributionSource],
    ) -> List[AttackCampaign]:
        """Identify campaigns this document might belong to.

        Args:
            document: Document to check
            fingerprint: Document fingerprint
            sources: Identified sources

        Returns:
            List of potential campaigns
        """
        campaigns = []

        # Check source-based campaigns
        for source in sources:
            source_campaigns = self.find_campaigns_by_source(source.source_id)
            campaigns.extend(source_campaigns)

        # Check fingerprint similarity to campaign fingerprints
        for campaign in self._campaigns.values():
            for camp_fp in campaign.fingerprints:
                if fingerprint.similarity(camp_fp) > 0.7:
                    if campaign not in campaigns:
                        campaigns.append(campaign)
                    break

        return campaigns

    def _calculate_confidence(
        self,
        sources: List[AttributionSource],
        campaigns: List[AttackCampaign],
        similar: List[Tuple[str, float]],
    ) -> AttributionConfidence:
        """Calculate overall attribution confidence.

        Args:
            sources: Identified sources
            campaigns: Related campaigns
            similar: Similar attacks

        Returns:
            Overall confidence level
        """
        if not sources and not campaigns and not similar:
            return AttributionConfidence.UNCERTAIN

        # Score based on evidence
        score = 0.0

        # Source confidence
        if sources:
            max_source_conf = max(
                self._confidence_to_score(s.confidence) for s in sources
            )
            score = max(score, max_source_conf)

        # Campaign membership
        if campaigns:
            score = max(score, 0.6)

        # Similar attacks
        if similar:
            max_similarity = max(sim for _, sim in similar)
            score = max(score, max_similarity * 0.7)

        return self._score_to_confidence(score)

    def _generate_recommendations(
        self,
        document: Document,
        sources: List[AttributionSource],
        campaigns: List[AttackCampaign],
        confidence: AttributionConfidence,
    ) -> List[str]:
        """Generate recommendations based on attribution.

        Args:
            document: Analyzed document
            sources: Identified sources
            campaigns: Related campaigns
            confidence: Attribution confidence

        Returns:
            List of recommendations
        """
        recommendations = []

        if confidence in [AttributionConfidence.DEFINITE, AttributionConfidence.HIGH]:
            recommendations.append("Block identified source from future submissions")

        if sources:
            recommendations.append(
                f"Investigate {len(sources)} potential source(s) further"
            )

        if campaigns:
            recommendations.append(
                f"Review {len(campaigns)} related campaign(s) for additional IoCs"
            )
            recommendations.append("Check for other documents from same campaign")

        if confidence == AttributionConfidence.UNCERTAIN:
            recommendations.append("Collect more evidence for attribution")
            recommendations.append("Monitor for similar future attacks")

        return recommendations

    def _similarity_to_confidence(self, similarity: float) -> AttributionConfidence:
        """Convert similarity score to confidence level.

        Args:
            similarity: Similarity score (0-1)

        Returns:
            Confidence level
        """
        if similarity >= 0.95:
            return AttributionConfidence.DEFINITE
        elif similarity >= 0.80:
            return AttributionConfidence.HIGH
        elif similarity >= 0.50:
            return AttributionConfidence.MEDIUM
        elif similarity >= 0.20:
            return AttributionConfidence.LOW
        else:
            return AttributionConfidence.UNCERTAIN

    def _confidence_to_score(self, confidence: AttributionConfidence) -> float:
        """Convert confidence level to numeric score.

        Args:
            confidence: Confidence level

        Returns:
            Numeric score (0-1)
        """
        mapping = {
            AttributionConfidence.DEFINITE: 0.97,
            AttributionConfidence.HIGH: 0.87,
            AttributionConfidence.MEDIUM: 0.65,
            AttributionConfidence.LOW: 0.35,
            AttributionConfidence.UNCERTAIN: 0.10,
        }
        return mapping.get(confidence, 0.0)

    def _score_to_confidence(self, score: float) -> AttributionConfidence:
        """Convert numeric score to confidence level.

        Args:
            score: Numeric score (0-1)

        Returns:
            Confidence level
        """
        if score >= 0.95:
            return AttributionConfidence.DEFINITE
        elif score >= 0.80:
            return AttributionConfidence.HIGH
        elif score >= 0.50:
            return AttributionConfidence.MEDIUM
        elif score >= 0.20:
            return AttributionConfidence.LOW
        else:
            return AttributionConfidence.UNCERTAIN

    def _generate_source_id(self, source_type: str, identifier: str) -> str:
        """Generate unique source ID.

        Args:
            source_type: Type of source
            identifier: Source identifier

        Returns:
            Unique source ID
        """
        data = f"{source_type}:{identifier}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _generate_campaign_id(self, name: str, doc_ids: List[str]) -> str:
        """Generate unique campaign ID.

        Args:
            name: Campaign name
            doc_ids: Documents in campaign

        Returns:
            Unique campaign ID
        """
        data = f"{name}:{':'.join(sorted(doc_ids))}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _dfs_group(
        self,
        start: str,
        graph: Dict[str, Dict[str, float]],
        visited: Set[str],
    ) -> Set[str]:
        """DFS to find connected component.

        Args:
            start: Starting node
            graph: Adjacency graph
            visited: Already visited nodes

        Returns:
            Set of connected nodes
        """
        group = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            group.add(node)

            for neighbor in graph.get(node, {}):
                if neighbor not in visited:
                    stack.append(neighbor)

        return group
