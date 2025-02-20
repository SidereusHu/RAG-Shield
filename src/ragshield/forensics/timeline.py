"""Attack timeline reconstruction for RAG systems.

Reconstructs the sequence and progression of poisoning attacks
for forensic analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from ragshield.forensics.provenance import ProvenanceTracker, ProvenanceEventType


class TimelineEventType(Enum):
    """Types of timeline events."""

    INGESTION = "ingestion"
    DETECTION = "detection"
    ATTACK_STARTED = "attack_started"
    ATTACK_ESCALATION = "attack_escalation"
    QUARANTINE = "quarantine"
    MITIGATION = "mitigation"
    RECOVERY = "recovery"


@dataclass
class TimelineEvent:
    """An event in the attack timeline.

    Attributes:
        timestamp: When the event occurred
        event_type: Type of timeline event
        doc_ids: Related document IDs
        description: Human-readable description
        severity: Severity level (0-10)
        metadata: Additional event data
    """

    timestamp: datetime
    event_type: TimelineEventType
    doc_ids: List[str]
    description: str
    severity: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPhase:
    """A phase in an attack campaign.

    Attributes:
        name: Phase name
        start_time: Phase start time
        end_time: Phase end time
        events: Events in this phase
        doc_count: Number of documents in phase
        characteristics: Phase characteristics
    """

    name: str
    start_time: datetime
    end_time: Optional[datetime]
    events: List[TimelineEvent]
    doc_count: int
    characteristics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineReport:
    """Complete timeline report.

    Attributes:
        campaign_id: Unique campaign identifier
        start_time: When attack campaign started
        end_time: When attack was detected/stopped
        phases: Attack phases
        events: All timeline events
        statistics: Campaign statistics
        recommendations: Recommended actions
    """

    campaign_id: str
    start_time: datetime
    end_time: Optional[datetime]
    phases: List[AttackPhase]
    events: List[TimelineEvent]
    statistics: Dict[str, Any]
    recommendations: List[str]


class AttackTimelineReconstructor:
    """Reconstructs attack timelines from provenance data.

    Analyzes provenance chains to build a coherent timeline of attack events.
    """

    def __init__(
        self,
        provenance_tracker: ProvenanceTracker,
        phase_gap_threshold: timedelta = timedelta(hours=1),
    ):
        """Initialize timeline reconstructor.

        Args:
            provenance_tracker: Provenance tracker with document history
            phase_gap_threshold: Gap between events to start new phase
        """
        self.provenance = provenance_tracker
        self.phase_gap_threshold = phase_gap_threshold

    def reconstruct(
        self,
        doc_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TimelineReport:
        """Reconstruct attack timeline.

        Args:
            doc_ids: Specific documents to include (None = all flagged)
            start_time: Timeline start time filter
            end_time: Timeline end time filter

        Returns:
            Complete timeline report
        """
        # Get relevant documents
        if doc_ids is None:
            doc_ids = self.provenance.get_flagged_documents()
            doc_ids.extend(self.provenance.get_quarantined_documents())
            doc_ids = list(set(doc_ids))

        # Extract events from provenance
        events = self._extract_events(doc_ids, start_time, end_time)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        if not events:
            return TimelineReport(
                campaign_id=self._generate_campaign_id(),
                start_time=start_time or datetime.now(),
                end_time=end_time,
                phases=[],
                events=[],
                statistics={},
                recommendations=["No attack events found"],
            )

        # Identify attack phases
        phases = self._identify_phases(events)

        # Calculate statistics
        statistics = self._calculate_statistics(events, phases)

        # Generate recommendations
        recommendations = self._generate_recommendations(events, phases, statistics)

        return TimelineReport(
            campaign_id=self._generate_campaign_id(),
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            phases=phases,
            events=events,
            statistics=statistics,
            recommendations=recommendations,
        )

    def get_attack_progression(
        self, doc_ids: List[str]
    ) -> List[Tuple[datetime, int, str]]:
        """Get attack progression over time.

        Args:
            doc_ids: Documents to analyze

        Returns:
            List of (timestamp, cumulative_count, event_description)
        """
        events = self._extract_events(doc_ids)
        events.sort(key=lambda e: e.timestamp)

        progression = []
        cumulative = 0

        for event in events:
            if event.event_type == TimelineEventType.INGESTION:
                cumulative += len(event.doc_ids)
            progression.append((event.timestamp, cumulative, event.description))

        return progression

    def detect_attack_waves(
        self, doc_ids: List[str], wave_gap: timedelta = timedelta(hours=2)
    ) -> List[List[str]]:
        """Detect waves of attack (batch injections).

        Args:
            doc_ids: Documents to analyze
            wave_gap: Gap between waves

        Returns:
            List of waves (each wave is list of doc_ids)
        """
        # Get ingestion times
        ingestion_times = []
        for doc_id in doc_ids:
            chain = self.provenance.get_chain(doc_id)
            if chain:
                origin = chain.get_origin()
                if origin:
                    ingestion_times.append((doc_id, origin.timestamp))

        # Sort by time
        ingestion_times.sort(key=lambda x: x[1])

        if not ingestion_times:
            return []

        # Group into waves
        waves = [[ingestion_times[0][0]]]
        last_time = ingestion_times[0][1]

        for doc_id, timestamp in ingestion_times[1:]:
            if timestamp - last_time > wave_gap:
                waves.append([doc_id])
            else:
                waves[-1].append(doc_id)
            last_time = timestamp

        return waves

    def find_correlated_documents(
        self, doc_id: str, time_window: timedelta = timedelta(minutes=30)
    ) -> List[str]:
        """Find documents ingested around the same time.

        Args:
            doc_id: Reference document
            time_window: Time window to search

        Returns:
            List of correlated document IDs
        """
        chain = self.provenance.get_chain(doc_id)
        if not chain:
            return []

        origin = chain.get_origin()
        if not origin:
            return []

        ref_time = origin.timestamp
        correlated = []

        for other_id, other_chain in self.provenance.chains.items():
            if other_id == doc_id:
                continue

            other_origin = other_chain.get_origin()
            if other_origin:
                time_diff = abs(other_origin.timestamp - ref_time)
                if time_diff <= time_window:
                    correlated.append(other_id)

        return correlated

    def _extract_events(
        self,
        doc_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[TimelineEvent]:
        """Extract timeline events from provenance.

        Args:
            doc_ids: Document IDs to analyze
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            List of timeline events
        """
        events = []

        for doc_id in doc_ids:
            chain = self.provenance.get_chain(doc_id)
            if not chain:
                continue

            for prov_event in chain.events:
                # Apply time filter
                if start_time and prov_event.timestamp < start_time:
                    continue
                if end_time and prov_event.timestamp > end_time:
                    continue

                # Map provenance event to timeline event
                timeline_event = self._map_provenance_event(doc_id, prov_event)
                if timeline_event:
                    events.append(timeline_event)

        return events

    def _map_provenance_event(
        self, doc_id: str, prov_event
    ) -> Optional[TimelineEvent]:
        """Map a provenance event to a timeline event.

        Args:
            doc_id: Document ID
            prov_event: Provenance event

        Returns:
            Timeline event or None
        """
        event_mapping = {
            ProvenanceEventType.INGESTED: (
                TimelineEventType.INGESTION,
                "Document ingested",
                3,
            ),
            ProvenanceEventType.FLAGGED: (
                TimelineEventType.DETECTION,
                "Suspicious document detected",
                7,
            ),
            ProvenanceEventType.QUARANTINED: (
                TimelineEventType.QUARANTINE,
                "Document quarantined",
                8,
            ),
            ProvenanceEventType.DELETED: (
                TimelineEventType.MITIGATION,
                "Document removed",
                6,
            ),
        }

        mapping = event_mapping.get(prov_event.event_type)
        if not mapping:
            return None

        event_type, description, severity = mapping

        return TimelineEvent(
            timestamp=prov_event.timestamp,
            event_type=event_type,
            doc_ids=[doc_id],
            description=f"{description}: {doc_id}",
            severity=severity,
            metadata=prov_event.details,
        )

    def _identify_phases(self, events: List[TimelineEvent]) -> List[AttackPhase]:
        """Identify attack phases from events.

        Args:
            events: Timeline events

        Returns:
            List of attack phases
        """
        if not events:
            return []

        phases = []
        current_phase_events = [events[0]]
        phase_start = events[0].timestamp

        for i in range(1, len(events)):
            event = events[i]
            prev_event = events[i - 1]

            # Check if this starts a new phase
            if event.timestamp - prev_event.timestamp > self.phase_gap_threshold:
                # Close current phase
                phases.append(self._create_phase(current_phase_events, phase_start))

                # Start new phase
                current_phase_events = [event]
                phase_start = event.timestamp
            else:
                current_phase_events.append(event)

        # Close final phase
        if current_phase_events:
            phases.append(self._create_phase(current_phase_events, phase_start))

        # Name phases
        for i, phase in enumerate(phases):
            phase.name = f"Phase {i + 1}"

        return phases

    def _create_phase(
        self, events: List[TimelineEvent], start_time: datetime
    ) -> AttackPhase:
        """Create an attack phase from events.

        Args:
            events: Events in the phase
            start_time: Phase start time

        Returns:
            Attack phase
        """
        end_time = events[-1].timestamp if events else None
        doc_ids = set()
        for event in events:
            doc_ids.update(event.doc_ids)

        # Analyze characteristics
        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type.value] += 1

        return AttackPhase(
            name="",
            start_time=start_time,
            end_time=end_time,
            events=events,
            doc_count=len(doc_ids),
            characteristics={
                "event_breakdown": dict(event_types),
                "avg_severity": (
                    sum(e.severity for e in events) / len(events) if events else 0
                ),
            },
        )

    def _calculate_statistics(
        self, events: List[TimelineEvent], phases: List[AttackPhase]
    ) -> Dict[str, Any]:
        """Calculate timeline statistics.

        Args:
            events: All timeline events
            phases: Attack phases

        Returns:
            Statistics dictionary
        """
        if not events:
            return {}

        duration = events[-1].timestamp - events[0].timestamp

        # Document counts
        all_docs = set()
        for event in events:
            all_docs.update(event.doc_ids)

        # Severity distribution
        severities = [e.severity for e in events]

        return {
            "total_events": len(events),
            "total_documents": len(all_docs),
            "total_phases": len(phases),
            "duration_seconds": duration.total_seconds(),
            "duration_human": str(duration),
            "avg_severity": sum(severities) / len(severities) if severities else 0,
            "max_severity": max(severities) if severities else 0,
            "events_per_phase": len(events) / len(phases) if phases else 0,
        }

    def _generate_recommendations(
        self,
        events: List[TimelineEvent],
        phases: List[AttackPhase],
        statistics: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on timeline analysis.

        Args:
            events: Timeline events
            phases: Attack phases
            statistics: Calculated statistics

        Returns:
            List of recommendations
        """
        recommendations = []

        if statistics.get("total_documents", 0) > 10:
            recommendations.append(
                "Large-scale attack detected: consider full knowledge base audit"
            )

        if len(phases) > 3:
            recommendations.append(
                "Multi-phase attack campaign: investigate persistent threat actor"
            )

        if statistics.get("avg_severity", 0) > 7:
            recommendations.append(
                "High severity attack: prioritize immediate containment"
            )

        # Check for ongoing attack
        if events:
            last_event = events[-1]
            time_since_last = datetime.now() - last_event.timestamp
            if time_since_last < timedelta(hours=1):
                recommendations.append(
                    "Active attack in progress: enable enhanced monitoring"
                )

        return recommendations

    def _generate_campaign_id(self) -> str:
        """Generate unique campaign identifier.

        Returns:
            Campaign ID
        """
        import hashlib
        import time

        data = f"{time.time()}-{id(self)}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
