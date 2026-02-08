import json
import logging
from pathlib import Path
from typing import List, Optional
from domain.events import Event

logger = logging.getLogger(__name__)


class EventStore:
    """
    JSON-based event store with append-only semantics.
    
    Events are written to JSONL format (one JSON object per line) for:
    - Easy streaming reads
    - Crash resilience (partial writes don't corrupt entire file)
    - Line-by-line replay
    """
    
    def __init__(self, session_id: str, log_dir: str = "logs"):
        """
        Initialize event store for a search session.
        
        Args:
            session_id: Unique identifier for this search session
            log_dir: Base directory for logs (default: "logs")
        """
        self.session_id = session_id
        self.log_dir = Path(log_dir) / session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "events.jsonl"
        logger.info(f"EventStore initialized: {self.log_file}")
    
    def record(self, event: Event) -> None:
        """
        Append an event to the log.
        
        Args:
            event: Event instance to persist
        """
        try:
            event_dict = event.to_dict()
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
            
            logger.debug(f"Recorded event: {event.event_type}")
        
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            raise
    
    def replay(self) -> List[dict]:
        """
        Replay all events from the log.
        
        Returns:
            List of event dictionaries in chronological order
        """
        events = []
        
        if not self.log_file.exists():
            logger.warning(f"Log file does not exist: {self.log_file}")
            return events
        
        try:
            with open(self.log_file, 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event_dict = json.loads(line)
                        events.append(event_dict)
                    except json.JSONDecodeError as e:
                        logger.error(f"Malformed JSON at line {line_num}: {e}")
                        continue
            
            logger.info(f"Replayed {len(events)} events from {self.log_file}")
            return events
        
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            raise
    
    def get_events_by_type(self, event_type: str) -> List[dict]:
        """
        Replay events filtered by type.
        
        Args:
            event_type: Event type to filter (e.g., "CoverageGained")
        
        Returns:
            List of events matching the type
        """
        all_events = self.replay()
        return [e for e in all_events if e.get("event_type") == event_type]
    
    def get_event_count(self) -> int:
        """Get total number of events in the log."""
        if not self.log_file.exists():
            return 0
        
        with open(self.log_file, 'r') as f:
            return sum(1 for line in f if line.strip())
    
    def clear(self) -> None:
        """
        Clear all events (use with caution).
        
        This is primarily for testing. Production code should append-only.
        """
        if self.log_file.exists():
            self.log_file.unlink()
            logger.warning(f"Cleared event log: {self.log_file}")
