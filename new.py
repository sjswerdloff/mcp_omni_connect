from collections import deque
import hashlib
from typing import Tuple, List, Set, Optional, Dict, Any, Union

def hash_text(text: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(text.encode()).hexdigest()

class RobustLoopDetector:
    def __init__(
        self, 
        maxlen: int = 10,
        min_calls: int = 3,
        same_output_threshold: int = 3,
        same_input_threshold: int = 3,
        full_dup_threshold: int = 3,
        pattern_detection: bool = True,
        max_pattern_length: int = 4
    ):
        """Initialize a robust loop detector.
        
        Args:
            maxlen: Maximum number of recent interactions to track
            min_calls: Minimum number of interactions before loop detection is active
            same_output_threshold: Maximum unique outputs before it's considered a loop
            same_input_threshold: Maximum unique inputs before it's considered a loop
            full_dup_threshold: Maximum unique interaction signatures before it's considered a loop
            pattern_detection: Whether to detect repeating patterns
            max_pattern_length: Maximum pattern length to detect
        """
        self.recent_interactions = deque(maxlen=maxlen)
        self.min_calls = min_calls
        self.same_output_threshold = same_output_threshold
        self.same_input_threshold = same_input_threshold
        self.full_dup_threshold = full_dup_threshold
        self.pattern_detection = pattern_detection
        self.max_pattern_length = max_pattern_length
        
        # Cache for performance optimization
        self._cache: Dict[str, Any] = {}
        self._interaction_count = 0
    
    def record_tool_call(self, tool_name: str, tool_input: str, tool_output: str) -> None:
        """Record a new tool call interaction.
        
        Args:
            tool_name: Name of the tool that was called
            tool_input: Input provided to the tool
            tool_output: Output returned by the tool
        """
        signature = ("tool", tool_name, hash_text(tool_input), hash_text(tool_output))
        self.recent_interactions.append(signature)
        self._interaction_count += 1
        
        # Invalidate cache
        self._cache = {}
    
    def record_message(self, user_message: str, assistant_message: str) -> None:
        """Record a new message exchange interaction.
        
        Args:
            user_message: Message from the user
            assistant_message: Response from the assistant
        """
        signature = ("message", "", hash_text(user_message), hash_text(assistant_message))
        self.recent_interactions.append(signature)
        self._interaction_count += 1
        
        # Invalidate cache
        self._cache = {}
        
    def record_interaction(self, interaction_type: str, input_data: str, output_data: str, metadata: str = "") -> None:
        """Generic method to record any type of interaction.
        
        Args:
            interaction_type: Type of interaction (e.g., "tool", "message", "function")
            input_data: Input for the interaction
            output_data: Output from the interaction
            metadata: Additional information about the interaction (e.g., tool name)
        """
        signature = (interaction_type, metadata, hash_text(input_data), hash_text(output_data))
        self.recent_interactions.append(signature)
        self._interaction_count += 1
        
        # Invalidate cache
        self._cache = {}
    
    def reset(self) -> None:
        """Reset the detector, clearing all recorded interactions."""
        self.recent_interactions.clear()
        self._cache = {}
        self._interaction_count = 0
    
    def _get_unique_inputs(self) -> Set[str]:
        """Get set of unique inputs (cached)."""
        if 'unique_inputs' not in self._cache:
            self._cache['unique_inputs'] = set(sig[2] for sig in self.recent_interactions)
        return self._cache['unique_inputs']
    
    def _get_unique_outputs(self) -> Set[str]:
        """Get set of unique outputs (cached)."""
        if 'unique_outputs' not in self._cache:
            self._cache['unique_outputs'] = set(sig[3] for sig in self.recent_interactions)
        return self._cache['unique_outputs']
    
    def _get_unique_signatures(self) -> Set[Tuple]:
        """Get set of unique full signatures (cached)."""
        if 'unique_signatures' not in self._cache:
            self._cache['unique_signatures'] = set(self.recent_interactions)
        return self._cache['unique_signatures']
    
    def is_ready(self) -> bool:
        """Check if we have enough data to start detecting loops."""
        return self._interaction_count >= self.min_calls
    
    def is_stuck_same_output(self) -> bool:
        """Detect if we're stuck getting the same outputs repeatedly."""
        if not self.is_ready() or len(self.recent_interactions) < self.recent_interactions.maxlen:
            return False
        return len(self._get_unique_outputs()) <= self.same_output_threshold
    
    def is_stuck_same_input(self) -> bool:
        """Detect if we're stuck using the same inputs repeatedly."""
        if not self.is_ready() or len(self.recent_interactions) < self.recent_interactions.maxlen:
            return False
        return len(self._get_unique_inputs()) <= self.same_input_threshold
    
    def is_fully_stuck(self) -> bool:
        """Detect if we're stuck in the same input-output combinations."""
        if not self.is_ready() or len(self.recent_interactions) < self.recent_interactions.maxlen:
            return False
        return len(self._get_unique_signatures()) <= self.full_dup_threshold
    
    def find_repeating_pattern(self) -> Optional[List[Tuple]]:
        """Find a repeating pattern in the interaction history.
        
        Returns:
            The detected pattern as a list of signatures, or None if no pattern found
        """
        if not self.pattern_detection or not self.is_ready():
            return None
            
        interactions = list(self.recent_interactions)
        
        # Check patterns of different lengths
        for pattern_len in range(1, min(self.max_pattern_length + 1, len(interactions) // 2 + 1)):
            # Check if the last N elements repeat the previous N elements
            pattern = interactions[-pattern_len:]
            prev_pattern = interactions[-2*pattern_len:-pattern_len]
            
            if pattern == prev_pattern:
                # Found a repeating pattern
                return pattern
                
        # For more complex patterns (like A-B-A-C-A-B-A-C)
        # We could implement more sophisticated algorithms
        
        return None
    
    def has_pattern_loop(self) -> bool:
        """Check if there's a repeating pattern loop."""
        return self.find_repeating_pattern() is not None
    
    def is_looping(self) -> bool:
        """Check if any loop detection method indicates a loop."""
        return (
            self.is_stuck_same_output() or 
            self.is_stuck_same_input() or 
            self.is_fully_stuck() or 
            self.has_pattern_loop()
        )
    
    def get_loop_type(self) -> List[str]:
        """Get detailed information about the type of loop detected.
        
        Returns:
            List of strings describing the detected loop types
        """
        if not self.is_looping():
            return []
            
        loop_types = []
        if self.is_stuck_same_output():
            loop_types.append("same_output")
        if self.is_stuck_same_input():
            loop_types.append("same_input")
        if self.is_fully_stuck():
            loop_types.append("full_duplication")
        
        pattern = self.find_repeating_pattern()
        if pattern:
            loop_types.append(f"repeating_pattern(len={len(pattern)})")
            
        return loop_types
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current state.
        
        Returns:
            Dictionary with statistics about inputs, outputs, etc.
        """
        if not self.recent_interactions:
            return {"interactions": 0}
            
        # Count different types of interactions
        interaction_types = {}
        for sig in self.recent_interactions:
            itype = sig[0]
            interaction_types[itype] = interaction_types.get(itype, 0) + 1
            
        return {
            "interactions": self._interaction_count,
            "queue_size": len(self.recent_interactions),
            "unique_inputs": len(self._get_unique_inputs()),
            "unique_outputs": len(self._get_unique_outputs()),
            "unique_signatures": len(self._get_unique_signatures()),
            "interaction_types": interaction_types,
            "repeating_pattern": self.find_repeating_pattern() is not None
        }
    
    def get_interaction_types(self) -> Dict[str, int]:
        """Get counts of each interaction type in the history.
        
        Returns:
            Dictionary mapping interaction types to their counts
        """
        type_counts = {}
        for sig in self.recent_interactions:
            itype = sig[0]
            type_counts[itype] = type_counts.get(itype, 0) + 1
        return type_counts

# Usage example
if __name__ == "__main__":
    detector = RobustLoopDetector()
    
    # Example 1: Tool call loop detection
    print("Example 1: Tool Call Loop Detection")
    detector.record_tool_call("search", "climate change", "global warming articles")
    detector.record_tool_call("search", "carbon emissions", "statistical data")
    detector.record_tool_call("search", "climate change", "global warming articles")  # Repeated
    detector.record_tool_call("search", "carbon emissions", "statistical data")  # Repeated
    detector.record_tool_call("search", "climate change", "global warming articles")  # Repeated
    detector.record_tool_call("search", "carbon emissions", "statistical data")  # Repeated
    
    if detector.is_looping():
        print(f"Tool call loop detected! Types: {detector.get_loop_type()}")
        print(f"Stats: {detector.get_stats()}")
    
    detector.reset()
    
    # Example 2: Message exchange loop detection
    print("\nExample 2: Message Exchange Loop Detection")
    detector.record_message("How do I reset my password?", "You can reset your password by clicking the 'Forgot Password' link.")
    detector.record_message("That didn't work", "I'm sorry to hear that. Please try clearing your browser cache and try again.")
    detector.record_message("Still not working", "Please try using a different browser or device.")
    detector.record_message("Still having issues", "Please try using a different browser or device.")  # Repeated assistant response
    detector.record_message("Still not working", "Please try using a different browser or device.")  # Repeated assistant response
    detector.record_message("Still not working", "Please try using a different browser or device.")  # Repeated assistant response
    
    if detector.is_looping():
        print(f"Message loop detected! Types: {detector.get_loop_type()}")
        print(f"Stats: {detector.get_stats()}")
    
    detector.reset()
    
    # Example 3: Mixed interaction type pattern detection
    print("\nExample 3: Mixed Interaction Type Pattern Detection")
    # First pattern sequence
    detector.record_message("What's the weather?", "It's sunny today.")
    detector.record_tool_call("weather_api", "location=NY", "72°F, Sunny")
    #detector.record_message("What about tomorrow?", "I'll check that for you.")
    #detector.record_tool_call("weather_api", "location=NY&day=tomorrow", "68°F, Partly Cloudy")
    
    # Repeat the pattern
    detector.record_message("What's the weather?", "It's sunny today.")
    detector.record_tool_call("weather_api", "location=NY", "72°F, Sunny")
    #detector.record_message("What about tomorrow?", "I'll check that for you.")
    #detector.record_tool_call("weather_api", "location=NY&day=tomorrow", "68°F, Partly Cloudy")
    
    if detector.is_looping():
        print(f"Mixed interaction pattern detected! Types: {detector.get_loop_type()}")
        print(f"Stats: {detector.get_stats()}")
        
        pattern = detector.find_repeating_pattern()
        if pattern:
            print("Detected pattern:")
            for i, sig in enumerate(pattern):
                itype, metadata, input_hash, output_hash = sig
                print(f"  {i+1}. Type: {itype}, Metadata: {metadata[:10]}...")
    else:
        print("No pattern detected. Debug info:")
        print(f"  Pattern detection enabled: {detector.pattern_detection}")
        print(f"  Min calls threshold met: {detector.is_ready()}")
        print(f"  Current interactions: {len(detector.recent_interactions)}/{detector.recent_interactions.maxlen}")
        print(f"  Max pattern length: {detector.max_pattern_length}")