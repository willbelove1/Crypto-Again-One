from datetime import datetime
from typing import Dict, Any

class AIModuleBase:
    def __init__(self, module_id: str, ai_service: str):
        self.module_id = module_id
        self.ai_service = ai_service
        self.metadata: Dict[str, Any] = {}

    async def execute(self, input_data: Dict) -> Dict:
        """Standard execution interface"""
        try:
            # Validate input
            validated_input = self.validate_input(input_data)

            # Execute core logic
            result = await self.process(validated_input)

            # Format output
            return self.format_output(result)

        except Exception as e:
            # Basic error handling, can be expanded
            return self.handle_error(e, input_data)

    def validate_input(self, data: Dict) -> Dict:
        """Override in each module to validate input data."""
        # Default implementation: return data as is
        # Specific modules should implement their own validation logic
        print(f"Warning: validate_input not implemented for {self.module_id}. Passing through data.")
        return data

    async def process(self, data: Dict) -> Dict:
        """Override in each module - core processing logic."""
        # This method must be implemented by subclasses
        raise NotImplementedError(f"Process method not implemented in {self.module_id}")

    def format_output(self, result: Dict) -> Dict:
        """Standard output format."""
        return {
            "module_id": self.module_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": result,
            "metadata": self.metadata,
            "status": "success"
        }

    def handle_error(self, error: Exception, input_data: Dict) -> Dict:
        """Handles errors during module execution."""
        # Basic error logging, can be expanded (e.g., to include Sentry, specific logging, etc.)
        print(f"Error in module {self.module_id}: {str(error)}")
        return {
            "module_id": self.module_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error",
            "error_message": str(error),
            "input_data": input_data, # Optionally include input data for debugging
            "metadata": self.metadata
        }
