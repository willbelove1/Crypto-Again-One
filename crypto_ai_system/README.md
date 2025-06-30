# Crypto AI System

This project is the foundational structure for a Crypto AI System as outlined in the `crypto_ai_tech_spec.md`. It aims to automate the process of data collection, analysis, recommendation generation, and reporting for cryptocurrency investments by leveraging multiple AI models.

## Project Purpose

The system is designed as a modular pipeline orchestrated by a core "Brain" module. It processes cryptocurrency market data through several stages:
1.  **Data Collection**: Gathers market data (prices, volume, order books) from various sources.
2.  **Signal Analysis**: Performs technical (TA) and fundamental (FA) analysis on the collected data to identify signals.
3.  **Recommendation**: Uses AI models to generate investment recommendations (BUY/SELL/HOLD) based on the analysis.
4.  **Reporting**: Compiles the findings and recommendations into various report formats.

## Current Status: Skeleton Implementation

This repository currently contains the initial directory structure and Python class skeletons for the system. Key functionalities are represented by placeholder methods and classes. **The system is not yet fully functional or tested.**

## Module Structure

The system is organized into the following core modules located in `crypto_ai_system/modules/`:

*   **`base.py`**: Contains `AIModuleBase`, the abstract base class for all processing modules, defining a standard interface.
*   **`core/`**:
    *   `brain.py`: Implements `APIPoolManager` for handling API keys and rate limits for various AI services, and `WorkflowController` for managing the execution pipeline of modules based on a Directed Acyclic Graph (DAG).
*   **`data/`**:
    *   `collector.py`: `DataCollector` module, responsible for fetching data from exchanges. (Primary AI intended: DeepSeek)
*   **`analysis/`**:
    *   `analyzer.py`: `SignalAnalyzer` module, responsible for TA/FA. (Primary AI intended: ChatGPT)
*   **`recommend/`**:
    *   `engine.py`: `RecommenderCore` module, for generating investment advice. (Primary AI intended: Gemini Pro)
*   **`report/`**:
    *   `writer.py`: `ReportWriter` module, for creating summaries and detailed reports. (Primary AI intended: Claude)

Each module directory also contains an `__init__.py` to define it as a Python package.

## Theoretical Usage (Running the Mock Pipeline)

While the system isn't fully implemented, you can theoretically run the `WorkflowController` with the current mock module implementations to see the structural flow.

**Prerequisites:**
*   Python 3.8+ (due to `asyncio` and type hinting usage)

**Example `main.py` (create this file in the `crypto_ai_system` directory):**

```python
import asyncio
import json

# Ensure PYTHONPATH includes the parent directory of crypto_ai_system if running from outside
# Or, install the package if setup.py is created. For now, direct import assumes correct path.

from modules.base import AIModuleBase # Required for mock module definition below
from modules.core.brain import WorkflowController
# Import actual module classes (even if they are skeletons)
from modules.data.collector import DataCollector
from modules.analysis.analyzer import SignalAnalyzer
from modules.recommend.engine import RecommenderCore
from modules.report.writer import ReportWriter

# --- Mock AIModuleBase for WorkflowController example if not using full classes ---
# This is only needed if you want to run the example without the full class implementations
# from datetime import datetime
# from typing import Dict, Any
# class AIModuleBase:
#     def __init__(self, module_id: str, ai_service: str):
#         self.module_id = module_id; self.ai_service = ai_service; self.metadata: Dict[str, Any] = {}; self.status = "init"
#     async def execute(self, input_data: Dict):
#         print(f"Executing {self.module_id} with input: {list(input_data.keys())}")
#         self.status = "processing"
#         validated_input = self.validate_input(input_data)
#         result = await self.process(validated_input)
#         self.status = "completed"
#         return self.format_output(result)
#     def validate_input(self, data: Dict) -> Dict:
#         print(f"Validating input for {self.module_id} (mock)")
#         return data
#     async def process(self, data: Dict) -> Dict:
#         print(f"Processing data in {self.module_id} (mock)")
#         await asyncio.sleep(0.1) # Simulate work
#         return {"mock_result": f"data from {self.module_id}", "processed_input_keys": list(data.keys())}
#     def format_output(self, result: Dict) -> Dict:
#         return {
#             "module_id": self.module_id,
#             "timestamp": datetime.utcnow().isoformat(),
#             "data": result,
#             "metadata": self.metadata,
#             "status": "success" # Assume success for mock
#         }
# class DataCollector(AIModuleBase): # Mock
#     def __init__(self): super().__init__("mod.data.collector", "deepseek_mock")
# class SignalAnalyzer(AIModuleBase): # Mock
#     def __init__(self): super().__init__("mod.analysis.ta_fa", "chatgpt_mock")
# class RecommenderCore(AIModuleBase): # Mock
#     def __init__(self): super().__init__("mod.recommend.engine", "gemini_mock")
# class ReportWriter(AIModuleBase): # Mock
#     def __init__(self): super().__init__("mod.report.writer", "claude_mock")
# --- End Mock AIModuleBase ---


async def main():
    print("Initializing Crypto AI System Workflow...")
    controller = WorkflowController()

    # Instantiate actual modules
    data_collector = DataCollector()
    signal_analyzer = SignalAnalyzer()
    recommender_core = RecommenderCore()
    report_writer = ReportWriter()

    # Register modules with the controller
    # The IDs must match those in the WorkflowController's DAG definition
    await controller.register_module("mod.data.collector", data_collector)
    await controller.register_module("mod.analysis.ta_fa", signal_analyzer)
    await controller.register_module("mod.recommend.engine", recommender_core)
    await controller.register_module("mod.report.writer", report_writer)

    print("\nExecuting full pipeline...")
    # The initial input to the first module (DataCollector) needs to be provided
    # if its validate_input expects something.
    # For DataCollector, the spec INPUT_SCHEMA is:
    initial_input_for_collector = {
        "symbols": ["BTC", "ETH"],
        "timeframe": "1h",
        "data_types": ["price", "volume", "orderbook"],
        "sources": ["binance_mock", "okx_mock"]
    }
    # The execute_pipeline method currently doesn't pass initial_input to the first module.
    # This is a simplification in the current WorkflowController.
    # For a real run, DataCollector's execute method would need to be adapted or
    # WorkflowController enhanced to pass initial parameters to the first module(s).
    # For now, DataCollector's validate_input will receive an empty dict from the current pipeline design.
    # We can simulate this by making its `execute` method robust to empty input or by modifying it.
    # For the sake of this README example, let's assume DataCollector.execute can be triggered
    # and will use its own default parameters or that the `validate_input` is lenient for the mock.

    # To make the current pipeline run, DataCollector's validate_input needs to handle an empty dict,
    # or its process method needs to use default data if the input is empty.
    # The current DataCollector.validate_input raises errors if keys are missing.
    # One way to run this example is to modify DataCollector to accept an optional initial_payload
    # or make its validate_input more lenient when its input_data is empty (from pipeline).

    # Let's assume we modify the DataCollector's execute method to use `initial_input_for_collector`
    # if its input_data from the pipeline is empty or lacks specific keys.
    # This is a workaround for the example.

    # A better way for the pipeline:
    # The `execute_pipeline` could take an `initial_context` argument.
    # controller.module_instances["mod.data.collector"].initial_payload = initial_input_for_collector
    # And then the module's execute method uses this.

    # For now, the pipeline passes {} to the first module. We'll see placeholder errors from DataCollector.validate_input.
    # To avoid this, you'd typically pass specific input to the pipeline start.
    # The `WorkflowController.execute_module` passes `input_data_for_module` which is built from dependency results.
    # For the first module, this will be empty.

    # To make DataCollector work with current pipeline:
    # Option 1: Modify DataCollector.validate_input to allow empty dict or specific conditions.
    # Option 2: Modify WorkflowController.execute_pipeline to pass a starting context.
    # For this README, we'll note this limitation. The printouts will show the flow.

    pipeline_results = await controller.execute_pipeline(pipeline_name="mock_crypto_run")

    print("\n--- Full Pipeline Execution Results ---")
    print(json.dumps(pipeline_results, indent=2))

    # To test APIPoolManager (theoretical, needs actual async tasks using it)
    # api_pool = controller.module_instances["mod.recommend.engine"].ai_pool # Example
    # async def mock_llm_task(api_key: str):
    #     print(f"Mock LLM task called with key: {api_key}")
    #     # Simulate a rate limit error for a specific key to see rotation
    #     if api_key == "key2": # Assuming "key2" is in "gemini" pool
    #         raise RateLimitError("Simulated rate limit!")
    #     await asyncio.sleep(0.1)
    #     return f"LLM response with {api_key}"

    # print("\n--- Testing APIPoolManager (Theoretical) ---")
    # try:
    #     # RecommenderCore uses "gemini" as primary
    #     response = await api_pool.execute_with_rotation("gemini", mock_llm_task, max_retries=4)
    #     print(f"APIPoolManager response: {response}")
    # except Exception as e:
    #     print(f"APIPoolManager test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

```

**To run the example:**
1.  Save the code above as `main.py` in the `crypto_ai_system` directory (alongside the `modules` directory and this `README.md`).
2.  Navigate to the `crypto_ai_system` directory in your terminal.
3.  Run `python main.py`.

You will see print statements indicating the flow through the mock modules and the structure of the data being passed. Note that `DataCollector.validate_input` will likely show errors due to the pipeline passing an empty dictionary as input to the first module. This highlights an area for refinement in `WorkflowController` or module design for initial inputs.

## Future Development

To make this system fully operational, the following steps are crucial:

1.  **Implement Placeholder Logic:**
    *   Flesh out all `process`, `validate_input`, and specific helper methods in each module (e.g., `get_price_data` in `DataCollector`, `call_llm_api` in `RecommenderCore`, actual TA/FA calculations).
    *   Implement the actual logic for `TechnicalAnalysisEngine`, `FundamentalAnalysisEngine`, and `ReportTemplateEngine`.
2.  **API Key Management:**
    *   Securely manage API keys for AI services (Gemini, OpenAI, Anthropic, DeepSeek). This might involve environment variables, a config file (e.g., `.env`), or a secrets management system.
    *   Update `APIPoolManager` to load keys from this configuration.
3.  **Refine AI Prompts & Parsing:**
    *   Develop and test robust prompts for each AI model interaction.
    *   Implement reliable parsing for AI model responses, especially for JSON or structured data.
4.  **Input Handling for Pipeline Start:**
    *   Refine `WorkflowController` or module interfaces to handle initial input parameters for the pipeline (e.g., symbols, timeframes for `DataCollector`).
5.  **Error Handling & Resilience:**
    *   Implement more comprehensive error handling, retry mechanisms (beyond `APIPoolManager`), and fallback strategies within each module and the pipeline.
6.  **Testing:**
    *   Write unit tests for each module's functionality (`pytest` is mentioned in the spec).
    *   Develop integration tests for the entire `WorkflowController` pipeline.
    *   Mock external services (APIs, exchanges) for reliable testing.
7.  **Configuration:**
    *   Add configuration files for parameters like DAG structure, module settings, API endpoints, etc.
8.  **Concurrency and Performance:**
    *   Optimize `asyncio` usage for better performance, especially in I/O-bound tasks like API calls and data fetching.
9.  **Logging & Monitoring:**
    *   Integrate structured logging throughout the application.
    *   Consider adding monitoring and alerting capabilities.
10. **Deployment:**
    *   Prepare for deployment (e.g., Docker containerization as mentioned in the spec).

## Dependencies
*   Python 3.8+
*   (No external libraries are used in the current skeleton, but will be needed for full implementation, e.g., `httpx`/`aiohttp` for API calls, `pandas` for data manipulation, specific AI SDKs.)

This README provides a starting point. As the project develops, it should be updated with more detailed setup instructions, API documentation (if any), and usage examples.
