import asyncio
import json

# Imports from the 'modules' package
from modules.core.brain import WorkflowController
from modules.data.collector import DataCollector
from modules.analysis.analyzer import SignalAnalyzer
from modules.recommend.engine import RecommenderCore
from modules.report.writer import ReportWriter
# AIModuleBase is not directly used here if we are using the actual (skeleton) modules
# from modules.base import AIModuleBase

async def main_logic():
    """
    Initializes and runs the Crypto AI System pipeline with skeleton modules.
    """
    print("Initializing Crypto AI System Workflow from crypto_ai_system/main.py...")
    controller = WorkflowController()

    # Instantiate the actual (skeleton) module classes
    data_collector = DataCollector()
    signal_analyzer = SignalAnalyzer()
    recommender_core = RecommenderCore()
    report_writer = ReportWriter()

    # Register module instances with the WorkflowController
    # The module IDs used here must match those defined in the WorkflowController's DAG
    await controller.register_module("mod.data.collector", data_collector)
    await controller.register_module("mod.analysis.ta_fa", signal_analyzer)
    await controller.register_module("mod.recommend.engine", recommender_core)
    await controller.register_module("mod.report.writer", report_writer)

    print("\nExecuting full pipeline with skeleton modules...")

    # Important Note on DataCollector input:
    # The current WorkflowController.execute_pipeline() method passes an empty dictionary `{}`
    # as the initial input to the first module in the DAG (which is DataCollector).
    # However, the DataCollector.validate_input() method in its current skeleton form
    # expects specific keys (like 'symbols', 'timeframe', etc.) and will likely raise a
    # ValueError if these keys are missing.
    #
    # To make the pipeline run further for demonstration purposes without modifying the
    # core logic of the skeleton DataCollector extensively for this main.py, one might typically:
    #   1. Temporarily modify DataCollector.validate_input to be more lenient for empty inputs.
    #   2. (Better for future) Enhance WorkflowController to accept an `initial_payload`
    #      that can be passed to the first module(s) of the pipeline.
    #
    # For now, running this script will likely result in an error originating from
    # DataCollector.validate_input due to the empty initial input from the pipeline.
    # This highlights a point of integration to be addressed in further development.

    pipeline_results = await controller.execute_pipeline(pipeline_name="crypto_skeleton_pipeline_run")

    print("\n--- Full Pipeline Execution Results ---")
    print(json.dumps(pipeline_results, indent=2))

if __name__ == "__main__":
    # When running this script directly (e.g., `python crypto_ai_system/main.py`),
    # Python adds the directory containing the script (in this case, `crypto_ai_system/`)
    # to sys.path. This allows the `from modules...` imports to work correctly.
    asyncio.run(main_logic())
