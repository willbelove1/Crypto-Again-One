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

## Running the System (Skeleton)

An entry point script `main.py` is provided in the `crypto_ai_system/` directory. This script initializes the `WorkflowController` and registers the skeleton modules to demonstrate the pipeline flow.

**To run the system:**

1.  Open your terminal.
2.  Navigate to the `crypto_ai_system` directory:
    ```bash
    cd path/to/crypto_ai_system
    ```
3.  Run the `main.py` script:
    ```bash
    python main.py
    ```

You will see print statements indicating:
*   Initialization of the workflow.
*   Registration of each module.
*   The order of execution based on the DAG.
*   The (mock) processing within each module.
*   The final aggregated results (which will be based on mock data).

**Initial Input for `DataCollector`:**
The `crypto_ai_system/main.py` script now defines and passes an initial set of parameters (symbols, timeframe, etc.) to the `DataCollector` module when the pipeline starts. This is achieved by using the `initial_inputs_for_modules` argument in the `WorkflowController.execute_pipeline()` method.

This ensures that `DataCollector` receives the necessary input to pass its validation step and begin its (mock) processing, allowing the pipeline to proceed through all modules. You should now see output from all skeleton modules when running `python main.py`.

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
