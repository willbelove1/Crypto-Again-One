from itertools import cycle
from typing import Dict, Any, Callable, Coroutine
import asyncio

# Placeholder for custom error
class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass

class APIPoolManager:
    def __init__(self):
        self.pools = {
            "gemini": cycle(["key1", "key2", "key3"]),
            "openai": cycle(["gpt_key1", "gpt_key2"]),
            "anthropic": cycle(["claude_key1"]),
            # Added from spec for DataCollector and SignalAnalyzer
            "deepseek": cycle(["deepseek_key1", "deepseek_key2"]),
            "chatgpt": cycle(["chatgpt_key1", "chatgpt_key2"]), # Assuming chatgpt is different from openai gpt keys
            "gpt-4o": cycle(["gpt4o_key1"]) # Added from spec for RecommenderCore backup
        }
        self.cooldowns: Dict[str, float] = {} # Stores key cooldown timestamps
        self.rate_limits: Dict[str, Dict[str, Any]] = {} # Stores service-specific rate limit info (e.g., requests per minute)
        self.key_specific_cooldown_duration: float = 60.0 # Default cooldown for a key in seconds
        self.service_cooldown_duration: float = 5.0 # Default cooldown for a service if all keys are on cooldown

    def is_key_available(self, key: str) -> bool:
        """Checks if a specific key is currently under cooldown."""
        # This is a simplified check. Real-world scenarios might involve checking API provider status.
        if key in self.cooldowns and self.cooldowns[key] > asyncio.get_event_loop().time():
            return False
        return True

    async def wait_and_retry(self, service: str, key_to_wait: str) -> str:
        """Waits for a key's cooldown or a general service cooldown, then retries getting a key."""
        if key_to_wait in self.cooldowns:
            wait_time = self.cooldowns[key_to_wait] - asyncio.get_event_loop().time()
            if wait_time > 0:
                print(f"Key {key_to_wait} for service {service} is on cooldown. Waiting for {wait_time:.2f}s.")
                await asyncio.sleep(wait_time)
        else:
            # If the specific key wasn't the issue, but get_available_key failed,
            # it implies a broader service issue or all keys are on cooldown.
            # Wait for a general service cooldown.
            print(f"All keys for {service} might be on cooldown or service unavailable. Waiting {self.service_cooldown_duration}s before retrying.")
            await asyncio.sleep(self.service_cooldown_duration)

        # After waiting, try to get a key again. This could be the same key or the next one.
        return await self.get_available_key(service)


    async def get_available_key(self, service: str) -> str:
        """
        Gets the next available key for a given service.
        If a key is on cooldown, it will try the next one.
        If all keys for the service are on cooldown, it will raise an Exception after trying all.
        """
        if service not in self.pools:
            raise ValueError(f"Unknown service: {service}")

        num_keys_for_service = len(self.pools[service].__self__) # Access the original iterable length

        for _ in range(num_keys_for_service):
            key = next(self.pools[service])
            if self.is_key_available(key):
                return key
            else:
                print(f"Key {key} for service {service} is on cooldown.")

        # If all keys were checked and none are available, wait for the first key encountered that was on cooldown
        # This is a simplification; a more robust system might track the earliest available key.
        print(f"All keys for service {service} are currently on cooldown. Waiting...")
        first_key_in_cycle = next(self.pools[service]) # Get the "first" key again to wait for its cooldown
        return await self.wait_and_retry(service, first_key_in_cycle)


    async def handle_cooldown(self, service: str, key: str):
        """Puts a key on cooldown."""
        # In a real system, this would be more sophisticated, potentially
        # using info from API response headers.
        cooldown_until = asyncio.get_event_loop().time() + self.key_specific_cooldown_duration
        self.cooldowns[key] = cooldown_until
        print(f"Key {key} for service {service} put on cooldown until {cooldown_until}.")

    async def execute_with_rotation(self, service: str, task: Callable[[str], Coroutine[Any, Any, Any]], max_retries: int = 3) -> Any:
        """
        Executes a task using a key from the specified service pool.
        Handles key rotation and retries on RateLimitError.
        """
        if service not in self.pools:
            # Try to find a matching service if a more specific one (e.g. gpt-4o for openai) is requested
            general_service = service.split('-')[0] if '-' in service else None
            if general_service and general_service in self.pools:
                print(f"Specific service '{service}' not found, using general service '{general_service}'.")
                service = general_service
            else:
                raise ValueError(f"Service {service} not found in API pools.")

        last_exception = None

        for attempt in range(max_retries):
            key = await self.get_available_key(service) # This might wait if keys are on cooldown

            try:
                print(f"Attempt {attempt + 1}/{max_retries} using key {key} for service {service}")
                return await task(key)
            except RateLimitError as e:
                print(f"Rate limit error with key {key} for service {service}. Attempt {attempt + 1}/{max_retries}.")
                await self.handle_cooldown(service, key)
                last_exception = e
                # Wait a bit before trying the next key, especially if it's the same key after cooldown
                await asyncio.sleep(1)
            except Exception as e:
                # For other errors, we might not want to rotate or retry indefinitely.
                print(f"Unhandled error with key {key} for service {service}: {e}")
                last_exception = e
                # Depending on the error, you might want to handle it differently,
                # e.g., put key on cooldown, or break immediately.
                # For now, let's put the key on a shorter cooldown and retry.
                await self.handle_cooldown(service, key) # Shorter cooldown for non-rate-limit errors?
                await asyncio.sleep(1)


        if last_exception:
            raise Exception(f"All retries failed for service {service}. Last error: {last_exception}")
        else:
            # This case should ideally not be reached if max_retries > 0
            raise Exception(f"All retries failed for service {service} without a specific error.")

class WorkflowController:
    def __init__(self):
        self.states = {
            "IDLE": self.handle_idle,
            "COLLECTING": self.handle_collecting,
            "ANALYZING": self.handle_analyzing,
            "RECOMMENDING": self.handle_recommending,
            "REPORTING": self.handle_reporting
        }
        self.current_state = "IDLE"
        self.dag = self.build_dag()
        self.module_instances: Dict[str, Any] = {} # To store instantiated modules

    def build_dag(self) -> Dict[str, list[str]]:
        # Dependencies for each module
        return {
            "mod.data.collector": [],
            "mod.analysis.ta_fa": ["mod.data.collector"],
            "mod.recommend.engine": ["mod.analysis.ta_fa"],
            "mod.report.writer": ["mod.recommend.engine", "mod.analysis.ta_fa"] # Report writer might need analysis data too
        }

    def topological_sort(self, dag: Dict[str, list[str]]) -> list[str]:
        """Performs a topological sort on the DAG."""
        # Standard Kahn's algorithm for topological sorting
        in_degree = {u: 0 for u in dag}
        for u in dag:
            for v in dag[u]:
                in_degree[v] = in_degree.get(v, 0) + 1 # Ensure all nodes are in in_degree

        # Initialize queue with all nodes with no incoming edges
        queue = [u for u in dag if u in in_degree and in_degree[u] == 0]

        sorted_order = []

        while queue:
            u = queue.pop(0)
            sorted_order.append(u)

            # For each neighbor v of u, reduce its in-degree by 1
            # This requires reversing the graph logic or iterating through all nodes
            for v in dag: # Check all nodes
                if u in dag[v]: # if u is a dependency of v
                    in_degree[v] -=1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(sorted_order) == len(dag):
            return sorted_order
        else:
            # Check for cycles by finding nodes with in_degree > 0
            # This part of the original Kahn's algorithm is slightly different
            # when checking for cycles. A simpler check is if sorted_order length matches dag length.
            # For more detailed cycle detection, one would typically track visited nodes during DFS/BFS.
            raise Exception("DAG has a cycle, topological sort failed.")


    async def register_module(self, module_id: str, instance: Any):
        """Registers a module instance."""
        self.module_instances[module_id] = instance
        print(f"Module {module_id} registered.")

    async def execute_module(self, module_id: str, dependencies_results: list[Dict]) -> Dict:
        """Executes a single module using its registered instance."""
        if module_id not in self.module_instances:
            raise Exception(f"Module {module_id} not registered or initialized.")

        module_instance = self.module_instances[module_id]

        # Prepare input for the current module
        # The input_data for a module could be a dictionary of its dependencies' outputs
        input_data_for_module: Dict[str, Any] = {}
        for dep_result in dependencies_results:
            if dep_result and "module_id" in dep_result and "data" in dep_result: # Ensure result is valid
                 input_data_for_module[dep_result["module_id"]] = dep_result # Pass the whole result object
            # else:
            #     print(f"Warning: Dependency result for {module_id} is not in expected format: {dep_result}")


        print(f"Executing module: {module_id} with input from: {list(input_data_for_module.keys())}")
        return await module_instance.execute(input_data_for_module)

    async def execute_pipeline(self, pipeline_name: str = "default_crypto_pipeline") -> Dict:
        """
        Executes the defined pipeline of modules based on DAG.
        For now, pipeline_name is not used but could select different DAGs in the future.
        """
        print(f"Starting pipeline: {pipeline_name} in state: {self.current_state}")
        if not self.module_instances:
            raise Exception("No modules registered. Cannot execute pipeline.")

        # In a real FSM, state transitions would be more explicit here
        self.current_state = "COLLECTING" # Example transition
        print(f"Transitioning to state: {self.current_state}")

        # Topological sort of the DAG to get execution order
        # The DAG is defined with values as dependencies, so we need to reverse it for typical Kahn's
        # Or, adjust topological sort. The provided spec has `mod.analysis.ta_fa: ["mod.data.collector"]`
        # meaning ta_fa depends on data.collector.
        # The current topological_sort needs the graph in {node: [dependents]} form,
        # but the spec's DAG is {node: [dependencies]}. Let's adapt.

        # Adapting DAG for current topological_sort or vice-versa.
        # The current build_dag is {module: [dependencies]}.
        # The current topological_sort expects {dependent: [dependencies]}.
        # Let's ensure the topological sort handles the {module: [dependencies]} structure correctly.
        # The provided sort algorithm seems to expect {node: [outgoing_edges]} (i.e. node -> its dependents)
        # Let's try to make the topological sort work with {node: [dependencies]}

        # Re-thinking topological sort for {node: [dependencies]}
        # Kahn's algorithm:
        # 1. Compute in-degrees (number of dependencies for each node)
        # 2. Initialize queue with nodes having 0 in-degree (no dependencies)
        # 3. While queue is not empty:
        #    a. Dequeue a node `u` (add to sorted list)
        #    b. For each node `v` that depends on `u`:
        #       i. Decrement in-degree of `v`
        #       ii. If in-degree of `v` becomes 0, enqueue `v`

        # To implement 3.b, we need a way to find nodes `v` that depend on `u`.
        # This means we need an adjacency list where adj[u] = list of nodes that depend on u.
        # Let's build this reversed graph (dependents_graph) for the sort.

        adj: Dict[str, list[str]] = {u: [] for u in self.dag}
        in_degree: Dict[str, int] = {u: 0 for u in self.dag}

        for u, dependencies in self.dag.items():
            in_degree[u] = len(dependencies)
            for dep in dependencies:
                if dep in adj: # Ensure dependency is a valid module ID
                    adj[dep].append(u) # dep is a prerequisite for u
                else:
                    # This case implies a dependency is listed which is not itself a module in the DAG keys.
                    # This could be an issue with DAG definition or an external input not part of this DAG.
                    # For now, we assume all dependencies are also nodes in the DAG.
                    print(f"Warning: Dependency '{dep}' for module '{u}' not found as a module key in DAG. Execution might fail if it's expected as a module result.")


        queue = [u for u, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            u = queue.pop(0)
            execution_order.append(u)

            for v in adj[u]: # For each module v that depends on u
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(execution_order) != len(self.dag):
            # Identify nodes part of a cycle or missing
            missing_or_cycle = [node for node, degree in in_degree.items() if degree > 0]
            raise Exception(f"DAG has a cycle or missing dependencies, topological sort failed. Problematic nodes: {missing_or_cycle}")

        print(f"Pipeline execution order: {execution_order}")

        results: Dict[str, Any] = {}

        for module_id in execution_order:
            self.current_state = f"EXECUTING_{module_id.upper().replace('.', '_')}" # Update state
            print(f"Transitioning to state: {self.current_state} for module: {module_id}")

            # Gather results from dependencies
            dependencies_for_module = self.dag[module_id]
            dependency_outputs = [results[dep_id] for dep_id in dependencies_for_module if dep_id in results]

            module_result = await self.execute_module(module_id, dependency_outputs)
            results[module_id] = module_result

            if module_result.get("status") == "error":
                print(f"Error executing module {module_id}. Pipeline terminating. Error: {module_result.get('error_message')}")
                self.current_state = "ERROR"
                return {"status": "error", "message": f"Pipeline failed at {module_id}", "results": results}

        self.current_state = "REPORTING" # Or COMPLETED
        print(f"Pipeline {pipeline_name} completed successfully. Final state: {self.current_state}")
        return {"status": "success", "results": results}

    # Placeholder FSM handlers
    async def handle_idle(self, data: Any = None) -> None:
        print("State: IDLE. Waiting for tasks.")
        # Could trigger collection based on a schedule or external event
        pass

    async def handle_collecting(self, data: Any = None) -> None:
        print("State: COLLECTING. Initiating data collection.")
        # This would typically call the data collector module
        # results = await self.execute_specific_module("mod.data.collector", data)
        # self.current_state = "ANALYZING"
        pass

    async def handle_analyzing(self, data: Any = None) -> None:
        print("State: ANALYZING. Processing collected data.")
        # Calls analysis module
        # self.current_state = "RECOMMENDING"
        pass

    async def handle_recommending(self, data: Any = None) -> None:
        print("State: RECOMMENDING. Generating recommendations.")
        # Calls recommender module
        # self.current_state = "REPORTING"
        pass

    async def handle_reporting(self, data: Any = None) -> None:
        print("State: REPORTING. Compiling and disseminating reports.")
        # Calls report writer module
        # self.current_state = "IDLE" # Or a completion state
        pass

    async def set_state(self, new_state: str, data: Any = None) -> None:
        if new_state in self.states:
            self.current_state = new_state
            await self.states[new_state](data)
        else:
            print(f"Error: Unknown state {new_state}")

# Example Usage (for testing purposes, typically not here)
# async def main():
#     # --- This is just for demonstration, actual module classes would be imported ---
#     class MockModule(AIModuleBase): # AIModuleBase would need to be importable or defined
#         async def process(self, data: Dict) -> Dict:
#             print(f"MockModule {self.module_id} processing: {data}")
#             await asyncio.sleep(0.1) # Simulate work
#             return {"result": f"data from {self.module_id}"}
#         def validate_input(self, data: Dict) -> Dict: return data #dummy

#     # --- Setup for AIModuleBase if not imported ---
#     from datetime import datetime
#     class AIModuleBase:
#         def __init__(self, module_id: str, ai_service: str):
#             self.module_id = module_id; self.ai_service = ai_service; self.metadata = {}
#         async def execute(self, input_data: Dict):
#             validated_input = self.validate_input(input_data)
#             result = await self.process(validated_input)
#             return self.format_output(result)
#         def validate_input(self, data: Dict): raise NotImplementedError
#         async def process(self, data: Dict): raise NotImplementedError
#         def format_output(self, result: Dict):
#             return {"module_id": self.module_id, "timestamp": datetime.utcnow().isoformat(), "data": result, "metadata": self.metadata, "status": "success"}
#     # --- End AIModuleBase setup ---


#     controller = WorkflowController()
#     # Instantiate and register mock modules
#     data_collector = MockModule("mod.data.collector", "deepseek")
#     signal_analyzer = MockModule("mod.analysis.ta_fa", "chatgpt")
#     recommender = MockModule("mod.recommend.engine", "gemini")
#     reporter = MockModule("mod.report.writer", "claude")

#     await controller.register_module("mod.data.collector", data_collector)
#     await controller.register_module("mod.analysis.ta_fa", signal_analyzer)
#     await controller.register_module("mod.recommend.engine", recommender)
#     await controller.register_module("mod.report.writer", reporter)

#     # Execute the pipeline
#     pipeline_results = await controller.execute_pipeline()
#     print("\nPipeline Execution Results:")
#     import json
#     print(json.dumps(pipeline_results, indent=2))

# if __name__ == "__main__":
#    asyncio.run(main())
