import json
from typing import Dict, Any, List

from ..base import AIModuleBase
from ..core.brain import APIPoolManager, RateLimitError # Assuming brain.py is in core sibling directory

# Placeholder for a more sophisticated AI parsing and validation
def parse_json_from_ai(ai_response_text: str) -> Dict:
    """
    Attempts to parse JSON from AI response.
    Handles cases where AI might return text with JSON embedded.
    """
    try:
        # Find the start and end of the JSON block if it's embedded
        json_start = ai_response_text.find('{')
        json_end = ai_response_text.rfind('}') + 1

        if json_start != -1 and json_end != 0:
            json_str = ai_response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            # Try parsing the whole string if no clear markers
            return json.loads(ai_response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from AI response: {e}")
        print(f"AI Response Text was: {ai_response_text}")
        # Fallback or error structure
        return {"error": "Failed to parse AI response", "details": str(e), "raw_response": ai_response_text}


class RecommenderCore(AIModuleBase):
    def __init__(self):
        super().__init__(module_id="mod.recommend.engine", ai_service="gemini") # Primary AI: Gemini Pro
        # APIPoolManager is used for making calls to the LLM (Gemini, GPT-4o)
        self.ai_pool = APIPoolManager()
        self.metadata["recommendation_strategy"] = "ai_assisted_multi_factor"

    def validate_input(self, data: Dict) -> Dict:
        """
        Validates input data for RecommenderCore.
        Expects data from mod.analysis.ta_fa.
        """
        if "mod.analysis.ta_fa" not in data:
            raise ValueError("Input data must contain 'mod.analysis.ta_fa' results.")

        analyzer_output = data["mod.analysis.ta_fa"]
        if not isinstance(analyzer_output, dict):
            raise ValueError("'mod.analysis.ta_fa' data must be a dictionary.")
        if "data" not in analyzer_output or not isinstance(analyzer_output["data"], dict):
            raise ValueError("'mod.analysis.ta_fa' must have a 'data' dictionary.")
        if analyzer_output.get("status") == "error":
            raise ValueError("SignalAnalyzer module reported an error. Cannot make recommendations.")

        # Check for combined_signal in analysis data
        for symbol, analysis_data in analyzer_output["data"].items():
            if "combined_signal" not in analysis_data:
                 print(f"Warning: 'combined_signal' missing for symbol {symbol} in analyzer output.")

        return data # Return the full input_data

    async def process(self, data: Dict) -> Dict:
        """
        Core logic for generating recommendations based on analysis data using an AI model.
        Input `data` is expected to be: {"mod.analysis.ta_fa": {"module_id": ..., "data": ..., ...}}
        """
        analyzer_output = data.get("mod.analysis.ta_fa")
        if not analyzer_output or analyzer_output.get("status") == "error":
            return {"error": "SignalAnalyzer output missing or in error state."}

        analysis_data = analyzer_output.get("data", {})
        if not analysis_data:
            return {"error": "No data found in SignalAnalyzer output."}

        # Prepare AI prompt using data from SignalAnalyzer
        prompt = self.build_recommendation_prompt(analysis_data)

        ai_response_str: str = ""
        try:
            # Call AI (Gemini or fallback GPT-4o) using APIPoolManager
            # The primary AI is "gemini", backup is "gpt-4o"
            # execute_with_rotation will try "gemini" first. If it fails (e.g. all keys exhausted for gemini),
            # we might need a mechanism to switch to "gpt-4o".
            # For now, let's assume execute_with_rotation handles this by trying available keys from the specified pool.
            # The spec says "Gemini Pro (primary), GPT-4o (backup)". This implies a preference.
            # APIPoolManager's execute_with_rotation currently only takes one service name.
            # We might need to enhance APIPoolManager or handle primary/backup logic here.

            # Simplified primary/backup logic:
            services_to_try = ["gemini", "gpt-4o"] # As per spec (Gemini primary, GPT-4o backup)

            for service_name in services_to_try:
                try:
                    print(f"Attempting recommendation with AI service: {service_name}")
                    # `self.ai_service` is "gemini" by default from __init__
                    # We should use the service_name from the loop.
                    ai_response_str = await self.ai_pool.execute_with_rotation(
                        service=service_name,
                        task=lambda key: self.call_llm_api(prompt, key, service_name), # Pass service_name to know which API to call
                        max_retries=2 # Fewer retries per service, as we have a backup
                    )
                    if ai_response_str and not (isinstance(ai_response_str, dict) and ai_response_str.get("error")): # Check if response is valid
                        print(f"Successfully received response from {service_name}")
                        self.metadata["ai_service_used"] = service_name
                        break # Success, exit loop
                except Exception as e:
                    print(f"Failed to get recommendation from {service_name}: {e}")
                    if service_name == services_to_try[-1]: # If last service also failed
                        raise Exception(f"All AI services ({', '.join(services_to_try)}) failed for RecommenderCore. Last error: {e}")
                    # else, loop continues to the next service
            else: # If loop completes without break (no service succeeded)
                 raise Exception(f"All AI services ({', '.join(services_to_try)}) failed to provide a response.")


        except Exception as e:
            print(f"Error during AI call for recommendations: {str(e)}")
            return {"error": "AI service call failed", "details": str(e)}

        # Parse AI response (expected to be JSON)
        parsed_recommendations = parse_json_from_ai(ai_response_str)
        if "error" in parsed_recommendations:
             # AI response parsing failed
            return parsed_recommendations # Return the error dict from parse_json_from_ai

        # The spec output is just the list of recommendations, not nested under "recommendations" key.
        # However, the prompt asks for a "recommendations" key. Let's adjust.
        final_recommendations_list = parsed_recommendations.get("recommendations", [])
        if not isinstance(final_recommendations_list, list):
            print("Warning: AI response 'recommendations' key did not contain a list.")
            return {"error": "AI response format error: 'recommendations' not a list.", "raw_response": ai_response_str}


        # Add confidence scoring or cross-check with original analysis data
        # The prompt asks AI for confidence, but we can also refine it.
        scored_recommendations = self.add_confidence_scores_and_validate(final_recommendations_list, analysis_data)

        self.metadata["recommendations_count"] = len(scored_recommendations)
        return {"recommendations": scored_recommendations} # Output format per spec seems to be a direct list. Let's adjust.
                                                          # The spec shows Module 3 output as Dict with "recommendations" key holding a list.

    def build_recommendation_prompt(self, analysis_data: Dict) -> str:
        # analysis_data is Dict[symbol, analysis_dict]
        # We need to format this for the prompt.
        formatted_analysis = []
        for symbol, data in analysis_data.items():
            if data.get("status") == "error": # Skip symbols with errors from analysis
                continue
            formatted_analysis.append({
                "symbol": symbol,
                "combined_signal_score": data.get("combined_signal", {}).get("score"),
                "signal_action": data.get("combined_signal", {}).get("action"),
                "signal_strength": data.get("combined_signal", {}).get("strength"),
                "analysis_confidence": data.get("confidence"),
                "ta_highlights": {
                    "trend": data.get("ta_signals", {}).get("trend"),
                    "rsi": data.get("ta_signals", {}).get("rsi")
                },
                "fa_score": data.get("fa_score")
            })

        # The prompt from the spec:
        return f"""
        Analyze the following crypto market signal analysis and provide investment recommendations.
        The analysis includes a combined signal score (0-1, >0.6 BUY, <0.4 SELL), an action (BUY/SELL/HOLD),
        signal strength (0-1), and overall analysis confidence (0-1).

        Market Analysis Summary:
        {json.dumps(formatted_analysis, indent=2)}

        For each symbol in the Market Analysis Summary that does not have an error:
        1.  Symbol: (e.g., "BTC")
        2.  Investment Action: (Strictly BUY, SELL, or HOLD) - align with or refine the signal_action.
        3.  Confidence Level: (A float between 0.0 and 1.0) for your recommended action.
        4.  Time Horizon: (Strictly short_term, mid_term, or long_term)
        5.  Risk Assessment: (Strictly low, medium, or high)
        6.  Key Reasoning Points: (A list of 2-3 brief strings explaining the recommendation based on the provided analysis AND general crypto investment principles. For example, "Strong bullish TA momentum", "Positive FA outlook despite market volatility", "HOLD due to conflicting signals and high risk")

        IMPORTANT: Respond ONLY with a single well-formed JSON object.
        The JSON object should have a single key "recommendations", which is a list of recommendation objects.
        Each recommendation object in the list should strictly follow this format:
        {{
            "symbol": "string",
            "action": "string (BUY/SELL/HOLD)",
            "confidence": float (0.0-1.0),
            "time_horizon": "string (short_term/mid_term/long_term)",
            "risk_level": "string (low/medium/high)",
            "reasoning": ["string", "string", ...]
        }}

        Example for one symbol:
        {{
            "symbol": "BTC",
            "action": "BUY",
            "confidence": 0.85,
            "time_horizon": "mid_term",
            "risk_level": "medium",
            "reasoning": ["Technical breakout confirmed on daily chart", "Positive fundamental news catalyst expected"]
        }}
        Do not include any explanations or text outside of this JSON object.
        """

    async def call_llm_api(self, prompt: str, api_key: str, service_name: str) -> str:
        """
        Placeholder for calling the actual LLM API (Gemini, GPT-4o, etc.).
        """
        print(f"Calling {service_name} API with key {api_key[:5]}... (placeholder)")
        # Simulate API call and response
        await asyncio.sleep(0.5) # Simulate network latency

        # This is where you'd use `httpx` or `aiohttp` to make a real API call
        # For example, for Gemini:
        # headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # data = {"contents": [{"parts": [{"text": prompt}]}]}
        # response = await client.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", json=data, headers=headers)
        # response.raise_for_status()
        # return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Placeholder response mimicking the JSON structure requested in the prompt
        if service_name == "gemini" or service_name == "gpt-4o": # Generic good response
            # Construct a plausible response based on a dummy analysis_data structure
            # This is highly simplified. A real LLM would generate this.
            example_symbol = "BTC" # Assume analysis_data had BTC
            mock_recommendation = {
                "symbol": example_symbol,
                "action": "BUY",
                "confidence": 0.75,
                "time_horizon": "mid_term",
                "risk_level": "medium",
                "reasoning": [
                    f"Based on combined signal for {example_symbol}.",
                    "Market conditions appear favorable for mid-term hold."
                ]
            }
            # If there were other symbols in the prompt, add them too.
            # For now, just one example.
            return json.dumps({"recommendations": [mock_recommendation]})

        # Simulate a RateLimitError for testing the pool manager
        # if api_key == "key_that_causes_rate_limit": # Example condition
        #     raise RateLimitError(f"Simulated rate limit for {service_name} with key {api_key}")

        return json.dumps({"recommendations": []}) # Default empty if not handled


    def add_confidence_scores_and_validate(self, recommendations: List[Dict], analysis_data: Dict) -> List[Dict]:
        """
        Refines AI-generated confidence scores and validates recommendations.
        - Cross-references with original analysis confidence.
        - Ensures enums for action, time_horizon, risk_level are valid.
        """
        validated_recs = []
        valid_actions = {"BUY", "SELL", "HOLD"}
        valid_horizons = {"short_term", "mid_term", "long_term"}
        valid_risks = {"low", "medium", "high"}

        for rec in recommendations:
            symbol = rec.get("symbol")
            if not symbol or symbol not in analysis_data:
                print(f"Warning: Recommendation for unknown or errored symbol '{symbol}' skipped.")
                continue

            original_analysis_confidence = analysis_data[symbol].get("confidence", 0.5)
            ai_confidence = rec.get("confidence", 0.5)

            # Example: Blend AI confidence with original analysis confidence
            # This is a simple average, could be more sophisticated
            blended_confidence = (float(ai_confidence) + float(original_analysis_confidence)) / 2.0
            rec["confidence"] = round(blended_confidence, 4)

            # Validate enum fields
            if rec.get("action") not in valid_actions:
                print(f"Warning: Invalid action '{rec.get('action')}' for {symbol}. Defaulting to HOLD.")
                rec["action"] = "HOLD" # Default or handle as error

            if rec.get("time_horizon") not in valid_horizons:
                print(f"Warning: Invalid time_horizon '{rec.get('time_horizon')}' for {symbol}. Defaulting to mid_term.")
                rec["time_horizon"] = "mid_term"

            if rec.get("risk_level") not in valid_risks:
                print(f"Warning: Invalid risk_level '{rec.get('risk_level')}' for {symbol}. Defaulting to medium.")
                rec["risk_level"] = "medium"

            if not (0.0 <= rec["confidence"] <= 1.0):
                print(f"Warning: Confidence for {symbol} out of bounds ({rec['confidence']}). Clamping.")
                rec["confidence"] = max(0.0, min(1.0, rec["confidence"]))

            validated_recs.append(rec)

        return validated_recs

# Note: The spec's `call_gemini_api` is now generalized to `call_llm_api`
# and integrated with `APIPoolManager`'s `execute_with_rotation`.
# The `parse_ai_recommendations` is handled by `parse_json_from_ai`.
# `add_confidence_scores` is now `add_confidence_scores_and_validate`.
