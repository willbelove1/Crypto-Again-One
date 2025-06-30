import json
from typing import Dict, Any, List

from ..base import AIModuleBase
from ..core.brain import APIPoolManager, RateLimitError # For AI calls

# Placeholder for a more sophisticated template engine
class ReportTemplateEngine:
    def __init__(self):
        print("ReportTemplateEngine initialized (placeholder).")

    def render(self, template_name: str, context: Dict) -> str:
        """Renders a report using a named template and context data."""
        print(f"Rendering template '{template_name}' with context keys: {list(context.keys())} (placeholder)")
        if template_name == "executive_summary_template":
            # Basic rendering for placeholder
            recs_summary = []
            for rec in context.get("recommendations", []):
                recs_summary.append(
                    f"- {rec.get('symbol')}: {rec.get('action')} (Confidence: {rec.get('confidence', 0):.2f}, Risk: {rec.get('risk_level', 'N/A')})"
                )
            return f"Executive Summary:\n" + "\n".join(recs_summary) + "\nKey Opportunities: ... Key Risks: ..."

        elif template_name == "detailed_analysis_template":
            # context here would include full analysis_data and recommendations
            return f"Detailed Analysis Report for symbols: {list(context.get('analysis_data', {}).keys())}\n..."

        elif template_name == "actionable_recommendations_template":
            # context includes recommendations
            return f"Actionable Recommendations:\n" + json.dumps(context.get("recommendations", []), indent=2)

        return "Template not found or rendering failed (placeholder)."

class ReportWriter(AIModuleBase):
    def __init__(self):
        super().__init__(module_id="mod.report.writer", ai_service="claude") # Primary AI: Claude
        self.template_engine = ReportTemplateEngine()
        self.ai_pool = APIPoolManager() # For AI-assisted report generation (e.g., summaries)
        self.metadata["report_formats"] = ["executive_summary", "detailed_analysis", "actionable_recommendations"]

    def validate_input(self, data: Dict) -> Dict:
        """
        Validates input data for ReportWriter.
        Expects data from mod.recommend.engine and potentially mod.analysis.ta_fa.
        The DAG in WorkflowController has report.writer depending on recommend.engine and analysis.ta_fa
        So, input `data` will be:
        {
            "mod.recommend.engine": { ... recommendation data ... },
            "mod.analysis.ta_fa": { ... analysis data ... }
        }
        """
        if "mod.recommend.engine" not in data:
            raise ValueError("Input data must contain 'mod.recommend.engine' results.")
        if "mod.analysis.ta_fa" not in data: # As per updated DAG
            raise ValueError("Input data must contain 'mod.analysis.ta_fa' results for detailed reports.")

        recommend_output = data["mod.recommend.engine"]
        analysis_output = data["mod.analysis.ta_fa"]

        if not isinstance(recommend_output, dict) or recommend_output.get("status") == "error":
            raise ValueError("RecommenderCore output missing, not a dict, or in error state.")
        if "data" not in recommend_output or not isinstance(recommend_output["data"], dict):
             raise ValueError("RecommenderCore 'data' missing or not a dict.")
        if "recommendations" not in recommend_output["data"] or not isinstance(recommend_output["data"]["recommendations"], list):
            raise ValueError("RecommenderCore 'data.recommendations' missing or not a list.")

        if not isinstance(analysis_output, dict) or analysis_output.get("status") == "error":
            raise ValueError("SignalAnalyzer output missing, not a dict, or in error state.")
        if "data" not in analysis_output or not isinstance(analysis_output["data"], dict):
            raise ValueError("SignalAnalyzer 'data' missing or not a dict for detailed reports.")

        return data

    async def process(self, data: Dict) -> Dict:
        """
        Core logic for generating various reports from recommendation and analysis data.
        """
        recommend_data_wrapper = data.get("mod.recommend.engine", {})
        analysis_data_wrapper = data.get("mod.analysis.ta_fa", {})

        # Extract the actual list of recommendations and dict of analysis results
        recommendations_list: List[Dict] = recommend_data_wrapper.get("data", {}).get("recommendations", [])
        analysis_by_symbol: Dict[str, Any] = analysis_data_wrapper.get("data", {})

        if not recommendations_list:
            print("Warning: No recommendations provided to ReportWriter.")
            # Return empty reports or a specific status
            return {
                "executive_summary": "No recommendations available to summarize.",
                "detailed_analysis": "No data available for detailed analysis.",
                "actionable_recommendations": "No actionable recommendations."
            }

        reports: Dict[str, Any] = {}

        # 1. Executive Summary (AI-assisted)
        # The spec shows this uses Claude (primary) or Gemini Pro (backup)
        reports["executive_summary"] = await self.generate_executive_summary(recommendations_list)

        # 2. Detailed Analysis Report (Template-based, using analysis and recommendations)
        # This would combine insights from analysis_by_symbol and link them to recommendations_list
        reports["detailed_analysis"] = await self.generate_detailed_analysis(recommendations_list, analysis_by_symbol)

        # 3. Actionable Recommendations List (Formatted list of recommendations)
        reports["actionable_recommendations"] = await self.generate_actionable_recommendations(recommendations_list)

        self.metadata["generated_reports_count"] = len(reports)
        return reports # This is the "data" part of the AIModuleBase output

    async def generate_executive_summary(self, recommendations: List[Dict]) -> str:
        """Generates a concise executive summary using an LLM."""
        if not recommendations:
            return "No recommendations to summarize."

        prompt = f"""
        Bạn là một chuyên gia phân tích đầu tư tiền điện tử.
        Dựa trên các khuyến nghị đầu tư sau đây, hãy viết một bản tóm tắt điều hành (executive summary) súc tích bằng tiếng Việt.
        Báo cáo nên chuyên nghiệp, tập trung vào các cơ hội và rủi ro chính.
        Giới hạn trong khoảng 150-200 từ.

        Dữ liệu khuyến nghị:
        {json.dumps(recommendations, indent=2, ensure_ascii=False)}

        Bắt đầu bản tóm tắt của bạn.
        """

        ai_response_str = "Lỗi khi tạo tóm tắt." # Default error message
        services_to_try = ["claude", "gemini"] # Claude primary, Gemini backup (as per spec)

        for service_name in services_to_try:
            try:
                print(f"Attempting executive summary generation with AI service: {service_name}")
                ai_response_str = await self.ai_pool.execute_with_rotation(
                    service=service_name,
                    task=lambda key: self.call_llm_for_summary(prompt, key, service_name),
                    max_retries=2
                )
                if ai_response_str and not (isinstance(ai_response_str, dict) and ai_response_str.get("error")):
                    print(f"Successfully received summary from {service_name}")
                    self.metadata["executive_summary_ai_service"] = service_name
                    return ai_response_str # Return the generated summary text

            except Exception as e:
                print(f"Failed to generate executive summary with {service_name}: {e}")
                if service_name == services_to_try[-1]: # Last service failed
                    ai_response_str = f"Lỗi: Không thể tạo tóm tắt điều hành sau khi thử các dịch vụ AI. Lỗi cuối: {e}"
                # else, loop continues

        return ai_response_str # Return last error or successful response

    async def call_llm_for_summary(self, prompt: str, api_key: str, service_name: str) -> str:
        """Placeholder for calling Claude or Gemini API for summary generation."""
        print(f"Calling {service_name} API for summary with key {api_key[:5]}... (placeholder)")
        await asyncio.sleep(0.3) # Simulate network latency

        # Example placeholder response
        if service_name == "claude":
            return (
                "Tóm tắt điều hành:\n"
                "Thị trường tiền điện tử hiện tại cho thấy một số cơ hội đầu tư tiềm năng ở BTC và ETH với tín hiệu MUA rõ ràng, "
                "đi kèm mức độ tự tin trung bình đến cao và rủi ro vừa phải, phù hợp cho đầu tư trung hạn. "
                "Tuy nhiên, cần thận trọng với các altcoin khác có thể biến động mạnh. "
                "Nhà đầu tư nên xem xét đa dạng hóa danh mục và quản lý rủi ro chặt chẽ."
            )
        elif service_name == "gemini":
             return (
                "Tóm lược khuyến nghị:\n"
                "Các phân tích gần đây chỉ ra cơ hội mua vào đối với Bitcoin (BTC) và Ethereum (ETH) trong trung hạn, "
                "dựa trên các chỉ báo kỹ thuật và điểm số cơ bản tích cực. Mức độ rủi ro được đánh giá ở mức trung bình. "
                "Đối với các tài sản khác, khuyến nghị là nắm giữ hoặc cần theo dõi thêm. "
                "Nhà đầu tư được khuyên nên cân nhắc kỹ lưỡng khẩu vị rủi ro cá nhân."
            )
        # Simulate RateLimitError for testing
        # if api_key == "claude_key_rate_limited":
        #     raise RateLimitError(f"Simulated rate limit for {service_name} with key {api_key}")
        return "Không thể tạo tóm tắt từ dịch vụ AI (placeholder)."


    async def generate_detailed_analysis(self, recommendations: List[Dict], analysis_data: Dict[str, Any]) -> str:
        """Generates a detailed analysis report using the template engine."""
        if not analysis_data and not recommendations:
            return "No analysis data or recommendations available for detailed report."

        # Context for the template: combine recommendations with their underlying analysis
        report_context: Dict[str, Any] = {
            "title": "Detailed Crypto Investment Analysis Report",
            "generated_at": datetime.utcnow().isoformat(),
            "symbols_analysis": []
        }

        recommendations_map = {rec["symbol"]: rec for rec in recommendations}

        for symbol, an_data in analysis_data.items():
            if an_data.get("status") == "error": continue # Skip errored analysis

            symbol_detail = {
                "symbol": symbol,
                "analysis": an_data, # Full analysis data (TA, FA, combined signal, confidence)
                "recommendation": recommendations_map.get(symbol, {"action": "N/A", "reasoning": ["No specific recommendation generated."]})
            }
            report_context["symbols_analysis"].append(symbol_detail)

        # Use template engine (placeholder for now)
        return self.template_engine.render("detailed_analysis_template", report_context)

    async def generate_actionable_recommendations(self, recommendations: List[Dict]) -> str:
        """Generates a report focusing on actionable recommendations."""
        if not recommendations:
            return "No actionable recommendations available."

        # Context for the template
        report_context = {
            "title": "Actionable Crypto Investment Recommendations",
            "generated_at": datetime.utcnow().isoformat(),
            "recommendations": recommendations
        }
        # Use template engine (placeholder for now)
        # This might just be a formatted JSON or a more user-friendly text.
        # The spec's example output for ReportWriter is just the string content.
        # So, this method should return a string.
        formatted_recs_str = self.template_engine.render("actionable_recommendations_template", report_context)
        return formatted_recs_str

import asyncio # For placeholder call_llm_for_summary if run directly
from datetime import datetime # For generate_detailed_analysis placeholder context
