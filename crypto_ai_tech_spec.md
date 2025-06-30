# Crypto AI System - Unified Technical Specification

## 🎯 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    mod.core.brain (FSM Orchestrator)           │
├─────────────────────────────────────────────────────────────────┤
│  API Pool Manager │ DAG Controller │ State Manager │ Retry Logic │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
    │ Data    │          │Analysis │          │Action   │
    │ Layer   │          │ Layer   │          │ Layer   │
    └─────────┘          └─────────┘          └─────────┘
```

## 🧠 Module 0: Core Brain (mod.core.brain)

**Primary Controller** - Điều phối toàn bộ hệ thống

### Core Components

#### 1. API Pool Manager
```python
class APIPoolManager:
    def __init__(self):
        self.pools = {
            "gemini": cycle(["key1", "key2", "key3"]),
            "openai": cycle(["gpt_key1", "gpt_key2"]),
            "anthropic": cycle(["claude_key1"])
        }
        self.cooldowns = {}
        self.rate_limits = {}
    
    async def get_available_key(self, service: str) -> str:
        key = next(self.pools[service])
        if self.is_key_available(key):
            return key
        return await self.wait_and_retry(service)
    
    async def execute_with_rotation(self, service: str, task: callable, max_retries=3):
        for attempt in range(max_retries):
            try:
                key = await self.get_available_key(service)
                return await task(key)
            except RateLimitError:
                await self.handle_cooldown(service)
                continue
        raise Exception(f"All {service} keys exhausted")
```

#### 2. FSM Workflow Controller
```python
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
    
    def build_dag(self):
        return {
            "mod.data.collector": [],
            "mod.analysis.ta_fa": ["mod.data.collector"],
            "mod.recommend.engine": ["mod.analysis.ta_fa"],
            "mod.report.writer": ["mod.recommend.engine"]
        }
    
    async def execute_pipeline(self, pipeline_name: str):
        execution_order = self.topological_sort(self.dag)
        results = {}
        
        for module_id in execution_order:
            dependencies = [results[dep] for dep in self.dag[module_id]]
            results[module_id] = await self.execute_module(module_id, dependencies)
        
        return results
```

#### 3. Standard Module Interface
```python
class AIModuleBase:
    def __init__(self, module_id: str, ai_service: str):
        self.module_id = module_id
        self.ai_service = ai_service
        self.metadata = {}
    
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
            return self.handle_error(e)
    
    def validate_input(self, data: Dict) -> Dict:
        """Override in each module"""
        pass
    
    async def process(self, data: Dict) -> Dict:
        """Override in each module - core processing logic"""
        pass
    
    def format_output(self, result: Dict) -> Dict:
        """Standard output format"""
        return {
            "module_id": self.module_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": result,
            "metadata": self.metadata,
            "status": "success"
        }
```

---

## 📊 Module 1: Data Collector (mod.data.collector)

**AI Assignment**: DeepSeek (primary), ChatGPT (backup)
**Extends**: AIModuleBase

### Implementation
```python
class DataCollector(AIModuleBase):
    def __init__(self):
        super().__init__("mod.data.collector", "deepseek")
        self.exchanges = self.init_exchanges()
        self.rate_limiter = RateLimiter(100, 60)  # 100 req/min
    
    def validate_input(self, data: Dict) -> Dict:
        schema = {
            "symbols": List[str],
            "timeframe": str,
            "data_types": List[str],
            "sources": List[str]
        }
        return validate_schema(data, schema)
    
    async def process(self, data: Dict) -> Dict:
        symbols = data["symbols"]
        tasks = []
        
        for symbol in symbols:
            task = self.collect_symbol_data(symbol, data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self.aggregate_results(symbols, results)
    
    async def collect_symbol_data(self, symbol: str, config: Dict) -> Dict:
        # Collect từ multiple sources
        price_data = await self.get_price_data(symbol)
        volume_data = await self.get_volume_data(symbol)
        orderbook = await self.get_orderbook(symbol)
        
        return {
            "symbol": symbol,
            "price_data": price_data,
            "volume_data": volume_data,
            "orderbook": orderbook,
            "collection_timestamp": datetime.utcnow().isoformat()
        }
```

### Input/Output Schema
```python
INPUT_SCHEMA = {
    "symbols": ["BTC", "ETH", "SOL"],
    "timeframe": "1h",
    "data_types": ["price", "volume", "orderbook"],
    "sources": ["binance", "okx"]
}

OUTPUT_SCHEMA = {
    "module_id": "mod.data.collector",
    "timestamp": "2025-01-01T00:00:00Z",
    "data": {
        "BTC": {
            "price_data": {"current": 45000, "24h_change": 0.05},
            "volume_data": {"24h_volume": 1000000},
            "orderbook": {"bids": [...], "asks": [...]}
        }
    },
    "metadata": {
        "collection_time": 1.2,
        "sources_status": {"binance": "ok", "okx": "ok"}
    },
    "status": "success"
}
```

---

## 🧮 Module 2: Signal Analyzer (mod.analysis.ta_fa)

**AI Assignment**: ChatGPT (primary), DeepSeek (backup)
**Extends**: AIModuleBase

### Implementation
```python
class SignalAnalyzer(AIModuleBase):
    def __init__(self):
        super().__init__("mod.analysis.ta_fa", "chatgpt")
        self.ta_engine = TechnicalAnalysisEngine()
        self.fa_engine = FundamentalAnalysisEngine()
    
    async def process(self, data: Dict) -> Dict:
        collector_data = data["mod.data.collector"]["data"]
        
        analysis_results = {}
        
        for symbol, market_data in collector_data.items():
            # Technical Analysis
            ta_signals = await self.ta_engine.analyze(market_data)
            
            # Fundamental Analysis
            fa_score = await self.fa_engine.analyze(symbol)
            
            # Combine signals
            combined_signal = self.combine_signals(ta_signals, fa_score)
            
            analysis_results[symbol] = {
                "ta_signals": ta_signals,
                "fa_score": fa_score,
                "combined_signal": combined_signal,
                "confidence": self.calculate_confidence(ta_signals, fa_score)
            }
        
        return analysis_results
    
    def combine_signals(self, ta_signals: Dict, fa_score: float) -> Dict:
        # Weighted combination logic
        ta_weight = 0.6
        fa_weight = 0.4
        
        combined_score = (ta_signals["overall_score"] * ta_weight + 
                         fa_score * fa_weight)
        
        return {
            "score": combined_score,
            "signal": "BUY" if combined_score > 0.6 else "SELL" if combined_score < 0.4 else "HOLD",
            "strength": abs(combined_score - 0.5) * 2
        }
```

---

## 💡 Module 3: Recommender Core (mod.recommend.engine)

**AI Assignment**: Gemini Pro (primary), GPT-4o (backup)
**Extends**: AIModuleBase

### Implementation với AI Integration
```python
class RecommenderCore(AIModuleBase):
    def __init__(self):
        super().__init__("mod.recommend.engine", "gemini")
        self.ai_pool = APIPoolManager()
    
    async def process(self, data: Dict) -> Dict:
        analysis_data = data["mod.analysis.ta_fa"]["data"]
        
        # Prepare AI prompt
        prompt = self.build_recommendation_prompt(analysis_data)
        
        # Call AI with rotation
        ai_response = await self.ai_pool.execute_with_rotation(
            "gemini", 
            lambda key: self.call_gemini_api(prompt, key)
        )
        
        # Parse AI response
        recommendations = self.parse_ai_recommendations(ai_response)
        
        # Add confidence scoring
        scored_recommendations = self.add_confidence_scores(recommendations, analysis_data)
        
        return scored_recommendations
    
    def build_recommendation_prompt(self, analysis_data: Dict) -> str:
        return f"""
        Analyze the following crypto market data and provide investment recommendations:
        
        Market Analysis: {json.dumps(analysis_data, indent=2)}
        
        For each symbol, provide:
        1. Investment action (BUY/SELL/HOLD)
        2. Confidence level (0-1)
        3. Time horizon (short/mid/long term)
        4. Risk assessment
        5. Key reasoning points
        
        Respond in JSON format:
        {{
            "recommendations": [
                {{
                    "symbol": "BTC",
                    "action": "BUY",
                    "confidence": 0.85,
                    "time_horizon": "mid_term",
                    "risk_level": "medium",
                    "reasoning": ["technical breakout", "positive fundamentals"]
                }}
            ]
        }}
        """
    
    async def call_gemini_api(self, prompt: str, api_key: str) -> str:
        # Implement Gemini API call
        pass
```

---

## 📝 Module 4: Report Writer (mod.report.writer)

**AI Assignment**: Claude (primary), Gemini Pro (backup)
**Extends**: AIModuleBase

### Implementation
```python
class ReportWriter(AIModuleBase):
    def __init__(self):
        super().__init__("mod.report.writer", "claude")
        self.template_engine = ReportTemplateEngine()
    
    async def process(self, data: Dict) -> Dict:
        recommendations = data["mod.recommend.engine"]["data"]
        
        # Generate different report formats
        reports = {}
        
        # Executive Summary
        reports["executive_summary"] = await self.generate_executive_summary(recommendations)
        
        # Detailed Analysis
        reports["detailed_analysis"] = await self.generate_detailed_analysis(data)
        
        # Actionable Recommendations
        reports["actionable_recommendations"] = await self.generate_actionable_recommendations(recommendations)
        
        return reports
    
    async def generate_executive_summary(self, recommendations: Dict) -> str:
        prompt = f"""
        Create a concise executive summary based on these crypto investment recommendations:
        
        {json.dumps(recommendations, indent=2)}
        
        Write in Vietnamese, professional investment report style.
        Focus on key opportunities and risks.
        Maximum 200 words.
        """
        
        return await self.ai_pool.execute_with_rotation(
            "claude",
            lambda key: self.call_claude_api(prompt, key)
        )
```

---

## 🔄 Development Workflow

### Phase 1: Base Infrastructure
```bash
# 1. Create base module structure
mkdir crypto_ai_system
cd crypto_ai_system

# 2. Setup base classes
touch modules/base.py
touch modules/brain.py

# 3. Create individual modules
mkdir modules/{data,analysis,recommend,report}
```

### Phase 2: Module Development (Parallel)
```python
# Each AI gets assigned specific modules
ASSIGNMENTS = {
    "deepseek": ["mod.data.collector"],
    "chatgpt": ["mod.analysis.ta_fa"],
    "gemini": ["mod.recommend.engine"],
    "claude": ["mod.report.writer"]
}
```

### Phase 3: Integration Testing
```python
# Test individual modules
pytest modules/data/test_collector.py
pytest modules/analysis/test_ta_fa.py

# Test integration
pytest tests/test_full_pipeline.py
```

### Phase 4: Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  brain:
    build: ./modules/brain
    environment:
      - GEMINI_KEYS=${GEMINI_KEYS}
      - OPENAI_KEYS=${OPENAI_KEYS}
  
  data-collector:
    build: ./modules/data
    depends_on:
      - brain
```

---

## 🎯 Key Improvements từ merge

**1. Standardized Module Interface**
- Base class cho tất cả modules
- Consistent input/output format
- Error handling chuẩn

**2. Smart API Management**
- Token pool rotation
- Automatic cooldown handling
- Fallback strategies

**3. FSM + DAG Architecture**
- Clear state management
- Dependency resolution
- Parallel execution optimization

**4. Production-Ready Features**
- Docker containerization
- Comprehensive testing
- Monitoring & alerting

**5. Multi-AI Coordination**
- Clear responsibility boundaries
- Backup AI assignments
- Load balancing

---

## 📋 Final Module Assignment

| Module ID | Primary AI | Backup AI | Complexity |
|-----------|------------|-----------|------------|
| mod.core.brain | Human/Framework | - | Core |
| mod.data.collector | DeepSeek | ChatGPT | Medium |
| mod.analysis.ta_fa | ChatGPT | DeepSeek | High |
| mod.recommend.engine | Gemini Pro | GPT-4o | High |
| mod.report.writer | Claude | Gemini Pro | Medium |

Ready to start development với standardized approach này?