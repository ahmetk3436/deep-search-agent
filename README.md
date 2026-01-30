# 🚀 Multi-Platform Deep Research Agent

A production-ready, intelligent research agent that automatically routes queries to the best search engine based on the nature of the request. Built with **LangGraph**, **DeepSeek-R1**, and **GLM-4.7**.

## 🧠 Architecture

### Core Components

1. **The Brain - DeepSeek-R1** (Router/Planner)
   - Analyzes user queries
   - Intelligently selects the best search tool
   - Generates optimized search queries
   - Decides when research is complete

2. **The Eyes - Multi-Platform Search Tools**
   - **Exa.ai**: For academic papers, technical documentation, PDFs, whitepapers
   - **Tavily**: For breaking news, market data, current events
   - **Serper (Google)**: For forums, Reddit threads, obscure content

3. **The Writer - GLM-4.7** (Synthesizer)
   - Consolidates all gathered research
   - Creates comprehensive, academic-grade reports
   - Uses proper citations and formatting

## ✨ Features

- 🎯 **Intelligent Routing**: Automatically selects the best search engine for each query type
- 🔄 **Multi-Iteration Research**: Performs up to 5 research rounds to gather comprehensive information
- 📚 **Vendor Lock-in Prevention**: Uses multiple search providers for diverse perspectives
- 💰 **Cost-Efficient**: DeepSeek for routing (cheap), GLM for writing (high quality)
- 🛡️ **Loop Protection**: Built-in safeguards prevent infinite research loops
- 📊 **Professional Reports**: Generates structured Markdown reports with citations

## 📋 Prerequisites

- Python 3.9 or higher
- API keys for the following services (see `.env` setup below)

## 🔧 Installation

### 1. Clone or download the project

```bash
cd deep-search-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```env
# Brain - DeepSeek-R1
DEEPSEEK_API_KEY=sk-your-deepseek-api-key

# Writer - GLM-4.7 (ZhipuAI)
ZHIPUAI_API_KEY=your-glm-api-key

# Search Tools
TAVILY_API_KEY=tvly-your-tavily-api-key
EXA_API_KEY=your-exa-api-key
SERPER_API_KEY=your-serper-api-key
```

### 4. Get API Keys

- **DeepSeek**: https://platform.deepseek.com/
- **ZhipuAI (GLM-4.7)**: https://open.bigmodel.cn/
- **Tavily**: https://tavily.com/
- **Exa.ai**: https://exa.ai/
- **Serper**: https://serper.dev/

## 🚀 Usage

### Run with Interactive Input

```bash
python main.py
```

You'll be prompted to enter your research query.

### Run with Command Line Argument

```bash
python main.py "Your research question here"
```

### Example Queries

**Academic/Technical Query** (routes to Exa.ai):
```bash
python main.py "GeoAI alanında son çıkan LoRA adaptör teknikleri üzerine makaleleri bul"
```

**News/Finance Query** (routes to Tavily):
```bash
python main.py "Bugün Bitcoin neden düştü?"
```

**Broad/Niche Query** (routes to Serper):
```bash
python main.py "Reddit'te en popüler yapay zeka projeleri neler?"
```

## 📊 How It Works

### Workflow

1. **Router Analysis**: DeepSeek-R1 analyzes the query type
2. **Tool Selection**: Intelligently routes to the best search engine
3. **Search Execution**: Executes searches with optimized queries
4. **Context Accumulation**: Gathers results across multiple iterations
5. **Synthesis**: GLM-4.7 creates a comprehensive report with citations
6. **Output**: Delivers a professional, well-structured Markdown report

### Tool Selection Logic

| Query Type | Selected Tool | Reason |
|------------|---------------|---------|
| Academic papers, technical docs | **Exa.ai** | Deep semantic search, PDF access |
| Breaking news, market data | **Tavily** | Real-time, news-focused |
| Forums, Reddit, obscure content | **Serper** | Broad web index coverage |

## 📁 Project Structure

```
deep-search-agent/
├── main.py              # Main application with LangGraph
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .env                # Your API keys (create this)
└── README.md           # This file
```

## 🎯 Key Features Explained

### Intelligent Routing

The agent doesn't just search - it **thinks** about where to search:

- Detects query type (technical, news, general)
- Selects the most appropriate search engine
- Generates optimized search queries for each platform
- Adapts search strategy based on gathered results

### Multi-Iteration Research

Unlike single-shot search tools, this agent:

- Performs up to 5 research iterations
- Refines queries based on previous results
- Accumulates context across multiple searches
- Decides when sufficient information is gathered

### Professional Report Generation

GLM-4.7 creates reports with:

- Executive Summary
- Background/Context
- Key Findings
- Detailed Analysis
- Conclusions & Recommendations
- Proper citations [Source X]

## 🛠️ Technical Details

- **Orchestrator**: LangGraph StateGraph
- **State Management**: Typed Pydantic models
- **LLM Integration**: OpenAI-compatible API clients
- **Error Handling**: Graceful fallbacks and error messages
- **Type Safety**: Full type hints throughout

## 🔍 Example Output

```
================================================================================
🚀 MULTI-PLATFORM DEEP RESEARCH AGENT
================================================================================

📝 Query: GeoAI alanında son çıkan LoRA adaptör teknikleri üzerine makaleleri bul

================================================================================
⚙️  STARTING RESEARCH WORKFLOW
================================================================================

🧠 [Router] Analyzing query (Step 1/5)...
   → Selected Tool: EXA
   → Queries: ['GeoAI LoRA adaptation techniques 2024', 'geospatial AI LoRA fine-tuning methods']
   → Reasoning: User is asking for academic papers and technical research on LoRA adaptation in GeoAI. Exa is best suited for finding academic papers and technical documentation.

📚 [Exa.ai] Searching for technical/academic content...
   → Query: GeoAI LoRA adaptation techniques 2024
   → Query: geospatial AI LoRA fine-tuning methods
   → ✓ Found 10 results from Exa

🧠 [Router] Analyzing query (Step 2/5)...
   → Selected Tool: EXA
   → Queries: ['LoRA fine-tuning remote sensing satellite imagery', 'adapter techniques geospatial deep learning']
   → Reasoning: Gathering more specific information on LoRA applications in remote sensing and geospatial deep learning.

...

✍️  [GLM-4.7] Synthesizing research into final report...
   → Processing 15432 characters of research data...
   → ✓ Report generated successfully!

================================================================================
📊 FINAL REPORT
================================================================================

# Executive Summary

This report provides a comprehensive analysis of the latest LoRA (Low-Rank Adaptation) techniques in the field of GeoAI...

[Full report continues...]

================================================================================
✅ RESEARCH COMPLETE
================================================================================
📈 Total research iterations: 3
📚 Total sources gathered: 15
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License

## 🙏 Acknowledgments

- DeepSeek for the powerful reasoning model
- ZhipuAI for the excellent GLM-4.7 writing model
- LangChain team for the amazing framework
- LangGraph for the workflow orchestration

---

**Built with ❤️ for deep, intelligent research**