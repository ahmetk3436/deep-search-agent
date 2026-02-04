"""
Multi-Platform Deep Research Agent
===================================
An intelligent research agent that routes queries to best search engine
based on nature of request using LangGraph.

Architecture:
- Brain: DeepSeek-R1 (Router/Planner)
- Eyes: Exa.ai, Tavily, Serper (Search Tools)
- Writer: GLM-4.7 (Synthesizer)
"""

import os
from typing import TypedDict, List, Literal, Annotated, Optional, Tuple
from operator import add
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from exa_py import Exa
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class AgentState(TypedDict):
    """State for the research agent workflow."""
    query: str  # Original user query
    research_context: Annotated[List[str], add]  # Accumulated search results
    selected_tool: Optional[str]  # Which search engine to use: "exa", "tavily", "serper"
    search_queries: Optional[List[str]]  # Queries to search for
    step_count: int  # Number of research iterations
    messages: List[object]  # Conversation history
    forced_providers: Optional[List[str]]  # Force specific providers (bypass router)
    max_iterations: int  # Maximum research iterations


# ============================================================================
# MODELS INITIALIZATION
# ============================================================================

# DeepSeek-R1 - The Brain (Router/Planner)
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1
)

# Writer - Using DeepSeek for now (GLM-4.7 has credit issues)
# Change back to GLM-4.7 when you have API credit
writer_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7
)


# ============================================================================
# TRANSLATION FUNCTION
# ============================================================================

def detect_and_translate_query(query: str) -> Tuple[str, bool]:
    """
    Detect if query is non-English and translate to English.
    
    Returns:
        (translated_query, was_translated)
    """
    # Simple check for non-ASCII characters
    is_non_english = any(ord(char) > 127 for char in query)
    
    if not is_non_english:
        return query, False
    
    # Translate to English using LLM
    translator_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Translate the given text to English accurately. Return ONLY the translation, nothing else."),
        ("human", "{text}")
    ])
    
    try:
        response = deepseek_llm.invoke(translator_prompt.format_messages(text=query))
        translated = response.content.strip()
        print(f"   🌐 Translated: '{query}' → '{translated}'")
        return translated, True
    except Exception as e:
        print(f"   ⚠️ Translation failed: {e}, using original query")
        return query, False


# ============================================================================
# ROUTER NODE - The Brain
# ============================================================================

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent research router. Analyze the user's query and decide which search engine is best suited.

Available Tools:
1. **exa** - Use for: Academic papers, technical documentation, PDFs, whitepapers, deep technical research
2. **tavily** - Use for: Breaking news, market data, general summaries, current events
3. **serper** - Use for: Obscure forums, Reddit threads, niche content, broad web search

Output Format (JSON):
{{
    "selected_tool": "exa" | "tavily" | "serper",
    "search_queries": ["query1", "query2", ...],
    "reasoning": "Brief explanation of why this tool was chosen",
    "should_continue": true
}}

Rules:
- Provide 2-4 specific search queries
- Set should_continue=false if enough information has been gathered
- Respect the maximum iteration count shown in the step counter
"""),
    ("human", "Query: {query}\n\nCurrent Research Context (if any):\n{context}\n\nStep: {step_count}/{max_iterations}")
])


def router_node(state: AgentState) -> AgentState:
    """
    Router node that analyzes the query and decides which search tool to use.
    Powered by DeepSeek-R1. Supports forced provider mode.
    """
    max_iter = state.get("max_iterations", 5)
    forced = state.get("forced_providers")

    print(f"\n🧠 [Router] Analyzing query (Step {state['step_count']}/{max_iter})...")

    # Detect and translate if non-English
    translated_query, was_translated = detect_and_translate_query(state["query"])

    # Store translated query if not already stored
    if was_translated and "translated_query" not in state:
        state["translated_query"] = translated_query

    # Use translated query for search if available, otherwise original
    search_query = state.get("translated_query", state["query"])

    # Format current context
    context_str = "\n".join(state.get("research_context", []))[-3000:] if state.get("research_context") else "No previous research."

    # --- FORCED PROVIDER MODE ---
    if forced:
        # Round-robin through forced providers
        provider_index = (state["step_count"] - 1) % len(forced)
        selected = forced[provider_index]

        print(f"   🔒 [Forced Mode] Using provider: {selected.upper()} (round-robin {provider_index + 1}/{len(forced)})")
        print(f"   📋 Provider rotation: {' → '.join(p.upper() for p in forced)}")

        # Still use DeepSeek to generate good search queries
        query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a search query generator. Given a research query and existing context, generate 2-4 specific search queries optimized for the search engine: {provider}.

Provider strengths:
- exa: Academic papers, technical documentation, PDFs, whitepapers
- tavily: Breaking news, market data, current events, advanced search
- serper: Broad web search, forums, Reddit, niche content, Google results

Output Format (JSON):
{{
    "search_queries": ["query1", "query2", ...],
    "should_continue": true
}}

Rules:
- Generate diverse, specific queries
- Optimize query style for the target search engine
- Set should_continue=false if enough information has been gathered"""),
            ("human", "Query: {query}\n\nTarget Provider: {provider}\n\nCurrent Research Context:\n{context}\n\nStep: {step_count}/{max_iterations}")
        ])

        try:
            response = deepseek_llm.invoke(
                query_gen_prompt.format_messages(
                    query=search_query,
                    provider=selected,
                    context=context_str,
                    step_count=state["step_count"],
                    max_iterations=max_iter
                )
            )

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decision = json.loads(content)
            state["search_queries"] = decision["search_queries"]
            should_continue = decision.get("should_continue", True)

            print(f"   → Queries: {state['search_queries']}")

        except Exception as e:
            print(f"   ⚠️ Query generation error: {e}, using original query")
            state["search_queries"] = [search_query]
            should_continue = True

        state["selected_tool"] = selected
        state["step_count"] = state["step_count"] + 1

        if not should_continue or state["step_count"] > max_iter:
            print("   → ✓ Research complete, moving to writer...")
            state["selected_tool"] = "finish"

        return state

    # --- NORMAL AUTO-ROUTING MODE ---
    response = deepseek_llm.invoke(
        router_prompt.format_messages(
            query=search_query,
            context=context_str,
            step_count=state["step_count"],
            max_iterations=max_iter
        )
    )

    try:
        # Parse JSON response
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        decision = json.loads(content)

        print(f"   → Selected Tool: {decision['selected_tool'].upper()}")
        print(f"   → Queries: {decision['search_queries']}")
        print(f"   → Reasoning: {decision['reasoning']}")

        # Update state
        state["selected_tool"] = decision["selected_tool"]
        state["search_queries"] = decision["search_queries"]
        state["step_count"] = state["step_count"] + 1

        # Check if we should continue or finish
        if not decision.get("should_continue", True) or state["step_count"] > max_iter:
            print("   → ✓ Research complete, moving to writer...")
            state["selected_tool"] = "finish"

    except Exception as e:
        print(f"   ⚠️ Error parsing router response: {e}")
        # Default fallback
        state["selected_tool"] = "tavily"
        state["search_queries"] = [state["query"]]

    return state


# ============================================================================
# SEARCH NODES - The Eyes
# ============================================================================

def exa_search_node(state: AgentState) -> AgentState:
    """
    Exa.ai node for technical/academic research.
    Best for finding papers, PDFs, technical documentation.
    """
    print(f"\n📚 [Exa.ai] Searching for technical/academic content...")
    
    try:
        exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        
        all_results = []
        for query in state["search_queries"]:
            print(f"   → Query: {query}")
            results = exa.search(
                query=query,
                type="auto",
                num_results=5
            )
            
            for i, result in enumerate(results.results):
                result_text = f"\n[Exa Result {i+1}] Source: {result.title}\n"
                result_text += f"URL: {result.url}\n"
                if hasattr(result, 'text') and result.text:
                    result_text += f"Content: {result.text[:800]}...\n"
                elif hasattr(result, 'highlights') and result.highlights:
                    result_text += f"Highlights: {result.highlights[:400]}...\n"
                all_results.append(result_text)
        
        state["research_context"].extend(all_results)
        print(f"   → ✓ Found {len(all_results)} results from Exa")
        
    except Exception as e:
        print(f"   ⚠️ Exa search error: {e}")
        state["research_context"].append(f"[Exa Error]: {str(e)}")
    
    return state


def tavily_search_node(state: AgentState) -> AgentState:
    """
    Tavily node for news and general research.
    Best for breaking news, market data, current events.
    """
    print(f"\n📰 [Tavily] Searching for current information...")
    
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        all_results = []
        for query in state["search_queries"]:
            print(f"   → Query: {query}")
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_raw_content=True
            )
            
            # Format results
            for i, result in enumerate(results.get("results", [])):
                result_text = f"\n[Tavily Result {i+1}] Title: {result.get('title', 'Unknown')}\n"
                result_text += f"URL: {result.get('url', 'N/A')}\n"
                result_text += f"Content: {result.get('content', 'N/A')[:800]}...\n"
                all_results.append(result_text)
        
        state["research_context"].extend(all_results)
        print(f"   → ✓ Found {len(all_results)} results from Tavily")
        
    except Exception as e:
        print(f"   ⚠️ Tavily search error: {e}")
        state["research_context"].append(f"[Tavily Error]: {str(e)}")
    
    return state


def serper_search_node(state: AgentState) -> AgentState:
    """
    Serper (Google) node for broad web search.
    Best for forums, Reddit, obscure content, niche topics.
    """
    print(f"\n🔍 [Serper/Google] Searching broad web index...")
    
    try:
        import requests
        
        serper_api_key = os.getenv("SERPER_API_KEY")
        all_results = []
        
        for query in state["search_queries"]:
            print(f"   → Query: {query}")
            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": serper_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                for i, item in enumerate(data.get("organic", [])):
                    result_text = f"\n[Google Result {i+1}] Title: {item.get('title', 'Unknown')}\n"
                    result_text += f"URL: {item.get('link', 'N/A')}\n"
                    result_text += f"Snippet: {item.get('snippet', 'N/A')[:400]}...\n"
                    all_results.append(result_text)
            else:
                state["research_context"].append(f"[Serper Error]: API request failed with status {response.status_code}")
        
        state["research_context"].extend(all_results)
        print(f"   → ✓ Found {len(all_results)} results from Google/Serper")
        
    except Exception as e:
        print(f"   ⚠️ Serper search error: {e}")
        state["research_context"].append(f"[Serper Error]: {str(e)}")
    
    return state


# ============================================================================
# WRITER NODE - The Writer (GLM-4.7)
# ============================================================================

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research synthesizer and technical writer. Your task is to create a comprehensive, academic-grade report based on the gathered research.

Guidelines:
1. Synthesize information from ALL sources
2. Use proper citations: [Source X] format
3. Structure report with clear sections:
   - Executive Summary
   - Background/Context
   - Key Findings
   - Detailed Analysis
   - Conclusions & Recommendations
4. Be thorough but concise
5. Highlight important insights and patterns
6. Use professional, academic language
7. Include relevant quotes or statistics with citations
8. Identify any gaps or limitations in the research

Output format: Markdown with proper headings, lists, and formatting."""),
    ("human", """Original Query: {query}

Research Context:
{context}

Create a comprehensive report based on this research.""")
])


def summarize_chunk(chunk: str, index: int, total_chunks: int) -> str:
    """
    Summarize a chunk of research context.
    
    Args:
        chunk: Text chunk to summarize
        index: Chunk index
        total_chunks: Total number of chunks
    
    Returns:
        Summary of the chunk
    """
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research summarizer. Summarize the following research results concisely but comprehensively.

Keep:
- Key findings and important information
- Source names and titles
- Relevant statistics and data
- Important quotes or claims

Discard:
- Repetitive information
- Minor details
- Formatting artifacts

Output format: Concise paragraph with bullet points for key facts."""),
        ("human", f"Chunk {index}/{total_chunks}:\n\n{chunk}")
    ])
    
    try:
        response = deepseek_llm.invoke(summary_prompt.format_messages())
        return response.content.strip()
    except Exception as e:
        print(f"   ⚠️ Error summarizing chunk {index}: {e}")
        return f"[Summary Error for chunk {index}]"


def process_large_context(context_str: str, max_chunk_size: int = 80000) -> str:
    """
    Process large context by splitting into chunks and summarizing each.
    
    Args:
        context_str: Full research context
        max_chunk_size: Maximum size of each chunk in characters
    
    Returns:
        Combined summary of all chunks
    """
    print(f"\n🔄 [Context Processing] Large context detected ({len(context_str):,} chars)")
    print(f"   → Splitting into chunks and summarizing...")
    
    # Split into chunks
    chunks = []
    for i in range(0, len(context_str), max_chunk_size):
        chunk = context_str[i:i + max_chunk_size]
        chunks.append(chunk)
    
    print(f"   → Divided into {len(chunks)} chunks")
    
    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"   → Summarizing chunk {i}/{len(chunks)}...")
        summary = summarize_chunk(chunk, i, len(chunks))
        summaries.append(f"\n### Chunk {i} Summary:\n{summary}")
    
    # Combine summaries
    combined_summary = "\n".join(summaries)
    print(f"   → ✓ Summarized {len(context_str):,} chars → {len(combined_summary):,} chars")
    
    return combined_summary


def writer_node(state: AgentState) -> AgentState:
    """
    Writer node that synthesizes all research into a final report.
    Powered by GLM-4.7.
    """
    print(f"\n✍️  [GLM-4.7] Synthesizing research into final report...")
    
    # Combine all research context
    context_str = "\n".join(state["research_context"])
    print(f"   → Total context: {len(context_str):,} characters")
    
    # Process large context if needed
    if len(context_str) > 100000:
        context_str = process_large_context(context_str)
        print(f"   → Using summarized context for report generation")
    else:
        print(f"   → Context fits in model limits, using as-is")
    
    print(f"   → Final context size: {len(context_str):,} characters")
    
    # Generate final report
    response = writer_llm.invoke(
        writer_prompt.format_messages(
            query=state["query"],
            context=context_str
        )
    )
    
    # Add final report to messages
    state["messages"].append(AIMessage(content=response.content))
    
    print(f"   → ✓ Report generated successfully!")
    
    return state


# ============================================================================
# SAVE REPORT FUNCTION
# ============================================================================

def save_report(query: str, report_content: str, research_context: List[str]):
    """
    Save research report to a file.
    
    Args:
        query: The original research query
        report_content: The generated report content
        research_context: List of research context strings
    """
    # Create output directory if it doesn't exist
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename: query-timestamp.md
    # Sanitize query for filename
    safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{safe_query}-{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Save report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Research Report: {query}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Sources:** {len(research_context)} (if accessible)\n\n")
        f.write("---\n\n")
        f.write(report_content)
    
    print(f"\n💾 Report saved to: {filepath}")
    return filepath


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def route_to_tool(state: AgentState) -> str:
    """Conditional routing based on router decision."""
    selected_tool = state.get("selected_tool", "tavily")
    
    if selected_tool == "finish":
        return "writer"
    elif selected_tool == "exa":
        return "exa_search"
    elif selected_tool == "tavily":
        return "tavily_search"
    elif selected_tool == "serper":
        return "serper_search"
    else:
        return "tavily_search"  # Default


def should_continue_research(state: AgentState) -> str:
    """Decide whether to continue researching or write the report."""
    max_iter = state.get("max_iterations", 5)
    if state["step_count"] > max_iter:
        return "writer"

    # Check if router signaled finish
    if state.get("selected_tool") == "finish":
        return "writer"

    return "router"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_graph() -> StateGraph:
    """Build the LangGraph for the research agent."""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("exa_search", exa_search_node)
    workflow.add_node("tavily_search", tavily_search_node)
    workflow.add_node("serper_search", serper_search_node)
    workflow.add_node("writer", writer_node)
    
    # Define edges
    workflow.set_entry_point("router")
    
    # Router to search tools
    workflow.add_conditional_edges(
        "router",
        route_to_tool,
        {
            "exa_search": "exa_search",
            "tavily_search": "tavily_search",
            "serper_search": "serper_search",
            "writer": "writer"
        }
    )
    
    # Search nodes back to router
    workflow.add_edge("exa_search", "router")
    workflow.add_edge("tavily_search", "router")
    workflow.add_edge("serper_search", "router")
    
    # Writer to end
    workflow.add_edge("writer", END)
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_research_agent(query: str, forced_providers: List[str] = None, max_iterations: int = 5):
    """
    Run the multi-platform deep research agent.

    Args:
        query: The research question or topic
        forced_providers: List of providers to force (e.g., ["serper", "tavily"]).
                         When set, bypasses the DeepSeek router and cycles through
                         these providers round-robin.
        max_iterations: Maximum number of research iterations (default: 5)
    """
    print("=" * 80)
    print("🚀 MULTI-PLATFORM DEEP RESEARCH AGENT")
    print("=" * 80)
    print(f"\n📝 Query: {query}")
    if forced_providers:
        print(f"🔒 Forced Providers: {', '.join(p.upper() for p in forced_providers)}")
    else:
        print(f"🤖 Mode: Auto-routing (DeepSeek selects best provider)")
    print(f"🔄 Max Iterations: {max_iterations}\n")

    # Initialize state
    initial_state = {
        "query": query,
        "research_context": [],
        "selected_tool": None,
        "search_queries": None,
        "step_count": 1,
        "messages": [HumanMessage(content=query)],
        "forced_providers": forced_providers,
        "max_iterations": max_iterations,
    }

    # Create and run the graph
    graph = create_graph()

    try:
        print("\n" + "=" * 80)
        print("⚙️  STARTING RESEARCH WORKFLOW")
        print("=" * 80)

        final_state = graph.invoke(initial_state)

        print("\n" + "=" * 80)
        print("📊 FINAL REPORT")
        print("=" * 80 + "\n")

        # Print final report
        final_message = final_state["messages"][-1]
        print(final_message.content)

        # Save report to file
        save_report(query, final_message.content, final_state['research_context'])

        print("\n" + "=" * 80)
        print("✅ RESEARCH COMPLETE")
        print("=" * 80)
        print(f"📈 Total research iterations: {final_state['step_count']}")
        print(f"📚 Total sources gathered: {len(final_state['research_context'])}")

    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    VALID_PROVIDERS = ["exa", "tavily", "serper"]

    parser = argparse.ArgumentParser(
        description="Multi-Platform Deep Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-mode (DeepSeek router picks best provider):
  python main.py "quantum computing advances"

  # Force specific providers (bypass router):
  python main.py --providers serper,tavily "WorldView-3 satellite datasets"
  python main.py -p serper "Reddit AI projects"
  python main.py -p exa,serper,tavily "forest height estimation LiDAR"

  # Control iterations:
  python main.py --providers serper,tavily --iterations 8 "deep search query"

Provider descriptions:
  exa      Academic papers, technical docs, PDFs, whitepapers
  tavily   Breaking news, market data, current events, advanced search
  serper   Google web search, forums, Reddit, niche content
        """
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Research query (or use interactive mode if omitted)"
    )

    parser.add_argument(
        "-p", "--providers",
        type=str,
        default=None,
        help="Comma-separated list of search providers to force (e.g., serper,tavily,exa). "
             "Bypasses the DeepSeek auto-router and cycles through providers round-robin."
    )

    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=5,
        help="Maximum research iterations (default: 5)"
    )

    args = parser.parse_args()

    # Parse query
    query = " ".join(args.query) if args.query else input("🔍 Enter your research query: ")

    # Parse providers
    forced_providers = None
    if args.providers:
        forced_providers = [p.strip().lower() for p in args.providers.split(",")]
        invalid = [p for p in forced_providers if p not in VALID_PROVIDERS]
        if invalid:
            parser.error(f"Invalid provider(s): {', '.join(invalid)}. Valid: {', '.join(VALID_PROVIDERS)}")

    run_research_agent(
        query=query,
        forced_providers=forced_providers,
        max_iterations=args.iterations
    )