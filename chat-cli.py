#!/usr/bin/env python3
"""
CLI Chat Interface for Research Paper Analysis
Interactive chat interface that uses RAG and knowledge graph to answer questions about research papers.
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Dict, List, Optional
from pathlib import Path
import logging

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
import click

# Import our RAG system
from research_rag_system import ResearchPaperRAG, Paper

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with local LLM servers (Ollama/vLLM)."""
    
    def __init__(self, server_type: str = "ollama", base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.server_type = server_type.lower()
        self.base_url = base_url
        self.model = model
        self.console = Console()
        
        if self.server_type == "ollama":
            self.api_endpoint = f"{base_url}/api/generate"
        elif self.server_type == "vllm":
            self.api_endpoint = f"{base_url}/v1/completions"
        else:
            raise ValueError("Unsupported server type. Use 'ollama' or 'vllm'")
    
    def generate_response(self, prompt: str, context: List[str] = None, max_tokens: int = 500) -> str:
        """Generate response from local LLM."""
        try:
            # Construct full prompt with context
            full_prompt = self._build_prompt(prompt, context)
            
            if self.server_type == "ollama":
                return self._call_ollama(full_prompt, max_tokens)
            elif self.server_type == "vllm":
                return self._call_vllm(full_prompt, max_tokens)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response. Please check if the {self.server_type} server is running."
    
    def _build_prompt(self, question: str, context: List[str] = None) -> str:
        """Build the complete prompt with context."""
        prompt_parts = [
            "You are a helpful research assistant analyzing scientific papers.",
            "Use the provided context to answer the user's question accurately and comprehensively.",
            "If the context doesn't contain enough information, say so clearly.",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "CONTEXT:",
                "=" * 50,
            ])
            
            for i, ctx in enumerate(context, 1):
                prompt_parts.extend([
                    f"Document {i}:",
                    ctx,
                    "-" * 30,
                ])
            
            prompt_parts.extend([
                "",
                "QUESTION:",
                question,
                "",
                "ANSWER:"
            ])
        else:
            prompt_parts.extend([
                "QUESTION:",
                question,
                "",
                "ANSWER:"
            ])
        
        return "\n".join(prompt_parts)
    
    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        """Call Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    def _call_vllm(self, prompt: str, max_tokens: int) -> str:
        """Call vLLM API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("text", "No response generated").strip()
            return "No response generated"
        else:
            raise Exception(f"vLLM API error: {response.status_code} - {response.text}")
    
    def test_connection(self) -> bool:
        """Test connection to LLM server."""
        try:
            if self.server_type == "ollama":
                # Test with a simple prompt
                response = requests.post(
                    self.api_endpoint,
                    json={"model": self.model, "prompt": "Hello", "stream": False},
                    timeout=10
                )
                return response.status_code == 200
            elif self.server_type == "vllm":
                # Test with vLLM health endpoint or simple completion
                response = requests.post(
                    self.api_endpoint,
                    json={"model": self.model, "prompt": "Hello", "max_tokens": 1},
                    timeout=10
                )
                return response.status_code == 200
        except:
            return False

class ChatInterface:
    """Interactive chat interface for research paper analysis."""
    
    def __init__(self, rag_system: ResearchPaperRAG, llm_client: LLMClient):
        self.rag_system = rag_system
        self.llm_client = llm_client
        self.console = Console()
        self.chat_history = []
        
    def start_chat(self, paper_path: str = None):
        """Start interactive chat session."""
        self.console.print(Panel.fit(
            "[bold cyan]Research Paper Analysis Chat Interface[/bold cyan]\n"
            "Ask questions about the research paper and its related work.\n"
            "Type 'help' for commands, 'quit' to exit.",
            title="Welcome"
        ))
        
        # Test LLM connection
        if not self.llm_client.test_connection():
            self.console.print(
                "[bold red]Warning: Cannot connect to LLM server. "
                f"Please ensure {self.llm_client.server_type} is running at {self.llm_client.base_url}[/bold red]"
            )
        
        if paper_path:
            self.console.print(f"[dim]Loaded paper: {paper_path}[/dim]")
        
        self.console.print("\n[bold green]Ready! Ask me anything about the research paper.[/bold green]\n")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[dim]Goodbye![/dim]")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.chat_history.clear()
                    self.console.clear()
                    self.console.print("[dim]Chat history cleared.[/dim]\n")
                    continue
                
                # Process question
                self._process_question(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[dim]Goodbye![/dim]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def _process_question(self, question: str):
        """Process user question and generate response."""
        # Add to chat history
        self.chat_history.append({"role": "user", "content": question})
        
        with self.console.status("[bold green]Searching knowledge base...[/bold green]"):
            # Query RAG system
            rag_results = self.rag_system.query(question, top_k=5)
        
        # Extract context from retrieved documents
        context_texts = []
        sources = []
        
        for doc in rag_results['retrieved_documents']:
            context_texts.append(doc['text'])
            sources.append({
                'title': doc['metadata'].get('title', 'Unknown'),
                'score': doc['relevance_score'],
                'chunk_id': doc['metadata'].get('chunk_id', 0)
            })
        
        # Show retrieved sources
        if sources:
            self._display_sources(sources)
        
        with self.console.status("[bold green]Generating response...[/bold green]"):
            # Generate response using LLM
            response = self.llm_client.generate_response(
                question, 
                context_texts, 
                max_tokens=800
            )
        
        # Display response
        self._display_response(response)
        
        # Add to chat history
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Show related papers if available
        if rag_results['related_papers']:
            self._display_related_papers(rag_results['related_papers'])
    
    def _display_sources(self, sources: List[Dict]):
        """Display retrieved sources."""
        table = Table(title="📚 Retrieved Sources", show_header=True, header_style="bold magenta")
        table.add_column("Paper", style="cyan", no_wrap=True)
        table.add_column("Relevance", style="green", justify="center")
        table.add_column("Chunk", style="dim", justify="center")
        
        for source in sources[:3]:  # Show top 3 sources
            title = source['title'][:40] + "..." if len(source['title']) > 40 else source['title']
            relevance = f"{source['score']:.2f}"
            chunk = str(source['chunk_id'])
            
            table.add_row(title, relevance, chunk)
        
        self.console.print(table)
        self.console.print()
    
    def _display_response(self, response: str):
        """Display AI response."""
        self.console.print(Panel(
            Markdown(response),
            title="🤖 Assistant",
            title_align="left",
            border_style="green"
        ))
        self.console.print()
    
    def _display_related_papers(self, related_papers: List[str]):
        """Display related papers."""
        if not related_papers:
            return
            
        self.console.print(Panel(
            f"[dim]Related papers in knowledge graph: {', '.join(related_papers[:3])}[/dim]",
            title="🔗 Related Work",
            border_style="dim"
        ))
        self.console.print()
    
    def _show_help(self):
        """Show help information."""
        help_text = """
[bold]Available Commands:[/bold]
• [cyan]help[/cyan] - Show this help message
• [cyan]stats[/cyan] - Show knowledge base statistics  
• [cyan]history[/cyan] - Show chat history
• [cyan]clear[/cyan] - Clear chat history
• [cyan]quit/exit/q[/cyan] - Exit the chat

[bold]Tips:[/bold]
• Ask specific questions about the paper's methodology, results, or conclusions
• Request explanations of technical concepts or terms
• Ask for comparisons with related work
• Inquire about the paper's contributions or limitations
"""
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
        self.console.print()
    
    def _show_stats(self):
        """Show knowledge base statistics."""
        try:
            num_papers = len(self.rag_system.all_papers)
            num_embeddings = len(self.rag_system.rag_system.embeddings) if self.rag_system.rag_system.embeddings is not None else 0
            num_graph_nodes = len(self.rag_system.knowledge_graph.graph.nodes)
            num_graph_edges = len(self.rag_system.knowledge_graph.graph.edges)
            
            stats_text = f"""
[bold]Knowledge Base Statistics:[/bold]

📄 Papers: {num_papers}
🧩 Text chunks: {num_embeddings}
🔗 Graph nodes: {num_graph_nodes}
🔗 Graph edges: {num_graph_edges}
💬 Chat messages: {len(self.chat_history)}

[dim]LLM Server: {self.llm_client.server_type} ({self.llm_client.model})[/dim]
"""
            
            self.console.print(Panel(stats_text, title="Statistics", border_style="yellow"))
            
        except Exception as e:
            self.console.print(f"[red]Error getting statistics: {e}[/red]")
        
        self.console.print()
    
    def _show_history(self):
        """Show chat history."""
        if not self.chat_history:
            self.console.print("[dim]No chat history yet.[/dim]\n")
            return
        
        self.console.print(Panel.fit("[bold]Chat History[/bold]", border_style="blue"))
        
        for i, message in enumerate(self.chat_history[-10:], 1):  # Show last 10 messages
            role = "You" if message["role"] == "user" else "Assistant"
            color = "blue" if message["role"] == "user" else "green"
            
            content = message["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            
            self.console.print(f"[bold {color}]{role}:[/bold {color}] {content}")
        
        if len(self.chat_history) > 10:
            self.console.print(f"[dim]... and {len(self.chat_history) - 10} more messages[/dim]")
        
        self.console.print()

async def main():
    """Main entry point for CLI chat interface."""
    parser = argparse.ArgumentParser(description="Research Paper Chat Interface")
    parser.add_argument("--paper", required=True, help="Path to research paper PDF")
    parser.add_argument("--depth", type=int, default=2, help="Reference analysis depth (default: 2)")
    parser.add_argument("--server-type", choices=["ollama", "vllm"], default="ollama", 
                       help="LLM server type (default: ollama)")
    parser.add_argument("--server-url", default="http://localhost:11434", 
                       help="LLM server URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="llama2", 
                       help="LLM model name (default: llama2)")
    parser.add_argument("--download-dir", default="./papers", 
                       help="Directory for downloaded papers (default: ./papers)")
    parser.add_argument("--rebuild", action="store_true", 
                       help="Rebuild RAG system from scratch")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Verify paper exists
    if not os.path.exists(args.paper):
        console.print(f"[red]Error: Paper file '{args.paper}' not found![/red]")
        sys.exit(1)
    
    # Initialize LLM client
    console.print(f"[dim]Connecting to {args.server_type} server at {args.server_url}...[/dim]")
    llm_client = LLMClient(
        server_type=args.server_type,
        base_url=args.server_url,
        model=args.model
    )
    
    # Initialize RAG system
    console.print("[dim]Initializing RAG system...[/dim]")
    rag_system = ResearchPaperRAG(download_dir=args.download_dir)
    
    # Check if RAG system already exists for this paper
    rag_db_path = f"rag_{Path(args.paper).stem}.db"
    needs_build = args.rebuild or not os.path.exists(rag_db_path)
    
    if needs_build:
        console.print(f"[bold yellow]Building RAG system for '{args.paper}' (depth: {args.depth})...[/bold yellow]")
        console.print("[dim]This may take several minutes depending on the number of references...[/dim]")
        
        try:
            # Build RAG system
            await rag_system.build_rag_from_paper(args.paper, max_depth=args.depth)
            console.print("[bold green]RAG system built successfully![/bold green]\n")
        except Exception as e:
            console.print(f"[red]Error building RAG system: {e}[/red]")
            console.print("[yellow]Continuing with basic functionality...[/yellow]\n")
            
            # Load just the main paper if RAG build fails
            try:
                paper = rag_system._load_paper_from_pdf(args.paper)
                if paper:
                    processed_paper = await rag_system._process_paper(paper)
                    if processed_paper:
                        rag_system.all_papers.append(processed_paper)
                        rag_system.rag_system.add_documents([processed_paper])
            except Exception as e2:
                console.print(f"[red]Failed to load even the main paper: {e2}[/red]")
                sys.exit(1)
    else:
        console.print("[dim]Using existing RAG system...[/dim]")
        # TODO: Load existing RAG system from database
    
    # Start chat interface
    chat_interface = ChatInterface(rag_system, llm_client)
    chat_interface.start_chat(args.paper)

if __name__ == "__main__":
    asyncio.run(main())