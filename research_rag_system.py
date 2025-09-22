#!/usr/bin/env python3
"""
Research Paper RAG System
A comprehensive tool for downloading, analyzing, and building knowledge graphs from research papers.
"""

import os
import re
import json
import requests
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import hashlib
from datetime import datetime

import PyPDF2
import fitz  # PyMuPDF
import networkx as nx
from scholarly import scholarly
import arxiv
from crossref.restful import Works
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Represents a research paper with metadata and content."""
    title: str
    authors: List[str] = None
    abstract: str = ""
    content: str = ""
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    venue: str = ""
    year: int = 0
    references: List[str] = None
    pdf_path: str = ""
    embeddings: np.ndarray = None
    concepts: List[str] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.references is None:
            self.references = []
        if self.concepts is None:
            self.concepts = []

class PaperDownloader:
    """Handles downloading papers from various sources."""
    
    def __init__(self, download_dir: str = "./papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.works = Works()
        
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search and download papers from arXiv."""
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[str(author) for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    arxiv_id=result.entry_id.split('/')[-1],
                    year=result.published.year if result.published else 0
                )
                
                # Download PDF
                pdf_path = self._download_arxiv_pdf(result)
                if pdf_path:
                    paper.pdf_path = str(pdf_path)
                    paper.content = self._extract_text_from_pdf(pdf_path)
                
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            
        return papers
    
    def _download_arxiv_pdf(self, result) -> Optional[Path]:
        """Download PDF from arXiv."""
        try:
            filename = f"{result.entry_id.split('/')[-1]}.pdf"
            pdf_path = self.download_dir / filename
            
            if pdf_path.exists():
                return pdf_path
                
            result.download_pdf(dirpath=str(self.download_dir), filename=filename)
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None
    
    def search_crossref(self, title: str) -> Optional[Paper]:
        """Search for paper metadata using Crossref."""
        try:
            results = self.works.query(title).select("title", "author", "abstract", "DOI", "URL").sample(1)
            
            for item in results:
                if 'title' in item and item['title']:
                    paper = Paper(
                        title=item['title'][0] if isinstance(item['title'], list) else item['title'],
                        authors=[f"{author.get('given', '')} {author.get('family', '')}" 
                                for author in item.get('author', [])],
                        abstract=item.get('abstract', ''),
                        doi=item.get('DOI', ''),
                        url=item.get('URL', ''),
                        year=item.get('published-print', {}).get('date-parts', [[0]])[0][0] or
                             item.get('published-online', {}).get('date-parts', [[0]])[0][0] or 0
                    )
                    return paper
                    
        except Exception as e:
            logger.error(f"Error searching Crossref: {e}")
            
        return None
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            
            for page in doc:
                text += page.get_text()
                
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

class ReferenceExtractor:
    """Extracts references from research papers."""
    
    def __init__(self):
        self.reference_patterns = [
            r'\[(\d+)\]\s*([^\[\]]+)',  # [1] Author et al.
            r'\(([^)]+\d{4}[^)]*)\)',   # (Author, 2023)
            r'(?:(?:doi|DOI)[:=]\s*)([^\s,]+)',  # DOI references
            r'arXiv:(\d{4}\.\d{4,5})',  # arXiv IDs
        ]
    
    def extract_references(self, text: str) -> List[str]:
        """Extract reference strings from paper text."""
        references = set()
        
        # Look for references section
        ref_section_patterns = [
            r'(?i)references\s*\n(.*?)(?:\n\s*(?:appendix|acknowledgment)|\Z)',
            r'(?i)bibliography\s*\n(.*?)(?:\n\s*(?:appendix|acknowledgment)|\Z)',
        ]
        
        ref_text = ""
        for pattern in ref_section_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                ref_text = match.group(1)
                break
        
        # If no references section found, search entire text
        if not ref_text:
            ref_text = text
        
        # Extract references using patterns
        for pattern in self.reference_patterns:
            matches = re.findall(pattern, ref_text)
            for match in matches:
                if isinstance(match, tuple):
                    ref_string = ' '.join(match).strip()
                else:
                    ref_string = match.strip()
                
                if len(ref_string) > 10:  # Filter out short matches
                    references.add(ref_string)
        
        return list(references)
    
    def extract_title_from_reference(self, ref_string: str) -> Optional[str]:
        """Extract paper title from reference string."""
        # Simple heuristic: look for quoted titles or capitalize words
        title_patterns = [
            r'"([^"]+)"',  # Quoted titles
            r'[A-Z][^.!?]*[.!?]',  # Sentences starting with capital
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, ref_string)
            if match:
                title = match.group(1) if pattern.startswith('"') else match.group(0)
                if len(title.split()) > 3:  # Must have at least 3 words
                    return title.strip()
        
        return None

class ConceptExtractor:
    """Extracts key concepts and entities from research papers."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.stop_words = set(stopwords.words('english'))
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        concepts = set()
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                    concepts.add(ent.text.lower())
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit phrase length
                    clean_chunk = ' '.join([token.lemma_ for token in chunk 
                                          if not token.is_stop and token.is_alpha])
                    if clean_chunk:
                        concepts.add(clean_chunk.lower())
        
        # Extract technical terms (words with certain patterns)
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:[-_]\w+)+\b',  # Hyphenated/underscored terms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            concepts.update([match.lower() for match in matches])
        
        # Filter out common words and short terms
        filtered_concepts = [concept for concept in concepts 
                           if concept not in self.stop_words and len(concept) > 2]
        
        return filtered_concepts[:50]  # Limit to top 50 concepts

class KnowledgeGraph:
    """Builds and manages knowledge graph of papers and concepts."""
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.graph = nx.DiGraph()
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                content TEXT,
                url TEXT,
                doi TEXT,
                arxiv_id TEXT,
                venue TEXT,
                year INTEGER,
                pdf_path TEXT,
                concepts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS references (
                paper_id TEXT,
                reference_id TEXT,
                reference_text TEXT,
                FOREIGN KEY (paper_id) REFERENCES papers (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                paper_id TEXT,
                concept TEXT,
                FOREIGN KEY (paper_id) REFERENCES papers (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_paper(self, paper: Paper) -> str:
        """Add paper to knowledge graph and database."""
        paper_id = hashlib.md5(paper.title.encode()).hexdigest()
        
        # Add to graph
        self.graph.add_node(paper_id, type='paper', **asdict(paper))
        
        # Add concepts as nodes and connect to paper
        for concept in paper.concepts:
            concept_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()}"
            self.graph.add_node(concept_id, type='concept', name=concept)
            self.graph.add_edge(paper_id, concept_id, relation='contains_concept')
        
        # Add to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO papers 
            (id, title, authors, abstract, content, url, doi, arxiv_id, venue, year, pdf_path, concepts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper_id, paper.title, json.dumps(paper.authors), paper.abstract,
            paper.content, paper.url, paper.doi, paper.arxiv_id, paper.venue,
            paper.year, paper.pdf_path, json.dumps(paper.concepts)
        ))
        
        for ref in paper.references:
            cursor.execute('''
                INSERT INTO references (paper_id, reference_text) VALUES (?, ?)
            ''', (paper_id, ref))
        
        for concept in paper.concepts:
            cursor.execute('''
                INSERT INTO concepts (paper_id, concept) VALUES (?, ?)
            ''', (paper_id, concept))
        
        conn.commit()
        conn.close()
        
        return paper_id
    
    def add_reference_link(self, source_paper_id: str, target_paper_id: str):
        """Add reference relationship between papers."""
        self.graph.add_edge(source_paper_id, target_paper_id, relation='references')
    
    def get_related_papers(self, paper_id: str, max_depth: int = 2) -> List[str]:
        """Get papers related to given paper within specified depth."""
        try:
            if paper_id not in self.graph:
                return []
            
            related = set()
            current_level = {paper_id}
            
            for depth in range(max_depth):
                next_level = set()
                for node in current_level:
                    neighbors = list(self.graph.neighbors(node))
                    neighbors.extend(list(self.graph.predecessors(node)))
                    
                    for neighbor in neighbors:
                        if self.graph.nodes[neighbor].get('type') == 'paper':
                            related.add(neighbor)
                            next_level.add(neighbor)
                
                current_level = next_level
            
            return list(related)
            
        except Exception as e:
            logger.error(f"Error getting related papers: {e}")
            return []
    
    def save_graph(self, filepath: str = "knowledge_graph.json"):
        """Save knowledge graph to JSON file."""
        data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

class RAGSystem:
    """Retrieval-Augmented Generation system for research papers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.index = None
        self.dimension = None
    
    def add_documents(self, papers: List[Paper]):
        """Add papers to RAG system."""
        for paper in tqdm(papers, desc="Adding documents to RAG"):
            # Split content into chunks
            chunks = self._chunk_text(paper.content, chunk_size=512)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Skip very short chunks
                    embedding = self.encoder.encode([chunk])[0]
                    
                    self.embeddings.append(embedding)
                    self.texts.append(chunk)
                    self.metadata.append({
                        'paper_id': hashlib.md5(paper.title.encode()).hexdigest(),
                        'title': paper.title,
                        'chunk_id': i,
                        'authors': paper.authors,
                        'year': paper.year
                    })
        
        # Build FAISS index
        if self.embeddings:
            self.embeddings = np.array(self.embeddings).astype('float32')
            self.dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(word_tokenize(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Retrieve relevant documents for query."""
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], self.metadata[idx], float(score)))
        
        return results

class ResearchPaperRAG:
    """Main class orchestrating the entire RAG system."""
    
    def __init__(self, download_dir: str = "./papers", db_path: str = "research_rag.db"):
        self.downloader = PaperDownloader(download_dir)
        self.ref_extractor = ReferenceExtractor()
        self.concept_extractor = ConceptExtractor()
        self.knowledge_graph = KnowledgeGraph(db_path)
        self.rag_system = RAGSystem()
        
        self.processed_papers = set()
        self.all_papers = []
    
    async def build_rag_from_paper(self, paper_path: str, max_depth: int = 2) -> None:
        """Build RAG system starting from a single paper."""
        logger.info(f"Building RAG from paper: {paper_path}")
        
        # Extract initial paper
        initial_paper = self._load_paper_from_pdf(paper_path)
        if not initial_paper:
            logger.error("Failed to load initial paper")
            return
        
        # Process paper iteratively
        papers_to_process = [initial_paper]
        current_depth = 0
        
        while papers_to_process and current_depth < max_depth:
            logger.info(f"Processing depth {current_depth + 1}")
            next_batch = []
            
            for paper in tqdm(papers_to_process, desc=f"Depth {current_depth + 1}"):
                if paper.title in self.processed_papers:
                    continue
                
                # Process current paper
                processed_paper = await self._process_paper(paper)
                if processed_paper:
                    self.all_papers.append(processed_paper)
                    paper_id = self.knowledge_graph.add_paper(processed_paper)
                    self.processed_papers.add(paper.title)
                    
                    # Find referenced papers
                    referenced_papers = await self._find_referenced_papers(processed_paper)
                    next_batch.extend(referenced_papers)
                    
                    # Add reference links
                    for ref_paper in referenced_papers:
                        if ref_paper.title in self.processed_papers:
                            ref_id = hashlib.md5(ref_paper.title.encode()).hexdigest()
                            self.knowledge_graph.add_reference_link(paper_id, ref_id)
            
            papers_to_process = next_batch
            current_depth += 1
        
        # Build RAG system
        logger.info("Building RAG index...")
        self.rag_system.add_documents(self.all_papers)
        
        # Save knowledge graph
        self.knowledge_graph.save_graph()
        
        logger.info(f"RAG system built with {len(self.all_papers)} papers")
    
    def _load_paper_from_pdf(self, pdf_path: str) -> Optional[Paper]:
        """Load paper from PDF file."""
        try:
            content = self.downloader._extract_text_from_pdf(Path(pdf_path))
            if not content:
                return None
            
            # Extract title (heuristic: first line or largest font)
            lines = content.split('\n')
            title = lines[0].strip() if lines else "Unknown Title"
            
            # Create paper object
            paper = Paper(
                title=title,
                content=content,
                pdf_path=pdf_path
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error loading paper from {pdf_path}: {e}")
            return None
    
    async def _process_paper(self, paper: Paper) -> Optional[Paper]:
        """Process a single paper to extract references and concepts."""
        try:
            # Extract references
            paper.references = self.ref_extractor.extract_references(paper.content)
            
            # Extract concepts
            paper.concepts = self.concept_extractor.extract_concepts(paper.content)
            
            # Try to get additional metadata if we have title
            if not paper.doi and not paper.arxiv_id:
                metadata = self.downloader.search_crossref(paper.title)
                if metadata:
                    paper.authors = metadata.authors or paper.authors
                    paper.doi = metadata.doi or paper.doi
                    paper.year = metadata.year or paper.year
                    paper.abstract = metadata.abstract or paper.abstract
            
            return paper
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.title}: {e}")
            return None
    
    async def _find_referenced_papers(self, paper: Paper) -> List[Paper]:
        """Find and download papers referenced by the given paper."""
        referenced_papers = []
        
        for ref in paper.references[:10]:  # Limit to first 10 references
            try:
                # Extract potential title
                title = self.ref_extractor.extract_title_from_reference(ref)
                if not title:
                    continue
                
                # Search for paper
                found_papers = []
                
                # Try arXiv search
                arxiv_papers = self.downloader.search_arxiv(title, max_results=1)
                if arxiv_papers:
                    found_papers.extend(arxiv_papers)
                
                # Try Crossref search if no arXiv results
                if not found_papers:
                    crossref_paper = self.downloader.search_crossref(title)
                    if crossref_paper:
                        found_papers.append(crossref_paper)
                
                referenced_papers.extend(found_papers)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error finding referenced paper: {e}")
                continue
        
        return referenced_papers
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the RAG system."""
        # Retrieve relevant documents
        results = self.rag_system.retrieve(question, top_k)
        
        # Format response
        response = {
            'query': question,
            'retrieved_documents': [],
            'related_papers': []
        }
        
        paper_ids = set()
        for text, metadata, score in results:
            response['retrieved_documents'].append({
                'text': text,
                'metadata': metadata,
                'relevance_score': score
            })
            paper_ids.add(metadata['paper_id'])
        
        # Get related papers from knowledge graph
        for paper_id in paper_ids:
            related = self.knowledge_graph.get_related_papers(paper_id, max_depth=1)
            response['related_papers'].extend(related[:3])  # Limit results
        
        return response

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Paper RAG System")
    parser.add_argument("--paper", required=True, help="Path to initial PDF paper")
    parser.add_argument("--depth", type=int, default=2, help="Maximum reference depth")
    parser.add_argument("--download-dir", default="./papers", help="Directory to store downloaded papers")
    
    args = parser.parse_args()
    
    # Create and run RAG system
    rag_system = ResearchPaperRAG(download_dir=args.download_dir)
    
    # Run the async build process
    asyncio.run(rag_system.build_rag_from_paper(args.paper, max_depth=args.depth))