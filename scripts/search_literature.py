#!/usr/bin/env python3
"""
Literature Search Module for Healthcare AI Equity Textbook

This module searches PubMed, arXiv, and monitors top journals/conferences
for relevant papers to update textbook chapters.

Author: Sanjay Basu, MD PhD
License: MIT
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import quote

import requests
import feedparser
import yaml
from Bio import Entrez

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a research paper with metadata."""
    title: str
    authors: List[str]
    abstract: str
    source: str  # 'pubmed', 'arxiv', 'journal'
    source_id: str
    url: str
    publication_date: str
    journal: Optional[str] = None
    doi: Optional[str] = None
    arxiv_category: Optional[str] = None
    pmid: Optional[str] = None
    citations: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class LiteratureSearcher:
    """Main class for searching literature across multiple sources."""
    
    def __init__(
        self,
        config_dir: str = "config",
        email: str = "sanjay.basu@waymarkcare.com",
        api_key: Optional[str] = None
    ):
        """
        Initialize the literature searcher.
        
        Args:
            config_dir: Directory containing configuration files
            email: Email for PubMed API (required)
            api_key: PubMed API key (optional but recommended)
        """
        self.config_dir = Path(config_dir)
        self.email = email
        self.api_key = api_key or os.getenv('PUBMED_API_KEY')
        
        # Configure Entrez
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
        
        # Load configurations
        self.journals = self._load_config('journals.yml')
        self.conferences = self._load_config('conferences.yml')
        self.arxiv_categories = self._load_config('arxiv_categories.yml')
        
        # Track seen papers to avoid duplicates
        self.seen_papers: Set[str] = set()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.34 if self.api_key else 3.0  # seconds
        
    def _load_config(self, filename: str) -> Dict:
        """Load YAML configuration file."""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _is_duplicate(self, paper_id: str) -> bool:
        """Check if paper has already been processed."""
        if paper_id in self.seen_papers:
            return True
        self.seen_papers.add(paper_id)
        return False
    
    def search_pubmed(
        self,
        query: str,
        date_range: int = 7,
        max_results: int = 100
    ) -> List[Paper]:
        """
        Search PubMed for papers.
        
        Args:
            query: Search query string
            date_range: Number of days to look back
            max_results: Maximum number of results to return
            
        Returns:
            List of Paper objects
        """
        logger.info(f"Searching PubMed: {query}")
        papers = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_range)
            
            date_filter = f"({start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[pdat])"
            full_query = f"{query} AND {date_filter}"
            
            # Search
            self._rate_limit()
            search_handle = Entrez.esearch(
                db="pubmed",
                term=full_query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results["IdList"]
            logger.info(f"Found {len(id_list)} PubMed results")
            
            if not id_list:
                return papers
            
            # Fetch details in batches
            batch_size = 20
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i+batch_size]
                self._rate_limit()
                
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_ids,
                    rettype="medline",
                    retmode="xml"
                )
                records = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                for record in records['PubmedArticle']:
                    try:
                        paper = self._parse_pubmed_record(record)
                        if paper and not self._is_duplicate(paper.source_id):
                            papers.append(paper)
                    except Exception as e:
                        logger.warning(f"Error parsing PubMed record: {e}")
                        continue
                
                # Progress logging
                logger.info(f"Processed {min(i+batch_size, len(id_list))}/{len(id_list)} PubMed records")
        
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
        
        return papers
    
    def _parse_pubmed_record(self, record: Dict) -> Optional[Paper]:
        """Parse a PubMed record into a Paper object."""
        try:
            article = record['MedlineCitation']['Article']
            
            # Extract title
            title = article.get('ArticleTitle', '')
            if isinstance(title, list):
                title = ' '.join(str(t) for t in title)
            
            # Extract authors
            authors = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    if 'LastName' in author:
                        name = f"{author.get('LastName', '')} {author.get('Initials', '')}".strip()
                        authors.append(name)
            
            # Extract abstract
            abstract = ""
            if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                abstract_parts = article['Abstract']['AbstractText']
                if isinstance(abstract_parts, list):
                    abstract = ' '.join(str(part) for part in abstract_parts)
                else:
                    abstract = str(abstract_parts)
            
            # Extract journal
            journal = article.get('Journal', {}).get('Title', '')
            
            # Extract publication date
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '01')
            day = pub_date.get('Day', '01')
            publication_date = f"{year}-{month:0>2}-{day:0>2}"
            
            # Extract PMID
            pmid = str(record['MedlineCitation']['PMID'])
            
            # Extract DOI
            doi = None
            if 'ArticleIdList' in record['PubmedData']:
                for article_id in record['PubmedData']['ArticleIdList']:
                    if hasattr(article_id, 'attributes') and article_id.attributes.get('IdType') == 'doi':
                        doi = str(article_id)
            
            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                source='pubmed',
                source_id=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                publication_date=publication_date,
                journal=journal,
                doi=doi,
                pmid=pmid
            )
            
        except Exception as e:
            logger.warning(f"Error parsing PubMed record: {e}")
            return None
    
    def search_arxiv(
        self,
        categories: List[str],
        keywords: List[str],
        date_range: int = 7,
        max_results: int = 100
    ) -> List[Paper]:
        """
        Search arXiv for papers.
        
        Args:
            categories: List of arXiv categories (e.g., ['cs.LG', 'cs.AI'])
            keywords: List of keywords to search for
            date_range: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of Paper objects
        """
        logger.info(f"Searching arXiv: categories={categories}, keywords={keywords}")
        papers = []
        
        try:
            # Build query
            keyword_query = " OR ".join(f'all:"{kw}"' for kw in keywords)
            category_query = " OR ".join(f"cat:{cat}" for cat in categories)
            query = f"({keyword_query}) AND ({category_query})"
            
            # Construct URL
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{base_url}?{'&'.join(f'{k}={quote(str(v))}' for k, v in params.items())}"
            
            # Fetch results
            self._rate_limit()
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.content)
            
            logger.info(f"Found {len(feed.entries)} arXiv results")
            
            # Filter by date and parse
            cutoff_date = datetime.now() - timedelta(days=date_range)
            
            for entry in feed.entries:
                try:
                    # Check publication date
                    pub_date = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
                    if pub_date < cutoff_date:
                        continue
                    
                    paper = self._parse_arxiv_entry(entry)
                    if paper and not self._is_duplicate(paper.source_id):
                        papers.append(paper)
                        
                except Exception as e:
                    logger.warning(f"Error parsing arXiv entry: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
        
        return papers
    
    def _parse_arxiv_entry(self, entry: Dict) -> Optional[Paper]:
        """Parse an arXiv entry into a Paper object."""
        try:
            # Extract arXiv ID
            arxiv_id = entry.id.split('/abs/')[-1]
            
            # Extract authors
            authors = [author.name for author in entry.authors]
            
            # Extract category
            categories = [tag.term for tag in entry.tags]
            primary_category = categories[0] if categories else None
            
            return Paper(
                title=entry.title,
                authors=authors,
                abstract=entry.summary,
                source='arxiv',
                source_id=arxiv_id,
                url=entry.link,
                publication_date=entry.published[:10],  # YYYY-MM-DD
                arxiv_category=primary_category
            )
            
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None
    
    def search_journals(
        self,
        date_range: int = 7
    ) -> List[Paper]:
        """
        Search specific journals for recent papers.
        
        Args:
            date_range: Number of days to look back
            
        Returns:
            List of Paper objects
        """
        logger.info("Searching monitored journals")
        all_papers = []
        
        for journal_name, journal_info in self.journals.get('journals', {}).items():
            try:
                logger.info(f"Searching {journal_name}")
                
                # Build PubMed query for this journal
                query = f'"{journal_info["pubmed_name"]}"[jour]'
                
                papers = self.search_pubmed(
                    query=query,
                    date_range=date_range,
                    max_results=20
                )
                
                all_papers.extend(papers)
                
            except Exception as e:
                logger.error(f"Error searching {journal_name}: {e}")
                continue
        
        logger.info(f"Found {len(all_papers)} papers from monitored journals")
        return all_papers
    
    def search_all(
        self,
        date_range: int = 7,
        include_pubmed: bool = True,
        include_arxiv: bool = True,
        include_journals: bool = True
    ) -> List[Paper]:
        """
        Search all sources for papers.
        
        Args:
            date_range: Number of days to look back
            include_pubmed: Whether to search PubMed
            include_arxiv: Whether to search arXiv
            include_journals: Whether to search specific journals
            
        Returns:
            Combined list of papers from all sources
        """
        all_papers = []
        
        # Search PubMed with general healthcare AI query
        if include_pubmed:
            pubmed_queries = [
                '("machine learning"[MeSH Terms] OR "deep learning"[Title/Abstract]) AND ("health equity"[Title/Abstract] OR "health disparities"[MeSH Terms])',
                '"artificial intelligence"[MeSH Terms] AND ("healthcare"[Title/Abstract] OR "clinical"[Title/Abstract]) AND ("bias"[Title/Abstract] OR "fairness"[Title/Abstract])',
                '("predictive modeling"[Title/Abstract] OR "risk prediction"[Title/Abstract]) AND ("underserved"[Title/Abstract] OR "vulnerable populations"[MeSH Terms])'
            ]
            
            for query in pubmed_queries:
                papers = self.search_pubmed(query, date_range, max_results=50)
                all_papers.extend(papers)
        
        # Search arXiv
        if include_arxiv:
            arxiv_cats = self.arxiv_categories.get('categories', [])
            keywords = [
                'healthcare artificial intelligence',
                'medical machine learning',
                'clinical decision support',
                'health equity AI',
                'algorithmic bias healthcare',
                'fair machine learning medicine'
            ]
            
            papers = self.search_arxiv(
                categories=arxiv_cats,
                keywords=keywords,
                date_range=date_range,
                max_results=50
            )
            all_papers.extend(papers)
        
        # Search monitored journals
        if include_journals:
            papers = self.search_journals(date_range)
            all_papers.extend(papers)
        
        # Remove duplicates
        unique_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            # Use title as duplicate key (normalized)
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        logger.info(f"Total unique papers found: {len(unique_papers)}")
        return unique_papers
    
    def save_results(
        self,
        papers: List[Paper],
        output_dir: str,
        filename: str = "papers.json"
    ):
        """
        Save search results to JSON file.
        
        Args:
            papers: List of Paper objects
            output_dir: Directory to save results
            filename: Name of output file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / filename
        
        # Convert to dictionaries
        papers_data = [paper.to_dict() for paper in papers]
        
        # Add metadata
        output_data = {
            'search_date': datetime.now().isoformat(),
            'total_papers': len(papers),
            'sources': {
                'pubmed': len([p for p in papers if p.source == 'pubmed']),
                'arxiv': len([p for p in papers if p.source == 'arxiv']),
                'journal': len([p for p in papers if p.source == 'journal'])
            },
            'papers': papers_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {len(papers)} papers to {output_file}")


def main():
    """Main entry point for literature search."""
    parser = argparse.ArgumentParser(
        description="Search literature for healthcare AI equity textbook updates"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/literature',
        help='Output directory for search results'
    )
    parser.add_argument(
        '--date-range',
        type=int,
        default=7,
        help='Number of days to look back'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Directory containing configuration files'
    )
    parser.add_argument(
        '--email',
        type=str,
        default='sanjay.basu@waymarkcare.com',
        help='Email for PubMed API'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without saving results'
    )
    parser.add_argument(
        '--skip-pubmed',
        action='store_true',
        help='Skip PubMed search'
    )
    parser.add_argument(
        '--skip-arxiv',
        action='store_true',
        help='Skip arXiv search'
    )
    parser.add_argument(
        '--skip-journals',
        action='store_true',
        help='Skip journal search'
    )
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = LiteratureSearcher(
        config_dir=args.config_dir,
        email=args.email
    )
    
    # Perform search
    logger.info(f"Starting literature search (looking back {args.date_range} days)")
    papers = searcher.search_all(
        date_range=args.date_range,
        include_pubmed=not args.skip_pubmed,
        include_arxiv=not args.skip_arxiv,
        include_journals=not args.skip_journals
    )
    
    # Save results
    if not args.dry_run:
        searcher.save_results(papers, args.output_dir)
    else:
        logger.info(f"Dry run: Found {len(papers)} papers (not saving)")
        
        # Print sample results
        if papers:
            logger.info("\nSample papers found:")
            for i, paper in enumerate(papers[:3], 1):
                logger.info(f"\n{i}. {paper.title}")
                logger.info(f"   Authors: {', '.join(paper.authors[:3])}")
                logger.info(f"   Source: {paper.source}")
                logger.info(f"   Date: {paper.publication_date}")
    
    logger.info("Literature search complete")


if __name__ == "__main__":
    main()
