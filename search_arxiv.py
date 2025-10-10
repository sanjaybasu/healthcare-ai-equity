#!/usr/bin/env python3
"""
arXiv Search for Healthcare AI Papers

Searches arXiv for recent preprints relevant to healthcare AI and population stratification.
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import time

import arxiv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# arXiv search queries for different topics
ARXIV_QUERIES = {
    "fairness_healthcare": 'cat:cs.LG AND (all:"fairness" OR all:"bias" OR all:"equity") AND (all:"healthcare" OR all:"medical" OR all:"clinical")',
    "medical_imaging": 'cat:cs.CV AND (all:"medical imaging" OR all:"radiology" OR all:"pathology" OR all:"x-ray" OR all:"CT" OR all:"MRI")',
    "clinical_nlp": 'cat:cs.CL AND (all:"clinical" OR all:"medical" OR all:"healthcare" OR all:"EHR" OR all:"electronic health")',
    "survival_analysis": 'cat:stat.ML AND (all:"survival analysis" OR all:"time-to-event" OR all:"competing risks")',
    "causal_inference": 'cat:stat.ME AND (all:"causal inference" OR all:"treatment effects" OR all:"instrumental variables") AND (all:"healthcare" OR all:"medical")',
    "federated_learning_health": 'cat:cs.LG AND all:"federated learning" AND (all:"healthcare" OR all:"medical" OR all:"clinical")',
    "interpretable_ml": 'cat:cs.LG AND (all:"interpretable" OR all:"explainable" OR all:"SHAP" OR all:"LIME") AND (all:"healthcare" OR all:"clinical")',
    "llm_healthcare": 'cat:cs.CL AND (all:"large language model" OR all:"GPT" OR all:"BERT") AND (all:"healthcare" OR all:"medical" OR all:"clinical")',
    "multimodal_health": 'cat:cs.LG AND all:"multimodal" AND (all:"healthcare" OR all:"medical" OR all:"clinical")',
    "continual_learning": 'cat:cs.LG AND (all:"continual learning" OR all:"lifelong learning" OR all:"catastrophic forgetting")',
}


class ArxivSearcher:
    """Search arXiv for relevant papers."""
    
    def __init__(self):
        """Initialize arXiv searcher."""
        self.client = arxiv.Client()
    
    def search_papers(
        self,
        query: str,
        days_back: int = 7,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers.
        
        Args:
            query: arXiv query string
            days_back: Number of days back to search
            max_results: Maximum results to return
            
        Returns:
            List of paper metadata
        """
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days_back)
        
        logger.info(f"Searching arXiv: {query}")
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                # Check if paper is recent enough
                if result.published.replace(tzinfo=None) < date_threshold:
                    continue
                
                # Extract authors
                authors = [
                    f"{author.name.split()[-1]} {author.name.split()[0][0]}"
                    for author in result.authors[:5]
                ]
                
                paper = {
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': authors,
                    'journal': 'arXiv preprint',
                    'year': result.published.year,
                    'doi': result.doi,
                    'pdf_url': result.pdf_url,
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'query': query,
                    'categories': result.categories,
                    'retrieved_date': datetime.now().isoformat()
                }
                
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} recent papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def search_all_topics(
        self,
        days_back: int = 7,
        output_dir: Path = Path("data/papers")
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search arXiv for all topics.
        
        Args:
            days_back: Number of days back to search
            output_dir: Output directory
            
        Returns:
            Dictionary of results by topic
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for topic, query in ARXIV_QUERIES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing topic: {topic}")
            logger.info(f"{'='*60}")
            
            papers = self.search_papers(query, days_back=days_back)
            all_results[topic] = papers
            
            # Save results
            topic_file = output_dir / f"arxiv_{topic}_papers.json"
            with open(topic_file, 'w') as f:
                json.dump(papers, f, indent=2)
            
            logger.info(f"Saved {len(papers)} papers to {topic_file}")
            
            # Rate limiting
            time.sleep(3)
        
        # Save combined results
        combined_file = output_dir / "arxiv_all_papers.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\nSaved combined results to {combined_file}")
        
        return all_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Search arXiv for healthcare AI papers"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days back to search"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting arXiv search")
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Output directory: {args.output_dir}")
    
    searcher = ArxivSearcher()
    results = searcher.search_all_topics(
        days_back=args.days_back,
        output_dir=args.output_dir
    )
    
    # Print summary
    total_papers = sum(len(papers) for papers in results.values())
    
    logger.info(f"\n{'='*60}")
    logger.info("SEARCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total papers found: {total_papers}")
    logger.info(f"Topics processed: {len(results)}")
    
    for topic, papers in results.items():
        if papers:
            logger.info(f"  {topic}: {len(papers)} papers")
    
    return 0


if __name__ == "__main__":
    exit(main())
