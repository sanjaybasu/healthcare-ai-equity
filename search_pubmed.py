#!/usr/bin/env python3
"""
PubMed Literature Search for Healthcare AI Textbook

This script searches PubMed for recent papers relevant to each chapter
of the healthcare AI textbook and saves results for further analysis.
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import time

import requests
from Bio import Entrez

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Chapter-specific search queries
CHAPTER_QUERIES = {
    "chapter_01_clinical_informatics": [
        "electronic health records AND (bias OR disparity OR equity)",
        "clinical data standards AND interoperability",
        "health information systems AND underserved populations"
    ],
    "chapter_02_mathematical_foundations": [
        "mathematical foundations AND machine learning AND healthcare",
        "linear algebra AND medical imaging",
        "bayesian methods AND clinical prediction"
    ],
    "chapter_03_healthcare_data_engineering": [
        "healthcare data quality AND bias",
        "missing data AND electronic health records",
        "data engineering AND clinical informatics"
    ],
    "chapter_04_machine_learning_fundamentals": [
        "machine learning AND healthcare AND fairness",
        "supervised learning AND clinical prediction",
        "feature engineering AND electronic health records"
    ],
    "chapter_05_deep_learning_healthcare": [
        "deep learning AND healthcare AND equity",
        "neural networks AND clinical prediction",
        "convolutional neural networks AND medical imaging"
    ],
    "chapter_06_clinical_nlp": [
        "natural language processing AND clinical text AND bias",
        "clinical named entity recognition",
        "clinical language models AND fairness"
    ],
    "chapter_07_medical_imaging": [
        "medical image analysis AND bias",
        "computer vision AND radiology AND disparities",
        "deep learning AND medical imaging AND fairness"
    ],
    "chapter_08_causal_inference": [
        "causal inference AND observational data AND healthcare",
        "propensity score AND health disparities",
        "instrumental variables AND treatment effects"
    ],
    "chapter_09_advanced_nlp": [
        "large language models AND healthcare",
        "clinical question answering",
        "transformer models AND clinical text"
    ],
    "chapter_10_survival_analysis": [
        "survival analysis AND competing risks AND disparities",
        "time to event AND healthcare AND equity",
        "cox proportional hazards AND bias"
    ],
    "chapter_11_reinforcement_learning": [
        "reinforcement learning AND clinical decision support",
        "off-policy evaluation AND healthcare",
        "contextual bandits AND personalized medicine"
    ],
    "chapter_12_federated_learning": [
        "federated learning AND healthcare AND privacy",
        "distributed machine learning AND clinical data",
        "differential privacy AND medical data"
    ],
    "chapter_13_fairness_metrics": [
        "algorithmic fairness AND healthcare",
        "fairness metrics AND clinical prediction",
        "bias mitigation AND machine learning AND medicine"
    ],
    "chapter_14_interpretability": [
        "interpretable machine learning AND healthcare",
        "explainable AI AND clinical decision support",
        "SHAP AND medical AI"
    ],
    "chapter_15_validation": [
        "external validation AND machine learning AND healthcare",
        "model validation AND health disparities",
        "prospective validation AND clinical AI"
    ],
    "chapter_16_uncertainty_quantification": [
        "uncertainty quantification AND healthcare",
        "conformal prediction AND clinical models",
        "calibration AND clinical risk prediction"
    ],
    "chapter_26_llms_healthcare": [
        "large language models AND clinical applications",
        "GPT AND healthcare",
        "foundation models AND medicine AND bias"
    ],
    "chapter_27_multimodal_learning": [
        "multimodal learning AND healthcare",
        "vision language models AND medical imaging",
        "multimodal fusion AND clinical data"
    ],
    "chapter_28_continual_learning": [
        "continual learning AND healthcare",
        "catastrophic forgetting AND clinical models",
        "online learning AND medical AI"
    ],
}

# High-impact journals to prioritize
HIGH_IMPACT_JOURNALS = [
    "Nature", "Science", "Cell", "Lancet", "New England Journal of Medicine",
    "JAMA", "JAMA Internal Medicine", "JAMA Network Open",
    "Nature Medicine", "Nature Biomedical Engineering", "Nature Machine Intelligence",
    "Lancet Digital Health", "NEJM AI", "npj Digital Medicine",
    "Journal of the American Medical Informatics Association", "JMIR",
    "Journal of Machine Learning Research", "Machine Learning", "IEEE Transactions"
]


class PubMedSearcher:
    """Search PubMed for relevant papers."""
    
    def __init__(self, api_key: str, email: str = "sanjay@example.com"):
        """
        Initialize PubMed searcher.
        
        Args:
            api_key: NCBI API key
            email: Contact email for NCBI
        """
        self.api_key = api_key
        Entrez.email = email
        Entrez.api_key = api_key
    
    def search_papers(
        self,
        query: str,
        days_back: int = 7,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed for papers matching query.
        
        Args:
            query: Search query string
            days_back: Number of days back to search
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[pdat]"
        
        # Construct full query
        full_query = f"{query} AND {date_range}"
        
        logger.info(f"Searching PubMed: {full_query}")
        
        try:
            # Search for PMIDs
            handle = Entrez.esearch(
                db="pubmed",
                term=full_query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results["IdList"]
            
            if not pmids:
                logger.info(f"No results found for query: {query}")
                return []
            
            logger.info(f"Found {len(pmids)} papers")
            
            # Fetch paper details
            time.sleep(0.34)  # Rate limiting: 3 requests per second with API key
            handle = Entrez.efetch(
                db="pubmed",
                id=pmids,
                rettype="medline",
                retmode="xml"
            )
            papers_data = Entrez.read(handle)
            handle.close()
            
            # Extract relevant information
            papers = []
            for article in papers_data['PubmedArticle']:
                try:
                    medline = article['MedlineCitation']
                    article_info = medline['Article']
                    
                    # Extract authors
                    authors = []
                    if 'AuthorList' in article_info:
                        for author in article_info['AuthorList'][:5]:  # First 5 authors
                            if 'LastName' in author and 'Initials' in author:
                                authors.append(f"{author['LastName']} {author['Initials']}")
                    
                    # Extract journal
                    journal = article_info.get('Journal', {}).get('Title', 'Unknown')
                    
                    # Extract publication date
                    pub_date = article_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                    year = pub_date.get('Year', 'Unknown')
                    
                    # Extract DOI if available
                    doi = None
                    if 'ELocationID' in article_info:
                        for eloc in article_info['ELocationID']:
                            if eloc.attributes.get('EIdType') == 'doi':
                                doi = str(eloc)
                    
                    paper = {
                        'pmid': str(medline['PMID']),
                        'title': str(article_info.get('ArticleTitle', 'No title')),
                        'abstract': str(article_info.get('Abstract', {}).get('AbstractText', [''])[0]),
                        'authors': authors,
                        'journal': journal,
                        'year': year,
                        'doi': doi,
                        'query': query,
                        'high_impact': any(j.lower() in journal.lower() for j in HIGH_IMPACT_JOURNALS),
                        'retrieved_date': datetime.now().isoformat()
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def search_all_chapters(
        self,
        days_back: int = 7,
        output_dir: Path = Path("data/papers")
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for papers relevant to all chapters.
        
        Args:
            days_back: Number of days back to search
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping chapter IDs to lists of papers
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for chapter_id, queries in CHAPTER_QUERIES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {chapter_id}")
            logger.info(f"{'='*60}")
            
            chapter_papers = []
            
            for query in queries:
                papers = self.search_papers(query, days_back=days_back)
                chapter_papers.extend(papers)
                time.sleep(0.5)  # Additional delay between queries
            
            # Remove duplicates based on PMID
            seen_pmids = set()
            unique_papers = []
            for paper in chapter_papers:
                if paper['pmid'] not in seen_pmids:
                    seen_pmids.add(paper['pmid'])
                    unique_papers.append(paper)
            
            # Sort by high-impact first, then by date
            unique_papers.sort(key=lambda x: (not x['high_impact'], x['year']), reverse=True)
            
            all_results[chapter_id] = unique_papers
            
            # Save chapter results
            chapter_file = output_dir / f"{chapter_id}_papers.json"
            with open(chapter_file, 'w') as f:
                json.dump(unique_papers, f, indent=2)
            
            logger.info(f"Found {len(unique_papers)} unique papers for {chapter_id}")
            logger.info(f"  - {sum(p['high_impact'] for p in unique_papers)} from high-impact journals")
            logger.info(f"Saved to {chapter_file}")
        
        # Save combined results
        combined_file = output_dir / "all_papers.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\nSaved combined results to {combined_file}")
        
        return all_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Search PubMed for papers relevant to textbook chapters"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("PUBMED_API_KEY"),
        help="PubMed API key (or set PUBMED_API_KEY environment variable)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days back to search (default: 7)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers"),
        help="Output directory for results (default: data/papers)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("PubMed API key required. Set PUBMED_API_KEY environment variable or use --api-key")
        return 1
    
    logger.info("Starting PubMed literature search")
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Output directory: {args.output_dir}")
    
    searcher = PubMedSearcher(api_key=args.api_key)
    results = searcher.search_all_chapters(
        days_back=args.days_back,
        output_dir=args.output_dir
    )
    
    # Print summary
    total_papers = sum(len(papers) for papers in results.values())
    total_high_impact = sum(
        sum(p['high_impact'] for p in papers)
        for papers in results.values()
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("SEARCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total papers found: {total_papers}")
    logger.info(f"High-impact papers: {total_high_impact}")
    logger.info(f"Chapters processed: {len(results)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
