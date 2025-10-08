#!/usr/bin/env python3
"""
arXiv Literature Monitor
Monitors arXiv for new papers relevant to healthcare AI and equity
"""

import arxiv
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# arXiv category mappings for healthcare AI
ARXIV_CATEGORIES = [
    'cs.LG',  # Machine Learning
    'cs.AI',  # Artificial Intelligence
    'cs.CV',  # Computer Vision
    'stat.ML',  # Statistics - Machine Learning
    'q-bio.QM',  # Quantitative Methods
    'cs.CY',  # Computers and Society (includes fairness)
]

# Search queries for each chapter
CHAPTER_QUERIES = {
    'chapter_01': 'healthcare informatics OR health equity OR clinical data',
    'chapter_02': 'machine learning mathematics OR optimization healthcare',
    'chapter_03': 'FHIR OR healthcare data OR electronic health records',
    'chapter_04': 'fairness machine learning OR algorithmic bias',
    'chapter_05': 'deep learning clinical OR neural networks healthcare',
    'chapter_06': 'clinical NLP OR medical text OR named entity recognition',
    'chapter_07': 'medical imaging OR radiology AI OR pathology deep learning',
    'chapter_08': 'clinical time series OR physiological signals',
    'chapter_09': 'knowledge graphs healthcare OR clinical information retrieval',
    'chapter_10': 'survival analysis OR time-to-event OR competing risks',
    'chapter_11': 'causal inference healthcare OR treatment effects',
    'chapter_12': 'federated learning OR differential privacy healthcare',
    'chapter_13': 'transfer learning medical OR domain adaptation',
    'chapter_14': 'explainable AI OR interpretability healthcare',
    'chapter_15': 'clinical validation OR external validation AI',
    'chapter_16': 'uncertainty quantification OR calibration healthcare',
    'chapter_17': 'FDA medical device OR regulatory AI',
    'chapter_18': 'implementation science OR clinical AI deployment',
    'chapter_19': 'human-AI collaboration OR decision support',
    'chapter_20': 'distribution shift OR model monitoring healthcare',
    'chapter_21': 'fairness metrics OR health equity evaluation',
    'chapter_22': 'clinical decision support OR diagnostic AI',
    'chapter_23': 'precision medicine OR genomics machine learning',
    'chapter_24': 'population health OR risk stratification',
    'chapter_25': 'social determinants health OR neighborhood effects',
    'chapter_26': 'large language models healthcare OR clinical LLM',
    'chapter_27': 'multimodal learning OR medical image text fusion',
    'chapter_28': 'continual learning OR lifelong learning healthcare',
    'chapter_29': 'global health AI OR resource-limited settings',
    'chapter_30': 'algorithmic fairness OR health disparities AI'
}


def search_arxiv(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Search arXiv for recent papers matching query
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        List of paper metadata dictionaries
    """
    try:
        # Create search query with categories
        category_filter = ' OR '.join([f'cat:{cat}' for cat in ARXIV_CATEGORIES])
        full_query = f'({query}) AND ({category_filter})'
        
        # Search arXiv
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        week_ago = datetime.now() - timedelta(days=7)
        
        for result in search.results():
            # Only include papers from last week
            if result.published.replace(tzinfo=None) < week_ago:
                continue
            
            paper = {
                'arxiv_id': result.entry_id.split('/')[-1],
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'published': result.published.isoformat(),
                'updated': result.updated.isoformat(),
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'doi': result.doi,
                'journal_ref': result.journal_ref,
                'primary_category': result.primary_category,
                'comment': result.comment
            }
            
            papers.append(paper)
        
        return papers
        
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []


def is_relevant_paper(paper: Dict[str, Any]) -> bool:
    """
    Check if paper is relevant to healthcare AI and equity
    
    Args:
        paper: Paper metadata dictionary
    
    Returns:
        True if relevant, False otherwise
    """
    # Keywords indicating relevance
    healthcare_keywords = [
        'health', 'medical', 'clinical', 'patient', 'hospital',
        'disease', 'diagnosis', 'treatment', 'healthcare'
    ]
    
    equity_keywords = [
        'fairness', 'bias', 'equity', 'disparity', 'discrimination',
        'underserved', 'marginalized', 'vulnerable', 'disparate'
    ]
    
    text = (paper['title'] + ' ' + paper['abstract']).lower()
    
    # Must have at least one healthcare keyword OR one equity keyword
    has_healthcare = any(kw in text for kw in healthcare_keywords)
    has_equity = any(kw in text for kw in equity_keywords)
    
    return has_healthcare or has_equity


def save_results(chapter_id: str, papers: List[Dict[str, Any]]) -> None:
    """
    Save search results to JSON file
    
    Args:
        chapter_id: Chapter identifier
        papers: List of paper metadata
    """
    output_dir = 'literature_updates'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d')
    output_file = f"{output_dir}/{chapter_id}_arxiv_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'chapter': chapter_id,
            'source': 'arxiv',
            'search_date': datetime.now().isoformat(),
            'papers': papers
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(papers)} papers for {chapter_id}")


def main():
    """Main execution function"""
    print("Starting arXiv literature monitoring...")
    print(f"Search date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Monitoring categories: {', '.join(ARXIV_CATEGORIES)}")
    print("-" * 80)
    
    total_papers = 0
    relevant_papers = 0
    
    for chapter_id, query in CHAPTER_QUERIES.items():
        print(f"\nSearching {chapter_id}: {query[:60]}...")
        
        papers = search_arxiv(query, max_results=30)
        
        # Filter for relevance
        filtered_papers = [p for p in papers if is_relevant_paper(p)]
        
        if filtered_papers:
            save_results(chapter_id, filtered_papers)
            total_papers += len(papers)
            relevant_papers += len(filtered_papers)
            
            # Print sample papers
            if filtered_papers:
                print(f"  Found {len(filtered_papers)} relevant papers:")
                for paper in filtered_papers[:2]:  # Show top 2
                    print(f"    - {paper['title'][:80]}...")
    
    print("\n" + "=" * 80)
    print(f"Total papers found: {total_papers}")
    print(f"Relevant papers: {relevant_papers}")
    print("arXiv monitoring complete!")
    
    # Set GitHub Actions output for major updates
    if relevant_papers >= 10:
        print("::set-output name=major_updates::true")
        os.environ['MAJOR_UPDATES'] = 'true'


if __name__ == '__main__':
    main()
