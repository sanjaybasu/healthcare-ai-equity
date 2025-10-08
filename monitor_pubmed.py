#!/usr/bin/env python3
"""
PubMed Literature Monitor
Monitors PubMed for new papers relevant to healthcare AI and equity
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from Bio import Entrez
import time

# Configure email for PubMed API
Entrez.email = os.environ.get('NCBI_EMAIL', 'sanjay.basu@waymarkcare.com')

# Search terms for each chapter topic
CHAPTER_QUERIES = {
    'chapter_01': 'healthcare informatics AND health equity',
    'chapter_02': 'machine learning mathematics AND healthcare',
    'chapter_03': 'FHIR AND healthcare data engineering',
    'chapter_04': 'machine learning fairness AND healthcare',
    'chapter_05': 'deep learning AND clinical applications',
    'chapter_06': 'natural language processing AND clinical text',
    'chapter_07': 'medical imaging AND computer vision AND fairness',
    'chapter_08': 'time series AND clinical data',
    'chapter_09': 'clinical NLP AND knowledge graphs',
    'chapter_10': 'survival analysis AND machine learning',
    'chapter_11': 'causal inference AND healthcare',
    'chapter_12': 'federated learning AND healthcare privacy',
    'chapter_13': 'transfer learning AND medical AI',
    'chapter_14': 'explainable AI AND healthcare',
    'chapter_15': 'clinical AI validation',
    'chapter_16': 'uncertainty quantification AND medical AI',
    'chapter_17': 'FDA AND AI medical devices',
    'chapter_18': 'implementation science AND clinical AI',
    'chapter_19': 'human-AI collaboration AND healthcare',
    'chapter_20': 'AI monitoring AND distribution shift',
    'chapter_21': 'health equity metrics AND AI',
    'chapter_22': 'clinical decision support systems',
    'chapter_23': 'precision medicine AND genomics AND AI',
    'chapter_24': 'population health AND risk stratification',
    'chapter_25': 'social determinants of health AND machine learning',
    'chapter_26': 'large language models AND healthcare',
    'chapter_27': 'multimodal learning AND clinical AI',
    'chapter_28': 'continual learning AND medical AI',
    'chapter_29': 'global health AND AI AND resource-limited',
    'chapter_30': 'algorithmic fairness AND health disparities'
}

# High-impact journals
PRIORITY_JOURNALS = [
    'Nature', 'Science', 'New England Journal of Medicine', 
    'JAMA', 'The Lancet', 'Nature Medicine', 'NEJM AI',
    'Nature Machine Intelligence', 'Nature Biomedical Engineering'
]


def search_pubmed(query: str, days_back: int = 7) -> List[Dict[str, Any]]:
    """
    Search PubMed for recent papers matching query
    
    Args:
        query: PubMed search query
        days_back: Number of days to search back
    
    Returns:
        List of paper metadata dictionaries
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    date_query = f"{query} AND {start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[PDAT]"
    
    try:
        # Search PubMed
        handle = Entrez.esearch(
            db='pubmed',
            term=date_query,
            retmax=50,
            sort='relevance'
        )
        results = Entrez.read(handle)
        handle.close()
        
        id_list = results['IdList']
        
        if not id_list:
            return []
        
        # Fetch paper details
        time.sleep(0.5)  # Rate limiting
        handle = Entrez.efetch(
            db='pubmed',
            id=id_list,
            rettype='medline',
            retmode='xml'
        )
        records = Entrez.read(handle)
        handle.close()
        
        papers = []
        for record in records['PubmedArticle']:
            try:
                article = record['MedlineCitation']['Article']
                
                # Extract metadata
                paper = {
                    'pmid': record['MedlineCitation']['PMID'],
                    'title': article['ArticleTitle'],
                    'abstract': article.get('Abstract', {}).get('AbstractText', [''])[0] if 'Abstract' in article else '',
                    'journal': article['Journal']['Title'],
                    'pub_date': article['Journal']['JournalIssue'].get('PubDate', {}),
                    'authors': [
                        f"{author.get('LastName', '')} {author.get('Initials', '')}"
                        for author in article.get('AuthorList', [])
                    ],
                    'doi': None,
                    'is_high_impact': article['Journal']['Title'] in PRIORITY_JOURNALS
                }
                
                # Extract DOI if available
                if 'ELocationID' in article:
                    for eloc in article['ELocationID']:
                        if eloc.attributes.get('EIdType') == 'doi':
                            paper['doi'] = str(eloc)
                            break
                
                papers.append(paper)
                
            except (KeyError, IndexError) as e:
                print(f"Error parsing record: {e}")
                continue
        
        return papers
        
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []


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
    output_file = f"{output_dir}/{chapter_id}_pubmed_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'chapter': chapter_id,
            'source': 'pubmed',
            'search_date': datetime.now().isoformat(),
            'papers': papers
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(papers)} papers for {chapter_id}")


def main():
    """Main execution function"""
    print("Starting PubMed literature monitoring...")
    print(f"Search date: {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 80)
    
    total_papers = 0
    high_impact_papers = 0
    
    for chapter_id, query in CHAPTER_QUERIES.items():
        print(f"\nSearching {chapter_id}: {query}")
        
        papers = search_pubmed(query, days_back=7)
        
        if papers:
            save_results(chapter_id, papers)
            total_papers += len(papers)
            high_impact_papers += sum(1 for p in papers if p['is_high_impact'])
            
            # Print high-impact papers
            high_impact = [p for p in papers if p['is_high_impact']]
            if high_impact:
                print(f"  Found {len(high_impact)} high-impact papers:")
                for paper in high_impact[:3]:  # Show top 3
                    print(f"    - {paper['title'][:80]}...")
        
        time.sleep(0.5)  # Rate limiting
    
    print("\n" + "=" * 80)
    print(f"Total papers found: {total_papers}")
    print(f"High-impact papers: {high_impact_papers}")
    print("PubMed monitoring complete!")
    
    # Set GitHub Actions output
    if high_impact_papers >= 5:
        print("::set-output name=major_updates::true")
        os.environ['MAJOR_UPDATES'] = 'true'


if __name__ == '__main__':
    main()
