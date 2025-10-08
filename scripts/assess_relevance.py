#!/usr/bin/env python3
"""
Relevance Assessment Module for Healthcare AI Equity Textbook

Uses Claude API to assess relevance of papers to specific chapters.

Author: Sanjay Basu, MD PhD
License: MIT
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import time

from anthropic import Anthropic, RateLimitError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RelevanceAssessment:
    """Represents a relevance assessment for a paper-chapter pair."""
    paper_id: str
    chapter_file: str
    relevance_score: float  # 0.0 to 1.0
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str
    should_include: bool
    suggested_section: Optional[str] = None
    key_contributions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RelevanceAssessor:
    """Assess relevance of papers to textbook chapters using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the relevance assessor.
        
        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"
        
    def _load_chapter_metadata(self, chapter_path: Path) -> Dict:
        """
        Extract metadata from a chapter markdown file.
        
        Args:
            chapter_path: Path to chapter file
            
        Returns:
            Dictionary with chapter metadata
        """
        try:
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title (first H1)
            title = ""
            for line in content.split('\n'):
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            # Extract learning objectives
            objectives = []
            in_objectives = False
            for line in content.split('\n'):
                if 'learning objectives' in line.lower():
                    in_objectives = True
                    continue
                if in_objectives:
                    if line.startswith('- ') or line.startswith('* '):
                        objectives.append(line[2:].strip())
                    elif line.startswith('#'):
                        break
            
            # Get first 1000 characters as context
            context = content[:1000]
            
            return {
                'title': title,
                'filename': chapter_path.name,
                'objectives': objectives,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error loading chapter {chapter_path}: {e}")
            return {
                'title': chapter_path.stem,
                'filename': chapter_path.name,
                'objectives': [],
                'context': ''
            }
    
    def _build_assessment_prompt(
        self,
        paper: Dict,
        chapter: Dict
    ) -> str:
        """
        Build prompt for Claude to assess relevance.
        
        Args:
            paper: Paper metadata dictionary
            chapter: Chapter metadata dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert academic editor for a comprehensive healthcare AI textbook focused on equity and serving underserved populations. Your task is to assess whether a research paper is relevant and valuable for inclusion in a specific chapter.

CHAPTER INFORMATION:
Title: {chapter['title']}
File: {chapter['filename']}

Learning Objectives:
{chr(10).join(f"- {obj}" for obj in chapter['objectives']) if chapter['objectives'] else "Not specified"}

Chapter Context:
{chapter['context']}

PAPER INFORMATION:
Title: {paper['title']}
Authors: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}
Source: {paper['source']}
Date: {paper['publication_date']}
{f"Journal: {paper['journal']}" if paper.get('journal') else ''}

Abstract:
{paper['abstract']}

ASSESSMENT CRITERIA:
1. **Topical Relevance**: Does the paper's content align with the chapter's focus?
2. **Academic Quality**: Is this high-quality, peer-reviewed or preprint research?
3. **Equity Focus**: Does the paper address health equity, bias, disparities, or underserved populations?
4. **Technical Contribution**: Does it advance methods, algorithms, or practical applications?
5. **Recency**: Is the research current and representative of state-of-the-art?
6. **Pedagogical Value**: Would this help readers understand concepts or implementations?

RESPONSE FORMAT:
Provide your assessment in the following JSON format:

{{
  "relevance_score": <float between 0.0 and 1.0>,
  "confidence": "<high|medium|low>",
  "should_include": <true|false>,
  "reasoning": "<2-3 sentence explanation>",
  "suggested_section": "<section name or null>",
  "key_contributions": [<list of 2-4 key contributions>]
}}

SCORING GUIDANCE:
- 0.9-1.0: Highly relevant, must include
- 0.7-0.89: Very relevant, strongly recommended
- 0.5-0.69: Moderately relevant, include if space permits
- 0.3-0.49: Somewhat relevant, probably skip
- 0.0-0.29: Not relevant, definitely skip

For healthcare AI equity focus:
- Heavily weight papers addressing bias, fairness, disparities
- Favor papers with real-world clinical validation
- Prefer papers focused on underserved/marginalized populations
- Value papers with practical implementation details

Provide only the JSON response, no additional text."""

        return prompt
    
    def assess_paper_chapter_relevance(
        self,
        paper: Dict,
        chapter: Dict,
        max_retries: int = 3
    ) -> RelevanceAssessment:
        """
        Assess relevance of a paper to a chapter using Claude.
        
        Args:
            paper: Paper metadata dictionary
            chapter: Chapter metadata dictionary
            max_retries: Maximum number of retry attempts
            
        Returns:
            RelevanceAssessment object
        """
        prompt = self._build_assessment_prompt(paper, chapter)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                # Extract JSON from response
                response_text = response.content[0].text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                # Parse JSON
                assessment_data = json.loads(response_text)
                
                # Create RelevanceAssessment object
                return RelevanceAssessment(
                    paper_id=paper['source_id'],
                    chapter_file=chapter['filename'],
                    relevance_score=float(assessment_data['relevance_score']),
                    confidence=assessment_data['confidence'],
                    reasoning=assessment_data['reasoning'],
                    should_include=bool(assessment_data['should_include']),
                    suggested_section=assessment_data.get('suggested_section'),
                    key_contributions=assessment_data.get('key_contributions')
                )
                
            except RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded, max retries reached")
                    raise
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response text: {response_text}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    # Return default low-relevance assessment
                    return RelevanceAssessment(
                        paper_id=paper['source_id'],
                        chapter_file=chapter['filename'],
                        relevance_score=0.0,
                        confidence='low',
                        reasoning="Failed to parse API response",
                        should_include=False
                    )
                    
            except APIError as e:
                logger.error(f"API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise
    
    def assess_all_papers(
        self,
        papers: List[Dict],
        chapters_dir: Path,
        threshold: float = 0.5
    ) -> Dict[str, List[RelevanceAssessment]]:
        """
        Assess all papers against all chapters.
        
        Args:
            papers: List of paper dictionaries
            chapters_dir: Directory containing chapter files
            threshold: Minimum relevance score for inclusion
            
        Returns:
            Dictionary mapping chapter files to relevant papers
        """
        logger.info(f"Assessing {len(papers)} papers against chapters in {chapters_dir}")
        
        # Load all chapters
        chapter_files = sorted(chapters_dir.glob("*.md"))
        chapters = [self._load_chapter_metadata(f) for f in chapter_files]
        
        logger.info(f"Found {len(chapters)} chapters")
        
        # Store assessments by chapter
        chapter_assessments: Dict[str, List[RelevanceAssessment]] = {
            ch['filename']: [] for ch in chapters
        }
        
        total_assessments = len(papers) * len(chapters)
        completed = 0
        
        # Assess each paper against each chapter
        for paper in papers:
            logger.info(f"Assessing: {paper['title'][:80]}...")
            
            for chapter in chapters:
                try:
                    assessment = self.assess_paper_chapter_relevance(paper, chapter)
                    
                    # Only store if meets threshold
                    if assessment.relevance_score >= threshold:
                        chapter_assessments[chapter['filename']].append(assessment)
                        logger.info(
                            f"  ✓ {chapter['filename']}: "
                            f"score={assessment.relevance_score:.2f}, "
                            f"include={assessment.should_include}"
                        )
                    
                    completed += 1
                    
                    # Rate limiting - Claude API typically allows 50 requests/min
                    time.sleep(1.2)
                    
                except Exception as e:
                    logger.error(f"Error assessing paper for {chapter['filename']}: {e}")
                    completed += 1
                    continue
            
            # Progress logging
            progress = (completed / total_assessments) * 100
            logger.info(f"Progress: {completed}/{total_assessments} ({progress:.1f}%)")
        
        # Summary
        total_relevant = sum(len(assessments) for assessments in chapter_assessments.values())
        logger.info(f"\nAssessment complete: {total_relevant} relevant paper-chapter pairs found")
        
        for chapter_file, assessments in chapter_assessments.items():
            if assessments:
                logger.info(f"  {chapter_file}: {len(assessments)} relevant papers")
        
        return chapter_assessments
    
    def save_assessments(
        self,
        assessments: Dict[str, List[RelevanceAssessment]],
        papers: List[Dict],
        output_file: Path
    ):
        """
        Save relevance assessments to JSON file.
        
        Args:
            assessments: Dictionary of assessments by chapter
            papers: Original list of papers
            output_file: Path to output file
        """
        # Create paper lookup
        paper_lookup = {p['source_id']: p for p in papers}
        
        # Build output structure
        output_data = {
            'assessment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_papers_assessed': len(papers),
            'chapters': {}
        }
        
        for chapter_file, chapter_assessments in assessments.items():
            if not chapter_assessments:
                continue
            
            output_data['chapters'][chapter_file] = {
                'relevant_papers_count': len(chapter_assessments),
                'papers': []
            }
            
            for assessment in chapter_assessments:
                paper = paper_lookup.get(assessment.paper_id)
                if paper:
                    paper_with_assessment = {
                        'paper': paper,
                        'assessment': assessment.to_dict()
                    }
                    output_data['chapters'][chapter_file]['papers'].append(paper_with_assessment)
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved assessments to {output_file}")


def main():
    """Main entry point for relevance assessment."""
    parser = argparse.ArgumentParser(
        description="Assess relevance of papers to textbook chapters"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/literature',
        help='Directory containing search results'
    )
    parser.add_argument(
        '--chapters-dir',
        type=str,
        default='chapters',
        help='Directory containing chapter files'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/relevant_papers.json',
        help='Output file for relevance assessments'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum relevance score for inclusion'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode with single paper and chapter'
    )
    
    args = parser.parse_args()
    
    # Load papers
    input_file = Path(args.input_dir) / 'papers.json'
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    with open(input_file, 'r') as f:
        data = json.load(f)
        papers = data.get('papers', [])
    
    logger.info(f"Loaded {len(papers)} papers")
    
    if args.test_mode:
        logger.info("Running in test mode (first paper and chapter only)")
        papers = papers[:1]
    
    # Initialize assessor
    assessor = RelevanceAssessor()
    
    # Assess papers
    chapters_dir = Path(args.chapters_dir)
    assessments = assessor.assess_all_papers(
        papers=papers,
        chapters_dir=chapters_dir,
        threshold=args.threshold
    )
    
    # Save results
    output_file = Path(args.output_file)
    assessor.save_assessments(assessments, papers, output_file)
    
    logger.info("Relevance assessment complete")


if __name__ == "__main__":
    main()
