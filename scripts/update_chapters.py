#!/usr/bin/env python3
"""
Chapter Update Module for Healthcare AI Equity Textbook

Updates chapter files with relevant papers and citations.

Author: Sanjay Basu, MD PhD
License: MIT
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a formatted citation."""
    key: str
    text: str
    paper_data: Dict


class ChapterUpdater:
    """Update chapter files with new papers and citations."""
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize the chapter updater.
        
        Args:
            dry_run: If True, don't write changes to files
        """
        self.dry_run = dry_run
        self.updates_made = {}
    
    def format_authors(self, authors: List[str], max_authors: int = 3) -> str:
        """
        Format author list for citations.
        
        Args:
            authors: List of author names
            max_authors: Maximum authors to show before using "et al."
            
        Returns:
            Formatted author string
        """
        if not authors:
            return "Anonymous"
        
        if len(authors) <= max_authors:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} and {authors[1]}"
            else:
                return ", ".join(authors[:-1]) + f", and {authors[-1]}"
        else:
            return f"{authors[0]} et al."
    
    def generate_citation_key(self, paper: Dict) -> str:
        """
        Generate a citation key for a paper.
        
        Format: FirstAuthorLastName:Year
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Citation key string
        """
        if not paper.get('authors'):
            first_author = "anonymous"
        else:
            # Extract last name from first author
            first_author_full = paper['authors'][0]
            # Handle formats like "Smith J" or "Smith, John"
            parts = first_author_full.replace(',', '').split()
            first_author = parts[0].lower()
        
        # Extract year from publication date
        pub_date = paper.get('publication_date', '')
        year = pub_date[:4] if pub_date else 'unknown'
        
        # Create base key
        key = f"{first_author}:{year}"
        
        return key
    
    def format_jmlr_citation(self, paper: Dict) -> str:
        """
        Format citation in JMLR style.
        
        JMLR format:
        Authors. Title. Journal/Conference, Volume:Pages, Year. doi:DOI
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Formatted citation string
        """
        # Format authors
        authors = self.format_authors(paper.get('authors', []))
        
        # Get title
        title = paper.get('title', 'Untitled')
        
        # Get venue (journal or arXiv)
        if paper.get('source') == 'arxiv':
            venue = f"arXiv preprint {paper.get('source_id', '')}"
        else:
            venue = paper.get('journal', 'Unknown venue')
        
        # Get year
        pub_date = paper.get('publication_date', '')
        year = pub_date[:4] if pub_date else 'n.d.'
        
        # Build citation
        citation = f"{authors}. {title}. *{venue}*, {year}."
        
        # Add DOI if available
        if paper.get('doi'):
            citation += f" doi:[{paper['doi']}](https://doi.org/{paper['doi']})"
        elif paper.get('url'):
            citation += f" [{paper['url']}]({paper['url']})"
        
        return citation
    
    def extract_existing_citations(self, content: str) -> Set[str]:
        """
        Extract existing citation keys from chapter content.
        
        Args:
            content: Chapter markdown content
            
        Returns:
            Set of citation keys
        """
        # Find bibliography section
        bib_pattern = r'## References\s+(.*?)(?=\n##|\Z)'
        bib_match = re.search(bib_pattern, content, re.DOTALL)
        
        if not bib_match:
            return set()
        
        bib_section = bib_match.group(1)
        
        # Extract citation keys (format: <a name="author:year"></a>)
        key_pattern = r'<a name="([^"]+)"></a>'
        keys = re.findall(key_pattern, bib_section)
        
        return set(keys)
    
    def find_insertion_point(self, content: str, section_name: Optional[str] = None) -> int:
        """
        Find appropriate insertion point in chapter for new content.
        
        Args:
            content: Chapter content
            section_name: Optional section name to insert near
            
        Returns:
            Character position for insertion
        """
        # If section specified, try to find it
        if section_name:
            section_pattern = rf'## {re.escape(section_name)}\s+'
            match = re.search(section_pattern, content)
            if match:
                # Insert at end of section (before next ## heading)
                next_section = re.search(r'\n##[^#]', content[match.end():])
                if next_section:
                    return match.end() + next_section.start()
                else:
                    # Insert before bibliography
                    bib_match = re.search(r'\n## References\s+', content)
                    if bib_match:
                        return bib_match.start()
        
        # Default: insert before bibliography
        bib_match = re.search(r'\n## References\s+', content)
        if bib_match:
            return bib_match.start()
        
        # If no bibliography, insert at end
        return len(content)
    
    def update_bibliography(
        self,
        content: str,
        new_citations: List[Citation]
    ) -> str:
        """
        Update bibliography section with new citations.
        
        Args:
            content: Chapter content
            new_citations: List of new Citation objects
            
        Returns:
            Updated content
        """
        # Find bibliography section
        bib_pattern = r'(## References\s+)(.*?)(?=\n##|\Z)'
        bib_match = re.search(bib_pattern, content, re.DOTALL)
        
        if not bib_match:
            # Create bibliography section if it doesn't exist
            content += "\n\n## References\n\n"
            bib_match = re.search(bib_pattern, content, re.DOTALL)
        
        existing_bib = bib_match.group(2)
        
        # Add new citations
        new_bib_entries = []
        for citation in new_citations:
            entry = f'<a name="{citation.key}"></a>\n{citation.text}\n'
            new_bib_entries.append(entry)
        
        # Combine and sort alphabetically
        updated_bib = existing_bib + "\n" + "\n".join(new_bib_entries)
        
        # Replace bibliography section
        updated_content = content[:bib_match.start(2)] + updated_bib + content[bib_match.end(2):]
        
        return updated_content
    
    def create_citation_context(
        self,
        paper: Dict,
        assessment: Dict
    ) -> str:
        """
        Create contextual text for citing a paper.
        
        Args:
            paper: Paper metadata
            assessment: Relevance assessment data
            
        Returns:
            Markdown text with citation
        """
        key = self.generate_citation_key(paper)
        contributions = assessment.get('key_contributions', [])
        
        if not contributions:
            return ""
        
        # Create a brief mention of the paper with citation
        context = f"\nRecent work has advanced this area"
        
        if len(contributions) == 1:
            context += f" by {contributions[0].lower()}"
        elif len(contributions) > 1:
            context += " through several contributions: "
            context += ", ".join(c.lower() for c in contributions[:-1])
            context += f", and {contributions[-1].lower()}"
        
        context += f" [[{key}]](#{key})."
        
        return context
    
    def update_chapter(
        self,
        chapter_file: Path,
        relevant_papers: List[Dict]
    ) -> bool:
        """
        Update a single chapter with relevant papers.
        
        Args:
            chapter_file: Path to chapter file
            relevant_papers: List of dicts with 'paper' and 'assessment' keys
            
        Returns:
            True if updates were made, False otherwise
        """
        logger.info(f"Updating {chapter_file.name}")
        
        try:
            # Read chapter content
            with open(chapter_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract existing citations
            existing_citations = self.extract_existing_citations(content)
            logger.info(f"  Found {len(existing_citations)} existing citations")
            
            # Filter to only new papers
            new_papers = []
            for item in relevant_papers:
                paper = item['paper']
                key = self.generate_citation_key(paper)
                if key not in existing_citations:
                    new_papers.append(item)
            
            if not new_papers:
                logger.info(f"  No new papers to add")
                return False
            
            logger.info(f"  Adding {len(new_papers)} new papers")
            
            # Create citations
            new_citations = []
            for item in new_papers:
                paper = item['paper']
                key = self.generate_citation_key(paper)
                citation_text = self.format_jmlr_citation(paper)
                
                citation = Citation(
                    key=key,
                    text=citation_text,
                    paper_data=paper
                )
                new_citations.append(citation)
            
            # Update bibliography
            updated_content = self.update_bibliography(content, new_citations)
            
            # Optionally add contextual mentions in the chapter body
            # (Conservative approach: only add to bibliography for now)
            # This could be enhanced to use Claude API to suggest integration points
            
            # Write updated content
            if not self.dry_run:
                with open(chapter_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                logger.info(f"  â Updated {chapter_file.name}")
            else:
                logger.info(f"  [DRY RUN] Would update {chapter_file.name}")
            
            # Track updates
            self.updates_made[chapter_file.name] = len(new_papers)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating {chapter_file}: {e}")
            return False
    
    def update_all_chapters(
        self,
        assessments_file: Path,
        chapters_dir: Path
    ) -> Dict[str, int]:
        """
        Update all chapters based on relevance assessments.
        
        Args:
            assessments_file: Path to relevance assessments JSON
            chapters_dir: Directory containing chapter files
            
        Returns:
            Dictionary mapping chapter files to number of updates
        """
        logger.info(f"Loading assessments from {assessments_file}")
        
        # Load assessments
        with open(assessments_file, 'r') as f:
            data = json.load(f)
        
        chapters_data = data.get('chapters', {})
        logger.info(f"Found assessments for {len(chapters_data)} chapters")
        
        # Update each chapter
        for chapter_filename, chapter_data in chapters_data.items():
            relevant_papers = chapter_data.get('papers', [])
            
            if not relevant_papers:
                continue
            
            # Filter to high-confidence papers
            high_quality_papers = [
                p for p in relevant_papers
                if p['assessment']['should_include']
            ]
            
            if not high_quality_papers:
                logger.info(f"Skipping {chapter_filename}: no high-quality papers")
                continue
            
            chapter_file = chapters_dir / chapter_filename
            if not chapter_file.exists():
                logger.warning(f"Chapter file not found: {chapter_file}")
                continue
            
            self.update_chapter(chapter_file, high_quality_papers)
        
        return self.updates_made
    
    def generate_summary(self) -> Dict:
        """
        Generate summary of updates.
        
        Returns:
            Summary dictionary
        """
        total_papers = sum(self.updates_made.values())
        
        summary = {
            'update_date': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'chapters_updated': len(self.updates_made),
            'total_papers_added': total_papers,
            'updates_by_chapter': self.updates_made
        }
        
        return summary


def main():
    """Main entry point for chapter updates."""
    parser = argparse.ArgumentParser(
        description="Update textbook chapters with relevant papers"
    )
    parser.add_argument(
        '--papers-file',
        type=str,
        default='data/relevant_papers.json',
        help='Path to relevance assessments file'
    )
    parser.add_argument(
        '--chapters-dir',
        type=str,
        default='_chapters',
        help='Directory containing chapter files'
    )
    parser.add_argument(
        '--dry-run',
        type=bool,
        default=False,
        help='Perform dry run without writing files'
    )
    parser.add_argument(
        '--summary-file',
        type=str,
        default='data/update_summary.json',
        help='Path to save update summary'
    )
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = ChapterUpdater(dry_run=args.dry_run)
    
    # Update chapters
    chapters_dir = Path(args.chapters_dir)
    assessments_file = Path(args.papers_file)
    
    if not assessments_file.exists():
        logger.error(f"Assessments file not found: {assessments_file}")
        return
    
    if not chapters_dir.exists():
        logger.error(f"Chapters directory not found: {chapters_dir}")
        return
    
    logger.info("Starting chapter updates...")
    updates = updater.update_all_chapters(assessments_file, chapters_dir)
    
    # Generate summary
    summary = updater.generate_summary()
    
    # Save summary
    summary_file = Path(args.summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Log summary
    logger.info("\n=== Update Summary ===")
    logger.info(f"Chapters updated: {summary['chapters_updated']}")
    logger.info(f"Total papers added: {summary['total_papers_added']}")
    logger.info(f"Dry run: {summary['dry_run']}")
    
    if updates:
        logger.info("\nUpdates by chapter:")
        for chapter, count in sorted(updates.items()):
            logger.info(f"  {chapter}: {count} papers")
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("Chapter updates complete")


if __name__ == "__main__":
    main()
