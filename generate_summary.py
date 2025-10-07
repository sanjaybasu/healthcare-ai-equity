#!/usr/bin/env python3
"""
Generate Update Summary

Creates a summary of all updates for the pull request description.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_summary(analysis_dir: Path, output_file: Path) -> str:
    """
    Generate markdown summary of updates.
    
    Args:
        analysis_dir: Directory containing analysis results
        output_file: Output file path
        
    Returns:
        Summary text
    """
    # Load all analyses
    all_analyses_file = analysis_dir / "all_analyses.json"
    
    if not all_analyses_file.exists():
        logger.error(f"Analyses file not found: {all_analyses_file}")
        return "No analysis data available."
    
    with open(all_analyses_file, 'r') as f:
        all_analyses = json.load(f)
    
    # Count totals
    total_chapters = len(all_analyses)
    total_papers = sum(result['total_analyzed'] for result in all_analyses.values())
    total_relevant = sum(result['relevant_count'] for result in all_analyses.values())
    
    # Generate summary
    summary_lines = [
        f"# Weekly Literature Update - {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Summary",
        "",
        f"- **Chapters Analyzed:** {total_chapters}",
        f"- **Papers Reviewed:** {total_papers}",
        f"- **Relevant Papers Found:** {total_relevant}",
        "",
        "## Updates by Chapter",
        ""
    ]
    
    # Add chapter-specific details
    for chapter_id, result in sorted(all_analyses.items()):
        if result['relevant_count'] == 0:
            continue
        
        summary_lines.append(f"### {chapter_id}")
        summary_lines.append(f"- Papers analyzed: {result['total_analyzed']}")
        summary_lines.append(f"- Relevant papers: {result['relevant_count']}")
        
        # Get top papers
        analyses = result.get('analyses', [])
        if analyses:
            summary_lines.append("- Top papers:")
            for paper in analyses[:3]:  # Top 3
                title = paper.get('paper_title', 'Unknown')
                score = paper.get('relevance_score', 0)
                sota = paper.get('is_sota', False)
                equity = paper.get('has_equity_focus', False)
                
                tags = []
                if sota:
                    tags.append("SOTA")
                if equity:
                    tags.append("Equity")
                
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                summary_lines.append(f"  - {title} (relevance: {score}/10){tag_str}")
        
        summary_lines.append("")
    
    # Check for high-impact discoveries
    high_impact = []
    for chapter_id, result in all_analyses.items():
        for paper in result.get('analyses', []):
            if paper.get('relevance_score', 0) >= 9 and paper.get('is_sota', False):
                high_impact.append({
                    'chapter': chapter_id,
                    'title': paper.get('paper_title', 'Unknown'),
                    'score': paper.get('relevance_score', 0)
                })
    
    if high_impact:
        summary_lines.extend([
            "## High-Impact Discoveries",
            "",
            "*Papers with relevance score ≥ 9 and state-of-the-art methods:*",
            ""
        ])
        
        for item in sorted(high_impact, key=lambda x: x['score'], reverse=True):
            summary_lines.append(f"- **{item['title']}** ({item['chapter']}, score: {item['score']}/10)")
        
        summary_lines.append("")
    
    # Load AI lab announcements if available
    announcements_file = analysis_dir.parent / "announcements" / "all_announcements.json"
    if announcements_file.exists():
        with open(announcements_file, 'r') as f:
            announcements = json.load(f)
        
        total_announcements = sum(len(posts) for posts in announcements.values())
        
        if total_announcements > 0:
            summary_lines.extend([
                "## AI Lab Announcements",
                "",
                f"Found {total_announcements} healthcare-related announcements:",
                ""
            ])
            
            for source_id, posts in announcements.items():
                if posts:
                    summary_lines.append(f"**{source_id}:** {len(posts)} posts")
                    for post in posts[:2]:  # Show top 2
                        summary_lines.append(f"- {post.get('title', 'Unknown')}")
            
            summary_lines.append("")
    
    # Add next steps
    summary_lines.extend([
        "## Next Steps",
        "",
        "1. Review the generated `*_update_suggestions.json` files for each chapter",
        "2. Review the `*_update_summary.md` files for human-readable summaries",
        "3. Manually integrate priority updates into chapters",
        "4. Update bibliographies with new citations",
        "5. Test any new code examples",
        "6. Commit and push changes",
        "",
        "---",
        "",
        "*This summary was automatically generated by the weekly literature update workflow.*"
    ])
    
    summary_text = "\n".join(summary_lines)
    
    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Generated summary: {output_file}")
    
    return summary_text


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate update summary"
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Directory containing analysis results"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("UPDATE_SUMMARY.md"),
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    logger.info("Generating update summary")
    
    summary = generate_summary(
        analysis_dir=args.analysis_dir,
        output_file=args.output_file
    )
    
    logger.info("Summary generated successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())
