#!/usr/bin/env python3
"""
Monitor AI Lab Announcements

Monitors blogs and announcement pages from major AI research organizations
for relevant healthcare AI developments.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import time

import requests
from bs4 import BeautifulSoup
import feedparser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# AI lab RSS feeds and blog URLs
AI_LAB_SOURCES = {
    "openai": {
        "name": "OpenAI",
        "blog_url": "https://openai.com/blog",
        "rss_url": "https://openai.com/blog/rss.xml",
        "type": "rss"
    },
    "anthropic": {
        "name": "Anthropic",
        "blog_url": "https://www.anthropic.com/news",
        "type": "web"
    },
    "google_deepmind": {
        "name": "Google DeepMind",
        "blog_url": "https://deepmind.google/discover/blog/",
        "rss_url": "https://deepmind.google/blog/feed/basic/",
        "type": "rss"
    },
    "google_health": {
        "name": "Google Health AI",
        "blog_url": "https://blog.google/technology/health/",
        "type": "web"
    },
    "meta_ai": {
        "name": "Meta AI",
        "blog_url": "https://ai.meta.com/blog/",
        "rss_url": "https://ai.meta.com/blog/rss/",
        "type": "rss"
    },
}


class AILabMonitor:
    """Monitor AI lab announcements."""
    
    def __init__(self):
        """Initialize monitor."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Healthcare-AI-Textbook-Bot/1.0)'
        })
    
    def fetch_rss_feed(self, url: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch and parse RSS feed.
        
        Args:
            url: RSS feed URL
            days_back: Number of days back to include
            
        Returns:
            List of post metadata
        """
        try:
            feed = feedparser.parse(url)
            
            threshold = datetime.now().timestamp() - (days_back * 24 * 60 * 60)
            
            posts = []
            for entry in feed.entries:
                # Parse publication date
                if hasattr(entry, 'published_parsed'):
                    pub_time = time.mktime(entry.published_parsed)
                elif hasattr(entry, 'updated_parsed'):
                    pub_time = time.mktime(entry.updated_parsed)
                else:
                    continue
                
                # Check if recent enough
                if pub_time < threshold:
                    continue
                
                # Check if healthcare-related
                text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                healthcare_terms = ['health', 'medical', 'clinical', 'patient', 'diagnosis', 
                                  'treatment', 'disease', 'hospital', 'doctor', 'medicine']
                
                if not any(term in text for term in healthcare_terms):
                    continue
                
                post = {
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'summary': entry.get('summary', ''),
                    'published': datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d'),
                    'retrieved_date': datetime.now().isoformat()
                }
                
                posts.append(post)
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")
            return []
    
    def fetch_web_page(self, url: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch and parse web page for blog posts.
        
        Args:
            url: Blog page URL
            days_back: Number of days back to include
            
        Returns:
            List of post metadata
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # This is a simplified parser - would need customization per site
            posts = []
            
            # Find article links (common patterns)
            articles = soup.find_all(['article', 'div'], class_=['post', 'article', 'blog-post'])
            
            for article in articles[:20]:  # Limit to recent posts
                # Try to find title
                title_elem = article.find(['h1', 'h2', 'h3', 'a'])
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                
                # Check if healthcare-related
                text = title.lower()
                healthcare_terms = ['health', 'medical', 'clinical', 'patient', 'diagnosis']
                
                if not any(term in text for term in healthcare_terms):
                    continue
                
                # Try to find link
                link_elem = article.find('a', href=True)
                link = link_elem['href'] if link_elem else ''
                
                if link and not link.startswith('http'):
                    from urllib.parse import urljoin
                    link = urljoin(url, link)
                
                post = {
                    'title': title,
                    'url': link,
                    'summary': '',  # Would need more sophisticated extraction
                    'published': datetime.now().strftime('%Y-%m-%d'),
                    'retrieved_date': datetime.now().isoformat()
                }
                
                posts.append(post)
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching web page {url}: {e}")
            return []
    
    def monitor_all_sources(
        self,
        days_back: int = 7,
        output_dir: Path = Path("data/announcements")
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Monitor all AI lab sources.
        
        Args:
            days_back: Number of days back to check
            output_dir: Output directory
            
        Returns:
            Dictionary of posts by source
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_posts = {}
        
        for source_id, source_info in AI_LAB_SOURCES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Monitoring {source_info['name']}")
            logger.info(f"{'='*60}")
            
            posts = []
            
            if source_info['type'] == 'rss' and 'rss_url' in source_info:
                posts = self.fetch_rss_feed(source_info['rss_url'], days_back)
            elif source_info['type'] == 'web':
                posts = self.fetch_web_page(source_info['blog_url'], days_back)
            
            if posts:
                logger.info(f"Found {len(posts)} healthcare-related posts")
                
                # Save posts
                source_file = output_dir / f"{source_id}_posts.json"
                with open(source_file, 'w') as f:
                    json.dump(posts, f, indent=2)
                
                logger.info(f"Saved to {source_file}")
            else:
                logger.info("No healthcare-related posts found")
            
            all_posts[source_id] = posts
            
            # Rate limiting
            time.sleep(2)
        
        # Save combined results
        combined_file = output_dir / "all_announcements.json"
        with open(combined_file, 'w') as f:
            json.dump(all_posts, f, indent=2)
        
        logger.info(f"\nSaved combined results to {combined_file}")
        
        return all_posts


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Monitor AI lab announcements"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days back to check"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/announcements"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting AI lab monitoring")
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Output directory: {args.output_dir}")
    
    monitor = AILabMonitor()
    results = monitor.monitor_all_sources(
        days_back=args.days_back,
        output_dir=args.output_dir
    )
    
    # Print summary
    total_posts = sum(len(posts) for posts in results.values())
    
    logger.info(f"\n{'='*60}")
    logger.info("MONITORING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total healthcare-related posts: {total_posts}")
    
    for source_id, posts in results.items():
        if posts:
            source_name = AI_LAB_SOURCES[source_id]['name']
            logger.info(f"  {source_name}: {len(posts)} posts")
    
    return 0


if __name__ == "__main__":
    exit(main())
