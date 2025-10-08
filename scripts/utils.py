#!/usr/bin/env python3
"""
Utility Functions for Healthcare AI Equity Textbook Literature System

Common utilities for searching, parsing, and formatting.

Author: Sanjay Basu, MD PhD
License: MIT
"""

import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def normalize_title(title: str) -> str:
    """
    Normalize paper title for comparison.
    
    Args:
        title: Original title
        
    Returns:
        Normalized title (lowercase, no punctuation)
    """
    # Convert to lowercase
    normalized = title.lower()
    
    # Remove punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def extract_doi(text: str) -> Optional[str]:
    """
    Extract DOI from text.
    
    Args:
        text: Text potentially containing a DOI
        
    Returns:
        DOI string or None
    """
    # DOI pattern: 10.xxxx/xxxxx
    doi_pattern = r'10\.\d{4,}/[^\s]+'
    match = re.search(doi_pattern, text)
    
    if match:
        return match.group(0)
    
    return None


def parse_author_name(name: str) -> Dict[str, str]:
    """
    Parse author name into components.
    
    Args:
        name: Author name (various formats)
        
    Returns:
        Dictionary with first, last, middle, initials
    """
    # Handle formats: "Smith J", "Smith, John", "John Smith"
    name = name.strip()
    
    if ',' in name:
        # Format: "Last, First Middle"
        parts = name.split(',')
        last = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        first_parts = rest.split()
        first = first_parts[0] if first_parts else ""
        middle = ' '.join(first_parts[1:]) if len(first_parts) > 1 else ""
    else:
        # Format: "First Middle Last" or "Last F"
        parts = name.split()
        if len(parts) == 1:
            last = parts[0]
            first = ""
            middle = ""
        elif len(parts) == 2:
            # Could be "First Last" or "Last F"
            if len(parts[1]) <= 2:  # Likely initials
                last = parts[0]
                first = parts[1]
                middle = ""
            else:
                first = parts[0]
                last = parts[1]
                middle = ""
        else:
            # "First Middle Last"
            first = parts[0]
            last = parts[-1]
            middle = ' '.join(parts[1:-1])
    
    # Generate initials
    initials = ""
    if first:
        initials = first[0].upper()
    if middle:
        initials += middle[0].upper()
    
    return {
        'first': first,
        'last': last,
        'middle': middle,
        'initials': initials,
        'full': name
    }


def format_author_list(
    authors: List[str],
    style: str = 'apa',
    max_authors: int = 20
) -> str:
    """
    Format list of authors according to citation style.
    
    Args:
        authors: List of author names
        style: Citation style ('apa', 'mla', 'chicago', 'jmlr')
        max_authors: Maximum authors before using "et al."
        
    Returns:
        Formatted author string
    """
    if not authors:
        return "Unknown"
    
    if style == 'jmlr':
        if len(authors) <= 3:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} and {authors[1]}"
            else:
                return f"{authors[0]}, {authors[1]}, and {authors[2]}"
        else:
            return f"{authors[0]} et al."
    
    elif style == 'apa':
        if len(authors) <= max_authors:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} & {authors[1]}"
            else:
                return ', '.join(authors[:-1]) + f", & {authors[-1]}"
        else:
            return f"{authors[0]} et al."
    
    else:
        # Default: simple comma-separated
        if len(authors) <= max_authors:
            return ', '.join(authors)
        else:
            return f"{authors[0]} et al."


def calculate_hash(text: str) -> str:
    """
    Calculate SHA-256 hash of text.
    
    Args:
        text: Input text
        
    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string in various formats.
    
    Args:
        date_str: Date string
        
    Returns:
        datetime object or None
    """
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%d %b %Y',
        '%B %d, %Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def format_date(date: datetime, format_str: str = '%Y-%m-%d') -> str:
    """
    Format datetime object as string.
    
    Args:
        date: datetime object
        format_str: Format string
        
    Returns:
        Formatted date string
    """
    return date.strftime(format_str)


def extract_year(date_str: str) -> Optional[str]:
    """
    Extract year from date string.
    
    Args:
        date_str: Date string
        
    Returns:
        Year as string or None
    """
    # Try to parse date
    date = parse_date(date_str)
    if date:
        return str(date.year)
    
    # Try to extract 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return year_match.group(0)
    
    return None


def clean_abstract(abstract: str) -> str:
    """
    Clean and normalize abstract text.
    
    Args:
        abstract: Raw abstract text
        
    Returns:
        Cleaned abstract
    """
    # Remove extra whitespace
    cleaned = ' '.join(abstract.split())
    
    # Remove common prefixes
    prefixes = ['Abstract:', 'ABSTRACT:', 'Summary:', 'SUMMARY:']
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    return cleaned


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract keywords from text (simple frequency-based).
    
    Args:
        text: Input text
        top_n: Number of keywords to return
        
    Returns:
        List of keywords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove common stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'we', 'us', 'our', 'i', 'you', 'your',
        'he', 'she', 'it', 'they', 'them', 'their'
    }
    
    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stopwords and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N
    return [word for word, _ in sorted_words[:top_n]]


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address
        
    Returns:
        True if valid format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def load_yaml_safe(filepath: Path) -> Dict:
    """
    Safely load YAML file with error handling.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Parsed YAML data or empty dict
    """
    try:
        import yaml
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def ensure_directory(dirpath: Path) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        dirpath: Directory path
        
    Returns:
        Path object
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def get_file_extension(filepath: Path) -> str:
    """
    Get file extension without dot.
    
    Args:
        filepath: File path
        
    Returns:
        Extension string (lowercase)
    """
    return filepath.suffix.lstrip('.').lower()


def is_url(text: str) -> bool:
    """
    Check if text is a URL.
    
    Args:
        text: Input text
        
    Returns:
        True if URL, False otherwise
    """
    url_pattern = r'^https?://[^\s]+$'
    return bool(re.match(url_pattern, text))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        name = name[:max_length - len(ext) - 1]
        sanitized = f"{name}.{ext}" if ext else name
    
    return sanitized


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity score between two texts.
    
    Uses Jaccard similarity on word sets.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time in minutes.
    
    Args:
        text: Input text
        words_per_minute: Reading speed
        
    Returns:
        Estimated minutes
    """
    word_count = count_words(text)
    return max(1, round(word_count / words_per_minute))


# Regular expression patterns (compiled for efficiency)
PATTERNS = {
    'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'url': re.compile(r'https?://[^\s]+'),
    'doi': re.compile(r'10\.\d{4,}/[^\s]+'),
    'pmid': re.compile(r'PMID:\s*(\d+)', re.IGNORECASE),
    'arxiv': re.compile(r'arXiv:(\d{4}\.\d{4,5})', re.IGNORECASE),
}


def extract_identifiers(text: str) -> Dict[str, List[str]]:
    """
    Extract various identifiers from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of identifier types and lists of found values
    """
    identifiers = {}
    
    for id_type, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            identifiers[id_type] = matches
    
    return identifiers


if __name__ == "__main__":
    # Simple tests
    print("Testing utilities...")
    
    # Test author parsing
    print("\nAuthor parsing:")
    print(parse_author_name("Smith J"))
    print(parse_author_name("Smith, John"))
    print(parse_author_name("John Smith"))
    
    # Test author formatting
    print("\nAuthor formatting:")
    authors = ["Smith J", "Doe A", "Johnson B"]
    print(format_author_list(authors, style='jmlr'))
    
    # Test date parsing
    print("\nDate parsing:")
    print(parse_date("2024-03-15"))
    print(extract_year("2024-03-15"))
    
    # Test similarity
    print("\nText similarity:")
    text1 = "machine learning in healthcare"
    text2 = "healthcare machine learning applications"
    print(f"Similarity: {calculate_similarity(text1, text2):.2f}")
    
    print("\nAll tests completed!")
