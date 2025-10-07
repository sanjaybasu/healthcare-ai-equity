#!/usr/bin/env python3
"""
Update Timestamp

Updates the last-updated timestamp on the landing page.
"""

import re
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_timestamp(index_file: Path = Path("index.md")) -> bool:
    """
    Update the timestamp in the index file.
    
    Args:
        index_file: Path to index.md
        
    Returns:
        True if successful
    """
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        return False
    
    # Read content
    with open(index_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Update patterns
    patterns = [
        (r'\*\*Last Updated:\*\* \[Automatically generated timestamp\]',
         f'**Last Updated:** {timestamp}'),
        (r'\*This is a living document\. Last automated update: \[timestamp\]\.',
         f'*This is a living document. Last automated update: {timestamp}.*'),
    ]
    
    updated = False
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            updated = True
            logger.info(f"Updated timestamp pattern: {pattern}")
    
    # If no patterns found, add timestamp at end
    if not updated:
        if not content.endswith('\n'):
            content += '\n'
        content += f'\n---\n\n*Last automated update: {timestamp}*\n'
        logger.info("Added timestamp at end of file")
    
    # Write back
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Updated timestamp in {index_file}")
    
    return True


def main():
    """Main execution function."""
    logger.info("Updating timestamp")
    
    success = update_timestamp()
    
    if success:
        logger.info("Timestamp updated successfully")
        return 0
    else:
        logger.error("Failed to update timestamp")
        return 1


if __name__ == "__main__":
    exit(main())
