#!/bin/bash
# Complete Literature Update Workflow
# Runs search, assessment, and chapter updates

set -e  # Exit on error

echo "==================================="
echo "Literature Update Workflow Starting"
echo "==================================="
echo ""

# Configuration
DATE_RANGE=${DATE_RANGE:-7}
DRY_RUN=${DRY_RUN:-false}
SKIP_SEARCH=${SKIP_SEARCH:-false}
SKIP_ASSESS=${SKIP_ASSESS:-false}

# Check environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable not set"
    exit 1
fi

# Create data directory if needed
mkdir -p data/literature

# Step 1: Search Literature
if [ "$SKIP_SEARCH" = false ]; then
    echo "Step 1/3: Searching literature..."
    echo "Looking back $DATE_RANGE days"
    python scripts/search_literature.py \
        --date-range "$DATE_RANGE" \
        --output-dir data/literature
    echo "✓ Search complete"
    echo ""
else
    echo "Step 1/3: Skipping literature search (using existing data)"
    echo ""
fi

# Check if papers were found
if [ ! -f "data/literature/papers.json" ]; then
    echo "Error: No papers found. Exiting."
    exit 1
fi

# Step 2: Assess Relevance
if [ "$SKIP_ASSESS" = false ]; then
    echo "Step 2/3: Assessing paper relevance..."
    python scripts/assess_relevance.py \
        --input-dir data/literature \
        --chapters-dir chapters \
        --output-file data/relevant_papers.json \
        --threshold 0.5
    echo "✓ Assessment complete"
    echo ""
else
    echo "Step 2/3: Skipping relevance assessment (using existing data)"
    echo ""
fi

# Check if assessments were created
if [ ! -f "data/relevant_papers.json" ]; then
    echo "Error: No relevance assessments found. Exiting."
    exit 1
fi

# Step 3: Update Chapters
echo "Step 3/3: Updating chapters..."
if [ "$DRY_RUN" = true ]; then
    echo "(DRY RUN MODE - no files will be modified)"
fi

python scripts/update_chapters.py \
    --papers-file data/relevant_papers.json \
    --chapters-dir chapters \
    --dry-run "$DRY_RUN" \
    --summary-file data/update_summary.json

echo "✓ Updates complete"
echo ""

# Display summary
echo "==================================="
echo "Workflow Summary"
echo "==================================="

if [ -f "data/update_summary.json" ]; then
    python -c "
import json
with open('data/update_summary.json', 'r') as f:
    summary = json.load(f)
print(f\"Chapters updated: {summary['chapters_updated']}\")
print(f\"Papers added: {summary['total_papers_added']}\")
print(f\"Dry run: {summary['dry_run']}\")
print()
print('Updates by chapter:')
for chapter, count in sorted(summary['updates_by_chapter'].items()):
    print(f\"  {chapter}: {count} papers\")
"
else
    echo "No summary file found"
fi

echo ""
echo "==================================="
echo "Workflow Complete!"
echo "==================================="

# Exit with success
exit 0
