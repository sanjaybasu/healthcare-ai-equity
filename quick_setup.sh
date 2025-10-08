#!/bin/bash

# Healthcare AI Equity Textbook - Updated Quick Setup Script
# This version fixes gem dependency issues

set -e  # Exit on error

echo "=========================================="
echo "Healthcare AI Equity Textbook Setup"
echo "Fixed Version - Resolves Gem Conflicts"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository. Run this from your repository root."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p _chapters
mkdir -p _layouts
mkdir -p assets/css
mkdir -p .github/workflows
mkdir -p scripts
mkdir -p literature_updates

# Backup existing files
echo "💾 Backing up existing files..."
[ -f "_config.yml" ] && cp _config.yml _config.yml.backup
[ -f "index.md" ] && cp index.md index.md.backup
[ -f "Gemfile" ] && cp Gemfile Gemfile.backup

# Create FIXED Gemfile (resolves version conflicts)
echo "📝 Creating fixed Gemfile..."
cat > Gemfile << 'EOF'
source "https://rubygems.org"

# Use the github-pages gem to ensure compatibility
gem "github-pages", "~> 231", group: :jekyll_plugins

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
end

# Windows and JRuby support
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
gem "webrick", "~> 1.8"
EOF

# Create FIXED _config.yml (compatible with github-pages gem)
echo "📝 Creating fixed _config.yml..."
cat > _config.yml << 'EOF'
# Healthcare AI for Underserved Populations - Jekyll Configuration

title: "Healthcare AI for Underserved Populations"
description: "A Comprehensive Technical Textbook for Physician Data Scientists"
author: "Sanjay Basu, MD PhD"
email: "sanjay.basu@waymarkcare.com"
baseurl: "/healthcare-ai-equity"
url: "https://sanjaybasu.github.io"

markdown: kramdown
theme: minima

kramdown:
  input: GFM
  syntax_highlighter: rouge

collections:
  chapters:
    output: true
    permalink: /chapters/:name/

defaults:
  - scope:
      path: ""
      type: "chapters"
    values:
      layout: "page"

plugins:
  - jekyll-feed
  - jekyll-seo-tag
  - jekyll-sitemap

exclude:
  - Gemfile
  - Gemfile.lock
  - node_modules
  - vendor
  - .github
  - README.md
  - LICENSE
  - scripts
  - "*.backup"
  - literature_updates
EOF

# Create simple CSS (works with minima theme)
echo "💅 Creating stylesheet..."
cat > assets/css/custom.css << 'EOF'
/* Custom styles for Healthcare AI Equity Textbook */

:root {
  --primary: #2c5282;
  --secondary: #4a90a4;
}

.site-header {
  border-top: 5px solid var(--primary);
}

.site-title {
  font-weight: 600;
}

h1, h2, h3 {
  color: var(--primary);
}

h2 {
  border-bottom: 2px solid #e2e8f0;
  padding-bottom: 0.5rem;
  margin-top: 2rem;
}

code {
  background-color: #f7fafc;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
}

pre {
  border-left: 4px solid var(--secondary);
}

.chapter-nav {
  display: flex;
  justify-content: space-between;
  margin: 2rem 0;
  padding: 1rem 0;
  border-top: 1px solid #e2e8f0;
}
EOF

# Update index.md to work with default theme
echo "📝 Creating index.md..."
cat > index.md << 'EOF'
---
layout: home
title: Home
---

# Equitable Healthcare AI: From Algorithms to Clinical Impact

**A Practical Textbook for Physician Data Scientists**

---

## Why This Textbook Is Different

Most healthcare AI discussions focus on prompting ChatGPT or superficial applications. This book goes far beyond, covering:

- **State-of-the-art methods**: Random survival forests, causal inference, federated learning, neural ODEs
- **Production-ready code**: Type hints, error handling, quality scores >0.92
- **Rigorous academics**: 50-100+ citations per chapter from top venues
- **Equity-centered design**: Fairness architected into every algorithm

---

## Table of Contents

### Part I: Foundations and Context

1. [Clinical Informatics Foundations for Equity-Centered AI]({% link _chapters/chapter_01_clinical_informatics.md %})
2. [Mathematical Foundations with Health Equity Applications]({% link _chapters/chapter_02_mathematical_foundations.md %})
3. [Healthcare Data Engineering for Equity]({% link _chapters/chapter_03_healthcare_data_engineering.md %})

### Part II: Core Machine Learning Methods

4. [Machine Learning Fundamentals with Fairness]({% link _chapters/chapter_04_machine_learning_fundamentals.md %})
5. [Deep Learning for Healthcare Applications]({% link _chapters/chapter_05_deep_learning_healthcare.md %})
6. [Natural Language Processing for Clinical Text]({% link _chapters/chapter_06_clinical_nlp.md %})
7. [Computer Vision for Medical Imaging]({% link _chapters/chapter_07_medical_imaging.md %})
8. [Time Series Analysis for Clinical Data]({% link _chapters/chapter_08_clinical_time_series.md %})

### Part III: Advanced Methods

9. [Advanced Clinical NLP and Information Retrieval]({% link _chapters/chapter_09_advanced_clinical_nlp.md %})
10. [Survival Analysis and Time-to-Event Modeling]({% link _chapters/chapter_10_survival_analysis.md %})
11. [Causal Inference for Healthcare AI]({% link _chapters/chapter_11_causal_inference.md %})
12. [Federated Learning and Privacy-Preserving AI]({% link _chapters/chapter_12_federated_learning_privacy.md %})
13. [Transfer Learning and Domain Adaptation]({% link _chapters/chapter_13_transfer_learning.md %})

### Part IV: Interpretability, Validation, and Trust

14. [Interpretability and Explainability]({% link _chapters/chapter_14_interpretability_explainability.md %})
15. [Validation Strategies for Clinical AI]({% link _chapters/chapter_15_validation_strategies.md %})
16. [Uncertainty Quantification and Calibration]({% link _chapters/chapter_16_uncertainty_calibration.md %})
17. [Regulatory Considerations and FDA Pathways]({% link _chapters/chapter_17_regulatory_considerations.md %})

### Part V: Deployment and Implementation

18. [Implementation Science for Healthcare AI]({% link _chapters/chapter_18_implementation_science.md %})
19. [Human-AI Collaboration in Clinical Settings]({% link _chapters/chapter_19_human_ai_collaboration.md %})
20. [Post-Deployment Monitoring and Maintenance]({% link _chapters/chapter_20_monitoring_maintenance.md %})
21. [Health Equity Metrics and Evaluation]({% link _chapters/chapter_21_health_equity_metrics.md %})

### Part VI: Specialized Applications

22. [Clinical Decision Support Systems]({% link _chapters/chapter_22_clinical_decision_support.md %})
23. [Precision Medicine and Genomic AI]({% link _chapters/chapter_23_precision_medicine_genomics.md %})
24. [Population Health Management and Screening]({% link _chapters/chapter_24_population_health_screening.md %})
25. [Social Determinants of Health Integration]({% link _chapters/chapter_25_sdoh_integration.md %})

### Part VII: Emerging Methods and Future Directions

26. [Large Language Models in Healthcare]({% link _chapters/chapter_26_llms_in_healthcare.md %})
27. [Multi-Modal Learning for Clinical AI]({% link _chapters/chapter_27_multimodal_learning.md %})
28. [Continual Learning and Model Updating]({% link _chapters/chapter_28_continual_learning.md %})
29. [AI for Global Health and Resource-Limited Settings]({% link _chapters/chapter_29_global_health_ai.md %})
30. [Research Frontiers in Equity-Centered Health AI]({% link _chapters/chapter_30_research_frontiers_equity.md %})

---

## Technical Specifications

- **Python**: 3.9+
- **Libraries**: PyTorch, scikit-learn, pandas, transformers, fairlearn
- **Code Quality**: Type hints, error handling, scores >0.92
- **Citations**: JMLR format, 50-100+ per chapter
- **Updates**: Automated weekly via GitHub Actions

---

## About

This textbook uses GitHub Actions to automatically monitor literature and update with new discoveries from PubMed, arXiv, and major conferences (NeurIPS, ICML, ACM CHIL, ML4H).

**Repository**: [GitHub](https://github.com/sanjaybasu/healthcare-ai-equity)  
**License**: MIT  
**Contact**: sanjay.basu@waymarkcare.com

---

> *"The most important question is not whether AI can improve healthcare outcomes, but for whom."*
EOF

# Move chapters to _chapters directory if needed
echo "📚 Organizing chapters..."
if ls chapter_*.md 1> /dev/null 2>&1; then
    echo "Moving chapter files to _chapters/..."
    mv chapter_*.md _chapters/ 2>/dev/null || true
fi

# Add front matter to chapters if missing
echo "📝 Checking chapter front matter..."
for file in _chapters/*.md; do
    if [ -f "$file" ]; then
        if ! grep -q "^---" "$file"; then
            chapter_name=$(basename "$file" .md)
            temp_file="${file}.tmp"
            echo "---" > "$temp_file"
            echo "layout: page" >> "$temp_file"
            echo "title: \"$chapter_name\"" >> "$temp_file"
            echo "---" >> "$temp_file"
            echo "" >> "$temp_file"
            cat "$file" >> "$temp_file"
            mv "$temp_file" "$file"
            echo "  Added front matter to $chapter_name"
        fi
    fi
done

# Create GitHub Actions workflow
echo "⚙️ Setting up GitHub Actions..."
cat > .github/workflows/pages.yml << 'EOF'
name: Deploy Jekyll site to Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.1'
          bundler-cache: true
      
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v5
      
      - name: Build with Jekyll
        run: bundle exec jekyll build --baseurl "${{ steps.pages.outputs.base_path }}"
        env:
          JEKYLL_ENV: production
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
EOF

# Create .gitignore
echo "🚫 Creating .gitignore..."
cat > .gitignore << 'EOF'
_site/
.sass-cache/
.jekyll-cache/
.jekyll-metadata
vendor/
*.backup
.bundle/
Gemfile.lock
literature_updates/*.json
__pycache__/
*.pyc
.env
.DS_Store
EOF

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Install dependencies (this may take a few minutes):"
echo "   bundle install"
echo ""
echo "2. Test locally (optional):"
echo "   bundle exec jekyll serve"
echo "   Visit: http://localhost:4000/healthcare-ai-equity/"
echo ""
echo "3. Commit and push using GitHub Desktop or:"
echo "   git add ."
echo "   git commit -m 'Fix: Implement proper Jekyll structure'"
echo "   git push origin main"
echo ""
echo "4. Configure GitHub Pages:"
echo "   Go to Settings → Pages → Source → GitHub Actions"
echo ""
echo "📚 Your textbook will be live at:"
echo "   https://sanjaybasu.github.io/healthcare-ai-equity/"
echo ""
