#!/bin/bash

# Setup script for Healthcare AI Textbook
# This script helps configure the repository for local development or deployment

set -e

echo "=========================================="
echo "Healthcare AI Textbook - Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS or Linux
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: ${MACHINE}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
echo ""

# Check Ruby
if command_exists ruby; then
    RUBY_VERSION=$(ruby -v | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Ruby ${RUBY_VERSION} installed"
else
    echo -e "${RED}✗${NC} Ruby not found"
    echo "  Please install Ruby 3.1+ from https://www.ruby-lang.org/"
    exit 1
fi

# Check Bundler
if command_exists bundle; then
    echo -e "${GREEN}✓${NC} Bundler installed"
else
    echo -e "${YELLOW}!${NC} Bundler not found. Installing..."
    gem install bundler
fi

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} installed"
else
    echo -e "${RED}✗${NC} Python 3 not found"
    echo "  Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Check pip
if command_exists pip3; then
    echo -e "${GREEN}✓${NC} pip installed"
else
    echo -e "${RED}✗${NC} pip not found"
    echo "  Please install pip"
    exit 1
fi

echo ""
echo "All prerequisites met!"
echo ""

# Install Ruby dependencies
echo "Installing Ruby dependencies..."
bundle install
echo -e "${GREEN}✓${NC} Ruby dependencies installed"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt
echo -e "${GREEN}✓${NC} Python dependencies installed"
echo ""

# Configure git hooks (optional)
read -p "Install git commit hooks for code quality? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p .git/hooks
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for code quality

echo "Running pre-commit checks..."

# Check Python files
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
if [ -n "$python_files" ]; then
    echo "Checking Python code style..."
    black --check $python_files || {
        echo "Python code style issues found. Run 'black .' to fix."
        exit 1
    }
fi

# Check for large files
max_size=5242880  # 5MB
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ $size -gt $max_size ]; then
            echo "Error: $file is larger than 5MB"
            exit 1
        fi
    fi
done

echo "Pre-commit checks passed!"
EOF
    chmod +x .git/hooks/pre-commit
    echo -e "${GREEN}✓${NC} Git hooks installed"
else
    echo "Skipping git hooks installation"
fi
echo ""

# Check for API keys
echo "Checking environment configuration..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}!${NC} ANTHROPIC_API_KEY not set"
    echo "  You'll need this for automated literature updates"
    read -p "  Enter your Anthropic API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        echo "export ANTHROPIC_API_KEY='$api_key'" >> ~/.bashrc
        export ANTHROPIC_API_KEY="$api_key"
        echo -e "${GREEN}✓${NC} API key saved to ~/.bashrc"
    fi
else
    echo -e "${GREEN}✓${NC} ANTHROPIC_API_KEY is set"
fi

if [ -z "$PUBMED_API_KEY" ]; then
    echo -e "${YELLOW}!${NC} PUBMED_API_KEY not set"
    echo "  You'll need this for PubMed literature searches"
    read -p "  Enter your PubMed API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        echo "export PUBMED_API_KEY='$api_key'" >> ~/.bashrc
        export PUBMED_API_KEY="$api_key"
        echo -e "${GREEN}✓${NC} API key saved to ~/.bashrc"
    fi
else
    echo -e "${GREEN}✓${NC} PUBMED_API_KEY is set"
fi
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p scripts
mkdir -p .github/workflows
mkdir -p _layouts
mkdir -p assets/css
mkdir -p assets/images
echo -e "${GREEN}✓${NC} Directories created"
echo ""

# Test Jekyll build
echo "Testing Jekyll build..."
if bundle exec jekyll build --config _config.yml 2>&1 | tail -n 1 | grep -q "done"; then
    echo -e "${GREEN}✓${NC} Jekyll builds successfully"
else
    echo -e "${YELLOW}!${NC} Jekyll build had warnings (check output above)"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Local development:"
echo "   bundle exec jekyll serve"
echo "   Then visit: http://localhost:4000"
echo ""
echo "2. Configure GitHub Pages:"
echo "   - Go to Settings → Pages"
echo "   - Source: GitHub Actions"
echo "   - Add secrets: ANTHROPIC_API_KEY, PUBMED_API_KEY"
echo ""
echo "3. Test literature update:"
echo "   python3 scripts/update_literature.py"
echo ""
echo "4. Read documentation:"
echo "   - README.md - Overview and setup"
echo "   - CONTRIBUTING.md - How to contribute"
echo ""
echo "Questions? Open an issue on GitHub!"
echo ""
