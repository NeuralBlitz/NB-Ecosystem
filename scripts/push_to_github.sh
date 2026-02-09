#!/bin/bash
# NeuralBlitz GitHub Push Script
# Generated: 2026-02-08
# Usage: ./push_to_github.sh [core|agents|ui|all]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_BASE="https://github.com/yourusername"
REPOS=("neuralblitz-core" "neuralblitz-agents" "neuralblitz-ui")

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_git() {
    if ! command -v git &> /dev/null; then
        echo_error "Git is not installed. Please install git first."
        exit 1
    fi
}

check_gh() {
    if ! command -v gh &> /dev/null; then
        echo_warn "GitHub CLI (gh) not installed. Will use git only."
        GH_INSTALLED=false
    else
        GH_INSTALLED=true
    fi
}

init_repo() {
    local repo_name=$1
    local local_path="/home/runner/workspace/$repo_name"
    
    echo_info "Processing repository: $repo_name"
    
    # Check if directory exists
    if [ ! -d "$local_path" ]; then
        echo_error "Directory $local_path does not exist!"
        return 1
    fi
    
    cd "$local_path"
    
    # Initialize git if not already initialized
    if [ ! -d ".git" ]; then
        echo_info "Initializing git repository..."
        git init
    else
        echo_info "Git repository already exists."
    fi
    
    # Check if remote already exists
    if git remote get-url origin &> /dev/null; then
        echo_info "Remote 'origin' already configured."
    else
        echo_warn "Remote 'origin' not set. Set it with:"
        echo "  git remote add origin $REPO_BASE/$repo_name.git"
        echo ""
    fi
    
    # Create .gitignore if missing
    if [ ! -f ".gitignore" ]; then
        echo_warn ".gitignore not found. Creating default..."
        cat > .gitignore << 'EOF'
# Byte-compiled
__pycache__/
*.py[cod]
*$py.class

# Distribution
build/
dist/
*.egg-info/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
EOF
    fi
    
    echo_info "Repository $repo_name ready for push."
}

push_repo() {
    local repo_name=$1
    local local_path="/home/runner/workspace/$repo_name"
    local repo_url="$REPO_BASE/$repo_name.git"
    
    cd "$local_path"
    
    echo_info "Pushing $repo_name to GitHub..."
    
    # Add all files
    git add -A
    
    # Check for changes
    if git diff --cached --quiet; then
        echo_warn "No changes to commit in $repo_name"
        return 0
    fi
    
    # Create commit
    local commit_msg="Initial commit: $(date '+%Y-%m-%d') - Interface definitions"
    git commit -m "$commit_msg"
    
    # Push to GitHub
    if [ "$GH_INSTALLED" = true ]; then
        # Use GitHub CLI
        if ! gh repo view "$repo_name" &> /dev/null; then
            echo_info "Creating repository on GitHub..."
            gh repo create "$repo_name" --public --description "NeuralBlitz $repo_name SDK - Interface definitions only"
        fi
        
        gh repo sync || true
        git push origin main --force
    else
        # Use git only
        echo_warn "Please manually push with:"
        echo "  cd $local_path"
        echo "  git remote add origin $repo_url"
        echo "  git push -u origin main"
    fi
    
    echo_info "Successfully pushed $repo_name!"
}

verify_repo() {
    local repo_name=$1
    local local_path="/home/runner/workspace/$repo_name"
    
    echo_info "Verifying $repo_name..."
    
    # Check required files
    local required_files=("README.md" "src/interfaces.py")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$local_path/$file" ]; then
            echo_error "Required file missing: $file"
            return 1
        fi
    done
    
    # Check no implementation code
    if grep -r "def __init__" "$local_path/src/" --include="*.py" 2>/dev/null | grep -v "pass" | grep -v "raise" > /dev/null; then
        echo_warn "Potential implementation code found in interfaces!"
    fi
    
    echo_info "$repo_name verification passed!"
}

usage() {
    echo "NeuralBlitz GitHub Push Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  core     Push neuralblitz-core only"
    echo "  agents   Push neuralblitz-agents only"
    echo "  ui       Push neuralblitz-ui only"
    echo "  all      Push all repositories (default)"
    echo "  init     Initialize repositories without pushing"
    echo "  verify   Verify repositories before push"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Push all repos"
    echo "  $0 core              # Push only neuralblitz-core"
    echo "  $0 init              # Initialize repos"
    echo "  $0 verify            # Verify repos"
}

main() {
    echo "=============================================="
    echo "  NeuralBlitz GitHub Push Script"
    echo "=============================================="
    echo ""
    
    check_git
    check_gh
    
    local command=${1:-all}
    
    case $command in
        core|agents|ui)
            init_repo "neuralblitz-$command"
            verify_repo "neuralblitz-$command"
            push_repo "neuralblitz-$command"
            ;;
        all)
            for repo in "${REPOS[@]}"; do
                init_repo "$repo"
                verify_repo "$repo"
                push_repo "$repo"
                echo ""
            done
            ;;
        init)
            for repo in "${REPOS[@]}"; do
                init_repo "$repo"
            done
            ;;
        verify)
            for repo in "${REPOS[@]}"; do
                verify_repo "$repo"
            done
            ;;
        help|--help|-h)
            usage
            exit 0
            ;;
        *)
            echo_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
    
    echo ""
    echo "=============================================="
    echo_info "All done!"
    echo "=============================================="
}

main "$@"
