#!/usr/bin/env python3
"""
Documentation Consolidation Script for NeuralBlitz Ecosystem

This script consolidates 150+ markdown documentation files into a unified
documentation site with proper structure, frontmatter, and redirects.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
WORKSPACE_ROOT = Path("/home/runner/workspace")
DOCS_SITE_ROOT = WORKSPACE_ROOT / "docs-site"
BACKUP_DIR = WORKSPACE_ROOT / "docs-backup"

# Document categories and their target directories
CATEGORIES = {
    "getting-started": {
        "patterns": [r"README\.md$", r"GETTING_STARTED", r"QUICKSTART", r"CONTRIBUTING", r"CODE_OF_CONDUCT"],
        "title": "Getting Started",
        "description": "Installation, setup, and contribution guidelines"
    },
    "architecture": {
        "patterns": [r"ARCHITECTURE", r"DATA_FLOW", r"SYSTEM_BLUEPRINT", r"API_CONTRACT_MATRIX"],
        "title": "Architecture",
        "description": "System architecture, data flow, and technical design"
    },
    "security": {
        "patterns": [r"SECURITY", r"VULNERABILITY", r"AUDIT", r"REMEDIATION", r"AUTH_", r"COMPLIANCE"],
        "title": "Security",
        "description": "Security reports, audits, and compliance documentation"
    },
    "api-reference": {
        "patterns": [r"API_REFERENCE", r"API_CONTRACT", r"ENDPOINTS", r"INTERFACE"],
        "title": "API Reference",
        "description": "API documentation and endpoint specifications"
    },
    "research": {
        "patterns": [r"RD_", r"TASK_", r"EXECUTIVE_SUMMARY", r"BENCHMARK", r"VALIDATION", r"ANALYSIS", r"REPORT"],
        "title": "Research & Development",
        "description": "R&D reports, benchmarks, and analysis"
    },
    "operations": {
        "patterns": [r"DEVELOPMENT_WORKFLOW", r"BACKUP_RECOVERY", r"ASSET_INVENTORY", r"DEPLOYMENT", r"MIGRATION"],
        "title": "Operations",
        "description": "Operational guides, workflows, and deployment"
    },
    "projects": {
        "patterns": [],
        "title": "Project Documentation",
        "description": "Individual project documentation"
    }
}

def categorize_file(filepath):
    """Categorize a markdown file based on its name and content."""
    filename = filepath.name.upper()
    
    for category, config in CATEGORIES.items():
        for pattern in config["patterns"]:
            if re.search(pattern, filename):
                return category
    
    # Default categorization based on location
    path_str = str(filepath)
    if "nb-omnibus-router" in path_str:
        return "api-reference" if "api" in filename.lower() else "projects"
    elif "lrs-agents" in path_str:
        return "projects"
    elif "NBX-LRS" in path_str:
        return "projects"
    elif "neuralblitz" in path_str:
        return "projects"
    elif "docs" in path_str and "docs-site" not in path_str:
        return "operations"
    else:
        return "research"  # Default for root-level reports

def extract_title(content):
    """Extract the title from markdown content."""
    # Look for H1 header
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # Look for title in frontmatter
    match = re.search(r'^title:\s*["\']?(.+?)["\']?$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    return None

def extract_description(content):
    """Extract description from markdown content."""
    # Look for description in frontmatter
    match = re.search(r'^description:\s*["\']?(.+?)["\']?$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # Look for first paragraph after H1
    match = re.search(r'^#\s+.+\n+(.+?)(?=\n#{1,2}|\Z)', content, re.MULTILINE | re.DOTALL)
    if match:
        desc = match.group(1).strip()
        # Limit to 150 characters
        if len(desc) > 150:
            desc = desc[:147] + "..."
        return desc
    
    return ""

def generate_frontmatter(title, description, category, original_path):
    """Generate YAML frontmatter for a document."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    # Generate tags based on category and content
    tags = [category.replace("-", " ")]
    if "security" in category:
        tags.extend(["security", "audit", "compliance"])
    elif "api" in category:
        tags.extend(["api", "reference", "endpoints"])
    elif "architecture" in category:
        tags.extend(["architecture", "design", "system"])
    elif "research" in category:
        tags.extend(["research", "analysis", "report"])
    
    frontmatter = f"""---
title: "{title}"
description: "{description}"
category: {category}
tags: {tags}
date: {timestamp}
original_path: "{original_path}"
---

"""
    return frontmatter

def update_internal_links(content, old_path, file_mapping):
    """Update internal links to point to new locations."""
    # Pattern to match markdown links
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_link(match):
        text = match.group(1)
        url = match.group(2)
        
        # Skip external links
        if url.startswith(('http://', 'https://', 'mailto:', '#')):
            return match.group(0)
        
        # Skip image links
        if url.startswith(('data:', 'javascript:')):
            return match.group(0)
        
        # Check if this is a relative link to another markdown file
        if url.endswith('.md') or '.md#' in url:
            # Try to find the new path in our mapping
            for old, new in file_mapping.items():
                if old in url or url in old:
                    # Create relative path from old_path to new_path
                    return f'[{text}]({new})'
        
        return match.group(0)
    
    return re.sub(link_pattern, replace_link, content)

def fix_broken_images(content, doc_dir):
    """Fix broken image references."""
    # Pattern to match image links
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt = match.group(1)
        src = match.group(2)
        
        # Skip external images
        if src.startswith(('http://', 'https://', 'data:')):
            return match.group(0)
        
        # Check if image exists
        img_path = doc_dir / src
        if not img_path.exists():
            # Image doesn't exist, add placeholder or note
            return f"![{alt}]({{{{site.baseurl}}}}/assets/images/placeholder.png)"
        
        return match.group(0)
    
    return re.sub(img_pattern, replace_image, content)

def create_redirect_page(old_path, new_path):
    """Create a redirect page for old paths."""
    redirect_content = f"""---
layout: redirect
title: Redirect
redirect_to: "{new_path}"
original_path: "{old_path}"
---

# This page has moved

This document has been moved to [the new location]({new_path}).

Please update your bookmarks.
"""
    return redirect_content

def main():
    """Main consolidation function."""
    print("=" * 70)
    print("NeuralBlitz Documentation Consolidation Tool")
    print("=" * 70)
    print()
    
    # Find all markdown files
    print("Step 1: Finding all markdown files...")
    md_files = []
    exclude_patterns = [
        '.cache', 'node_modules', 'gopath', '.local', 
        '.pythonlibs', 'Advanced-Research', 'Emergent-Prompt-Architecture',
        'ComputationalAxioms', 'ncx/.opencode'
    ]
    
    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(excl in str(Path(root) / d) for excl in exclude_patterns)]
        
        for file in files:
            if file.endswith('.md'):
                filepath = Path(root) / file
                md_files.append(filepath)
    
    print(f"Found {len(md_files)} markdown files")
    print()
    
    # Categorize files
    print("Step 2: Categorizing files...")
    categorized = defaultdict(list)
    for filepath in md_files:
        category = categorize_file(filepath)
        categorized[category].append(filepath)
    
    for category, files in sorted(categorized.items()):
        print(f"  {category}: {len(files)} files")
    print()
    
    # Create new docs site structure
    print("Step 3: Creating new documentation structure...")
    
    # Create directories
    for category in CATEGORIES.keys():
        (DOCS_SITE_ROOT / category).mkdir(parents=True, exist_ok=True)
    
    # Create special directories
    (DOCS_SITE_ROOT / "assets" / "images").mkdir(parents=True, exist_ok=True)
    (DOCS_SITE_ROOT / "_data").mkdir(parents=True, exist_ok=True)
    (DOCS_SITE_ROOT / "_includes").mkdir(parents=True, exist_ok=True)
    (DOCS_SITE_ROOT / "_layouts").mkdir(parents=True, exist_ok=True)
    
    print(f"Created structure at: {DOCS_SITE_ROOT}")
    print()
    
    # Create file mapping
    print("Step 4: Creating file mapping...")
    file_mapping = {}
    
    for category, files in categorized.items():
        for filepath in files:
            # Generate new filename
            rel_path = filepath.relative_to(WORKSPACE_ROOT)
            new_filename = str(rel_path).replace('/', '_').replace('\\', '_')
            new_path = f"/{category}/{new_filename}"
            file_mapping[str(filepath)] = new_path
    
    # Save mapping
    mapping_file = DOCS_SITE_ROOT / "_data" / "file_mapping.json"
    import json
    with open(mapping_file, 'w') as f:
        json.dump(file_mapping, f, indent=2)
    
    print(f"Created file mapping with {len(file_mapping)} entries")
    print()
    
    # Create index pages for each category
    print("Step 5: Creating category index pages...")
    for category, config in CATEGORIES.items():
        index_content = f"""---
title: "{config['title']}"
description: "{config['description']}"
category: {category}
layout: category
tags: [{category}, documentation]
---

# {config['title']}

{config['description']}

## Documents in this Category

{{% for doc in site.pages %}}
{{% if doc.category == "{category}" and doc.title != "{config['title']}" %}}
- [{{{{ doc.title }}}}]({{{{ doc.url | relative_url }}}})
{{% endif %}}
{{% endfor %}}
"""
        index_path = DOCS_SITE_ROOT / category / "index.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
        print(f"  Created: {index_path}")
    
    print()
    
    # Process and migrate files
    print("Step 6: Migrating content...")
    migrated_count = 0
    skipped_count = 0
    
    for old_path, new_url in file_mapping.items():
        old_filepath = Path(old_path)
        category = new_url.split('/')[1]
        new_filename = new_url.split('/')[-1]
        new_filepath = DOCS_SITE_ROOT / category / new_filename
        
        try:
            # Read original content
            with open(old_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if already has frontmatter
            if content.startswith('---'):
                # Already has frontmatter, just update it
                # Extract existing frontmatter
                end_frontmatter = content.find('---', 3)
                if end_frontmatter != -1:
                    existing_frontmatter = content[3:end_frontmatter]
                    body = content[end_frontmatter + 3:]
                    
                    # Parse and update
                    title = extract_title(existing_frontmatter) or extract_title(body) or new_filename
                    description = extract_description(existing_frontmatter) or extract_description(body)
                    
                    new_frontmatter = generate_frontmatter(title, description, category, str(old_filepath))
                    new_content = new_frontmatter + body
                else:
                    # Malformed frontmatter, treat as regular content
                    title = extract_title(content) or new_filename
                    description = extract_description(content)
                    new_frontmatter = generate_frontmatter(title, description, category, str(old_filepath))
                    new_content = new_frontmatter + content
            else:
                # No frontmatter, add it
                title = extract_title(content) or new_filename
                description = extract_description(content)
                new_frontmatter = generate_frontmatter(title, description, category, str(old_filepath))
                new_content = new_frontmatter + content
            
            # Update internal links
            new_content = update_internal_links(new_content, old_filepath, file_mapping)
            
            # Fix broken images
            new_content = fix_broken_images(new_content, old_filepath.parent)
            
            # Write to new location
            with open(new_filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            migrated_count += 1
            
        except Exception as e:
            print(f"  Error migrating {old_filepath}: {e}")
            skipped_count += 1
    
    print(f"  Migrated: {migrated_count} files")
    print(f"  Skipped: {skipped_count} files")
    print()
    
    # Create main index
    print("Step 7: Creating main index...")
    main_index = f"""---
title: "NeuralBlitz Documentation"
description: "Complete documentation for the NeuralBlitz AI Ecosystem"
layout: home
---

# NeuralBlitz Documentation

Welcome to the NeuralBlitz documentation site. This is the central hub for all documentation related to the NeuralBlitz AI Ecosystem v50.0.

## Documentation Categories

| Category | Description | Documents |
|----------|-------------|-----------|
"""
    
    for category, config in CATEGORIES.items():
        count = len(categorized[category])
        main_index += f"| [{config['title']}]({{{{site.baseurl}}}}/{category}/) | {config['description']} | {count} docs |\n"
    
    main_index += """
## Quick Links

- [Getting Started]({{site.baseurl}}/getting-started/)
- [API Reference]({{site.baseurl}}/api-reference/)
- [Architecture]({{site.baseurl}}/architecture/)
- [Security]({{site.baseurl}}/security/)

## About NeuralBlitz

NeuralBlitz is a quantum-classical hybrid AI ecosystem featuring:
- Quantum Spiking Neurons
- Multi-Reality Networks
- Consciousness Integration
- Cross-Reality Entanglement
- 11-Dimensional Computing
- Neuro-Symbiotic Integration
- Autonomous Self-Evolution
- Advanced Agent Framework

---

*Last updated: {timestamp}*
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(DOCS_SITE_ROOT / "index.md", 'w') as f:
        f.write(main_index)
    
    print(f"Created: {DOCS_SITE_ROOT / 'index.md'}")
    print()
    
    # Create configuration file
    print("Step 8: Creating site configuration...")
    config_content = f"""# NeuralBlitz Documentation Site Configuration

title: "NeuralBlitz Documentation"
description: "Complete documentation for the NeuralBlitz AI Ecosystem"
baseurl: ""
url: "https://docs.neuralblitz.ai"

# Build settings
markdown: kramdown
highlighter: rouge
permalink: pretty

# Collections
collections:
  docs:
    output: true
    permalink: /:name/

# Plugins
plugins:
  - jekyll-redirect-from
  - jekyll-sitemap
  - jekyll-seo-tag

# Exclude
exclude:
  - Gemfile
  - Gemfile.lock
  - node_modules
  - vendor
  - .git
  - .github

# Default frontmatter
defaults:
  - scope:
      path: ""
      type: "pages"
    values:
      layout: "default"

# Site metadata
version: "50.0"
last_updated: "{timestamp}"
total_documents: {len(md_files)}
""".format(timestamp=datetime.now().strftime("%Y-%m-%d"), len(md_files))
    
    with open(DOCS_SITE_ROOT / "_config.yml", 'w') as f:
        f.write(config_content)
    
    print(f"Created: {DOCS_SITE_ROOT / '_config.yml'}")
    print()
    
    # Create redirect pages
    print("Step 9: Creating redirect pages...")
    redirects_dir = DOCS_SITE_ROOT / "redirects"
    redirects_dir.mkdir(exist_ok=True)
    
    with open(redirects_dir / "index.md", 'w') as f:
        f.write("""---
title: "Old Documentation Redirects"
layout: page
---

# Documentation Relocation Notice

The NeuralBlitz documentation has been reorganized. 

Please use the navigation menu to find what you're looking for, or visit the [main documentation page](../).

## Common Redirects

""")
    
    print(f"Created: {redirects_dir / 'index.md'}")
    print()
    
    # Create migration report
    print("Step 10: Creating consolidation report...")
    report = f"""# Documentation Consolidation Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Files Processed:** {len(md_files)}
**Successfully Migrated:** {migrated_count}
**Skipped:** {skipped_count}

## Summary

This report documents the consolidation of {len(md_files)} markdown files into the new documentation site structure.

## Categories

"""
    
    for category, files in sorted(categorized.items()):
        report += f"\n### {CATEGORIES[category]['title']} ({len(files)} files)\n\n"
        for filepath in files:
            new_url = file_mapping.get(str(filepath), "NOT MAPPED")
            rel_path = filepath.relative_to(WORKSPACE_ROOT)
            report += f"- `{rel_path}` → `{new_url}`\n"
    
    report += """
## Next Steps

1. Review migrated content for accuracy
2. Update internal links manually where automatic updates failed
3. Add missing images to `assets/images/`
4. Test the documentation site locally
5. Deploy to production

## File Mapping

The complete file mapping is available in `_data/file_mapping.json`.

## Known Issues

- Some relative image paths may need manual fixing
- Cross-references between documents should be verified
- Old URLs should redirect to new locations (not yet implemented)

---

*Report generated by documentation consolidation tool*
"""
    
    with open(DOCS_SITE_ROOT / "CONSOLIDATION_REPORT.md", 'w') as f:
        f.write(report)
    
    print(f"Created: {DOCS_SITE_ROOT / 'CONSOLIDATION_REPORT.md'}")
    print()
    
    print("=" * 70)
    print("Consolidation Complete!")
    print("=" * 70)
    print()
    print(f"New documentation site created at: {DOCS_SITE_ROOT}")
    print(f"Total files processed: {len(md_files)}")
    print(f"Successfully migrated: {migrated_count}")
    print()
    print("Next steps:")
    print("1. Review the CONSOLIDATION_REPORT.md")
    print("2. Check the migrated content in docs-site/")
    print("3. Fix any broken links or images")
    print("4. Test the site locally with Jekyll")
    print()

if __name__ == "__main__":
    main()
