## Packages
react-markdown | Core markdown rendering
remark-math | Markdown plugin for math syntax
rehype-katex | HTML processor for KaTeX math rendering
remark-gfm | GitHub Flavored Markdown support (tables, etc.)
rehype-highlight | Syntax highlighting for code blocks
framer-motion | Smooth animations for sidebar and page transitions
katex | CSS styles for math equations

## Notes
Tailwind Config - extend fontFamily:
fontFamily: {
  sans: ["var(--font-sans)"],
  serif: ["var(--font-serif)"],
  mono: ["var(--font-mono)"],
}
API expects ?path=/path/to/file query param for fetching content
