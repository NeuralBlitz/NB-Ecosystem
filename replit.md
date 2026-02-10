# replit.md

## Overview

This is a wiki-style documentation viewer application that renders markdown files from a file system. Users can browse a hierarchical file tree, view markdown content with full rendering support (math equations, code highlighting, GFM tables), and bookmark favorite pages. The application serves as a knowledge base interface for viewing technical documentation and research notes.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Routing**: Wouter (lightweight alternative to React Router)
- **State Management**: TanStack React Query for server state, local React state for UI
- **Styling**: Tailwind CSS with shadcn/ui component library (New York style variant)
- **Markdown Rendering**: react-markdown with remark-gfm, remark-math, rehype-katex, and rehype-highlight plugins
- **Animations**: Framer Motion for smooth transitions

The frontend is a single-page application with a sidebar navigation pattern. The main WikiPage component displays file content, while the Sidebar component shows the file tree and favorites list.

### Backend Architecture
- **Framework**: Express.js with TypeScript
- **Build Tool**: Vite for development, esbuild for production bundling
- **API Design**: RESTful endpoints defined in shared/routes.ts with Zod validation schemas

The server serves both the API and static files. In development, Vite handles hot module replacement. In production, pre-built static files are served from dist/public.

### Data Storage
- **Database**: PostgreSQL via Drizzle ORM
- **Schema Location**: shared/schema.ts
- **Current Tables**: favorites (stores bookmarked file paths)
- **File Storage**: Markdown files stored in server/data/ directory, read directly from filesystem

The application uses a hybrid storage approach: user preferences (favorites) are stored in PostgreSQL, while the actual content files are served from the filesystem.

### API Structure
All API routes are prefixed with /api and defined in shared/routes.ts:
- `GET /api/files/tree` - Returns hierarchical file tree structure
- `GET /api/files/content?path=<filepath>` - Returns file content with type detection
- `GET /api/favorites` - Lists all bookmarked files
- `POST /api/favorites` - Toggles bookmark status for a file

### Key Design Patterns
- **Shared Types**: Schema and route definitions in shared/ directory are used by both client and server
- **Path Aliases**: @/ maps to client/src/, @shared/ maps to shared/
- **Database Migrations**: Drizzle Kit with `db:push` command for schema synchronization

## External Dependencies

### Database
- PostgreSQL (required, configured via DATABASE_URL environment variable)
- Drizzle ORM for type-safe database queries
- connect-pg-simple for session storage capability

### Frontend Libraries
- KaTeX (via CDN) for math equation rendering
- Highlight.js (via CDN) for code syntax highlighting
- Google Fonts (DM Sans, Fira Code, Geist Mono, Architects Daughter)

### Build Tools
- Vite with React plugin for development server and client bundling
- esbuild for server-side production bundling
- Replit-specific plugins for development (error overlay, cartographer, dev banner)