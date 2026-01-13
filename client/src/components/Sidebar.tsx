import { Search, BookMarked, Menu, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileTree } from './FileTree';
import type { FileNode } from '@/hooks/use-files';
import { useState, useMemo } from 'react';
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Button } from './ui/button';

interface SidebarProps {
  files?: FileNode[];
  favorites?: any[]; // Typed loosely for now
  selectedPath?: string;
  onSelect: (path: string) => void;
  isLoading: boolean;
}

export function Sidebar({ files = [], favorites = [], selectedPath, onSelect, isLoading }: SidebarProps) {
  const [search, setSearch] = useState("");

  // Recursive filter function
  const filterNodes = (nodes: FileNode[], term: string): FileNode[] => {
    return nodes.reduce((acc: FileNode[], node) => {
      const matches = node.name.toLowerCase().includes(term.toLowerCase());
      
      if (node.type === 'directory' && node.children) {
        const filteredChildren = filterNodes(node.children, term);
        if (filteredChildren.length > 0 || matches) {
          acc.push({ ...node, children: filteredChildren });
        }
      } else if (matches) {
        acc.push(node);
      }
      return acc;
    }, []);
  };

  const filteredFiles = useMemo(() => {
    if (!search) return files;
    return filterNodes(files, search);
  }, [files, search]);

  const SidebarContent = () => (
    <div className="flex flex-col h-full bg-background border-r border-border/40 selection:bg-primary/10">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-2xl shadow-primary/20 ring-1 ring-white/10">
            <BookMarked className="w-6 h-6 text-background" />
          </div>
          <div className="flex flex-col">
            <h1 className="font-display font-bold text-lg tracking-tight leading-none mb-1">Wiki Docs</h1>
            <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-medium">Knowledge Base</span>
          </div>
        </div>
        
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
          <Input 
            placeholder="Search documents..." 
            className="pl-10 h-11 bg-muted/20 border-border/20 focus:border-primary/30 focus:bg-muted/30 transition-all rounded-xl placeholder:text-muted-foreground/50"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <ScrollArea className="flex-1 px-3">
        {isLoading ? (
          <div className="p-4 space-y-4">
             <div className="h-4 w-3/4 bg-muted/40 animate-pulse rounded-lg" />
             <div className="h-4 w-1/2 bg-muted/40 animate-pulse rounded-lg" />
             <div className="h-4 w-5/6 bg-muted/40 animate-pulse rounded-lg" />
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground/60 text-sm italic">
            No matches for "{search}"
          </div>
        ) : (
          <div className="space-y-8 pb-10">
            {/* Favorites Section */}
            {favorites && favorites.length > 0 && (
              <div>
                <h3 className="px-3 mb-3 text-[10px] font-bold text-muted-foreground/50 uppercase tracking-[0.2em]">
                  Bookmarks
                </h3>
                <div className="space-y-0.5">
                  {favorites.map((fav) => (
                    <button
                      key={fav.filePath}
                      onClick={() => onSelect(fav.filePath)}
                      className={`w-full text-left px-3 py-2.5 text-sm rounded-lg transition-all flex items-center gap-3 group hover:bg-muted/40 ${selectedPath === fav.filePath ? 'bg-primary/5 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
                    >
                      <BookMarked className={`w-4 h-4 transition-colors ${selectedPath === fav.filePath ? 'text-primary' : 'text-muted-foreground group-hover:text-primary/70'}`} />
                      <span className="truncate font-medium">{fav.title || fav.filePath.split('/').pop()}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            
            {/* Files Section */}
            <div>
              <h3 className="px-3 mb-3 text-[10px] font-bold text-muted-foreground/50 uppercase tracking-[0.2em]">
                Documents
              </h3>
              <FileTree 
                nodes={filteredFiles} 
                selectedPath={selectedPath} 
                onSelect={onSelect} 
              />
            </div>
          </div>
        )}
      </ScrollArea>
      
      <div className="p-6 border-t border-border/40">
        <div className="flex items-center justify-center gap-2 text-[10px] text-muted-foreground/40 font-medium uppercase tracking-widest">
           <span>v1.0.0</span>
           <span className="w-1 h-1 rounded-full bg-border" />
           <span>Ready</span>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden md:block w-72 h-screen sticky top-0 shrink-0">
        <SidebarContent />
      </aside>

      {/* Mobile Drawer */}
      <div className="md:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden">
              <Menu className="w-5 h-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-80">
            <SidebarContent />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
