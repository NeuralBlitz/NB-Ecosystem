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
    <div className="flex flex-col h-full bg-background/50 backdrop-blur-xl border-r border-border/40">
      <div className="p-4 border-b border-border/40">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shadow-lg shadow-primary/25">
            <BookMarked className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-display font-bold text-xl tracking-tight">Wiki Docs</h1>
        </div>
        
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search files..." 
            className="pl-9 bg-muted/30 border-border/50 focus:bg-background transition-all"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <ScrollArea className="flex-1 py-4">
        {isLoading ? (
          <div className="p-4 space-y-3">
             <div className="h-4 w-3/4 bg-muted animate-pulse rounded" />
             <div className="h-4 w-1/2 bg-muted animate-pulse rounded" />
             <div className="h-4 w-5/6 bg-muted animate-pulse rounded" />
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground text-sm">
            No files found matching "{search}"
          </div>
        ) : (
          <div className="space-y-6">
            {/* Favorites Section */}
            {favorites && favorites.length > 0 && (
              <div className="mb-6">
                <h3 className="px-4 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Favorites
                </h3>
                <div className="space-y-1">
                  {favorites.map((fav) => (
                    <button
                      key={fav.filePath}
                      onClick={() => onSelect(fav.filePath)}
                      className="w-full text-left px-4 py-1.5 text-sm hover:bg-muted/50 transition-colors flex items-center gap-2 group"
                    >
                      <BookMarked className="w-3.5 h-3.5 text-accent opacity-70 group-hover:opacity-100" />
                      <span className="truncate">{fav.title || fav.filePath.split('/').pop()}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            
            {/* Files Section */}
            <div>
              <h3 className="px-4 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Explorer
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
      
      <div className="p-4 border-t border-border/40 text-xs text-muted-foreground text-center">
        Powered by Markdown Wiki
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
