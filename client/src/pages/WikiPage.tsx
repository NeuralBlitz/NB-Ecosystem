import { useFileContent, useFileTree, useFavorites, useToggleFavorite } from "@/hooks/use-files";
import { Sidebar } from "@/components/Sidebar";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { useLocation, useSearch } from "wouter";
import { Star, Menu, Share2, Download, AlertCircle, BookMarked } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { useToast } from "@/hooks/use-toast";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

export default function WikiPage() {
  const [location, setLocation] = useLocation();
  const searchStr = useSearch();
  const searchParams = new URLSearchParams(searchStr);
  const currentPath = searchParams.get("path");

  const { data: fileTree, isLoading: isLoadingTree } = useFileTree();
  const { data: contentData, isLoading: isLoadingContent, error } = useFileContent(currentPath);
  const { data: favorites } = useFavorites();
  const toggleFavorite = useToggleFavorite();
  const { toast } = useToast();

  const isFavorite = favorites?.some(f => f.filePath === currentPath);

  const handleNavigate = (path: string) => {
    setLocation(`/?path=${encodeURIComponent(path)}`);
  };

  const handleFavorite = () => {
    if (!currentPath) return;
    
    toggleFavorite.mutate({
      filePath: currentPath,
      title: currentPath.split('/').pop() || "Untitled"
    }, {
      onSuccess: (data) => {
        toast({
          title: data.added ? "Added to favorites" : "Removed from favorites",
          description: data.added 
            ? "This page is now in your bookmarks." 
            : "This page has been removed from your bookmarks.",
        });
      }
    });
  };

  // Mobile drawer state needed only for local mobile trigger
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      {/* Sidebar Navigation */}
      <Sidebar 
        files={fileTree} 
        favorites={favorites}
        selectedPath={currentPath || undefined} 
        onSelect={(path) => {
          handleNavigate(path);
          setIsMobileOpen(false);
        }}
        isLoading={isLoadingTree}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto">
        {/* Header */}
        <header className="sticky top-0 z-30 flex items-center justify-between px-4 py-3 bg-background/80 backdrop-blur-md border-b border-border/40">
          <div className="flex items-center gap-2 overflow-hidden">
            {/* Mobile Menu Trigger */}
            <div className="md:hidden">
              <Sheet open={isMobileOpen} onOpenChange={setIsMobileOpen}>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon" className="-ml-2">
                    <Menu className="w-5 h-5" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="p-0 w-80 border-r-border">
                  <Sidebar 
                    files={fileTree} 
                    favorites={favorites}
                    selectedPath={currentPath || undefined} 
                    onSelect={(path) => {
                      handleNavigate(path);
                      setIsMobileOpen(false);
                    }}
                    isLoading={isLoadingTree}
                  />
                </SheetContent>
              </Sheet>
            </div>
            
            <div className="flex flex-col min-w-0">
              <nav className="text-xs text-muted-foreground flex items-center gap-1 overflow-hidden whitespace-nowrap mask-linear-fade">
                 <span>wiki</span>
                 {currentPath?.split('/').map((part, i) => (
                   <span key={i} className="flex items-center">
                     <span className="mx-1 opacity-50">/</span>
                     <span className="truncate">{part}</span>
                   </span>
                 ))}
              </nav>
            </div>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            {currentPath && (
              <>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={handleFavorite}
                  disabled={toggleFavorite.isPending}
                  className={isFavorite ? "text-yellow-500 hover:text-yellow-600" : "text-muted-foreground"}
                >
                  <Star className={isFavorite ? "fill-current w-5 h-5" : "w-5 h-5"} />
                </Button>
                <Button variant="ghost" size="icon" className="hidden sm:inline-flex">
                  <Share2 className="w-4 h-4" />
                </Button>
              </>
            )}
          </div>
        </header>

        {/* Content Body */}
        <div className="flex-1 p-4 md:p-8 lg:p-12 max-w-5xl mx-auto w-full">
          {!currentPath ? (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center p-6">
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="max-w-md space-y-4"
              >
                <div className="w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <BookMarked className="w-10 h-10 text-primary" />
                </div>
                <h2 className="text-3xl font-display font-bold text-foreground">Welcome to Wiki</h2>
                <p className="text-muted-foreground text-lg leading-relaxed">
                  Select a document from the sidebar to start reading. You can search, favorite, and explore the knowledge base.
                </p>
                <Button 
                  size="lg" 
                  className="mt-6 shadow-lg shadow-primary/25"
                  onClick={() => {
                    // Try to find README.md or first file
                    const findFirstFile = (nodes: any[]): string | null => {
                      for (const node of nodes) {
                        if (node.type === 'file') return node.path;
                        if (node.children) {
                          const child = findFirstFile(node.children);
                          if (child) return child;
                        }
                      }
                      return null;
                    };
                    const first = fileTree ? findFirstFile(fileTree) : null;
                    if (first) handleNavigate(first);
                  }}
                >
                  Get Started
                </Button>
              </motion.div>
            </div>
          ) : isLoadingContent ? (
            <div className="space-y-8 max-w-3xl animate-in fade-in duration-500">
              <Skeleton className="h-12 w-3/4 rounded-lg" />
              <div className="space-y-4">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
              </div>
              <div className="space-y-4 pt-8">
                <Skeleton className="h-8 w-1/3" />
                <Skeleton className="h-32 w-full rounded-xl" />
              </div>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-[50vh] text-center">
              <AlertCircle className="w-12 h-12 text-destructive mb-4" />
              <h3 className="text-xl font-bold">Could not load document</h3>
              <p className="text-muted-foreground mt-2">
                The file you are looking for might have been moved or deleted.
              </p>
            </div>
          ) : contentData?.isBinary ? (
            <div className="flex flex-col items-center justify-center min-h-[50vh] p-4 bg-muted/30 rounded-2xl border border-border/40">
              {contentData.type.startsWith('image/') ? (
                <img 
                  src={contentData.data} 
                  alt={currentPath.split('/').pop()} 
                  className="max-w-full max-h-[70vh] rounded-xl shadow-2xl animate-in zoom-in-95 duration-500"
                />
              ) : (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                    <Download className="w-8 h-8 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold">Binary File</h3>
                  <p className="text-muted-foreground">This file cannot be displayed directly.</p>
                  <a href={contentData.data} download={currentPath.split('/').pop()}>
                    <Button>Download File</Button>
                  </a>
                </div>
              )}
            </div>
          ) : (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="mb-8 pb-4 border-b border-border/40">
                <h1 className="text-3xl md:text-4xl lg:text-5xl font-serif font-bold text-foreground mb-4">
                  {currentPath.split('/').pop()?.replace('.md', '')}
                </h1>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                   <span>{Math.ceil((contentData?.content.length || 0) / 1000)} min read</span>
                   <span>•</span>
                   <span className="font-mono bg-muted px-2 py-0.5 rounded text-xs">Markdown</span>
                </div>
              </div>
              
              <MarkdownRenderer content={contentData?.content || ''} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
