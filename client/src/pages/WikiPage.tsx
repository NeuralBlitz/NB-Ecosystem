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
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto bg-background/50 selection:bg-primary/20">
        {/* Header */}
        <header className="sticky top-0 z-30 flex items-center justify-between px-4 h-16 bg-background/60 backdrop-blur-xl border-b border-border/10">
          <div className="flex items-center gap-4 overflow-hidden">
            {/* Mobile Menu Trigger */}
            <div className="md:hidden">
              <Sheet open={isMobileOpen} onOpenChange={setIsMobileOpen}>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon" className="hover:bg-muted/40 rounded-xl">
                    <Menu className="w-5 h-5" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="p-0 w-80 border-r-border/20 bg-background shadow-2xl">
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
              <nav className="text-[10px] font-bold tracking-[0.2em] text-muted-foreground/60 uppercase flex items-center gap-2 overflow-hidden whitespace-nowrap">
                 <span>WIKI</span>
                 {currentPath?.split('/').map((part, i) => (
                   <span key={i} className="flex items-center">
                     <span className="mx-0.5 opacity-30">/</span>
                     <span className="truncate opacity-80">{part}</span>
                   </span>
                 ))}
              </nav>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {currentPath && (
              <>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={handleFavorite}
                  disabled={toggleFavorite.isPending}
                  className={`rounded-xl transition-all ${isFavorite ? "text-yellow-500 bg-yellow-500/10 hover:bg-yellow-500/20" : "text-muted-foreground hover:bg-muted/40"}`}
                >
                  <Star className={isFavorite ? "fill-current w-5 h-5" : "w-5 h-5"} />
                </Button>
                <Button variant="ghost" size="icon" className="hidden sm:inline-flex rounded-xl hover:bg-muted/40">
                  <Share2 className="w-4 h-4" />
                </Button>
              </>
            )}
          </div>
        </header>

        {/* Content Body */}
        <div className="flex-1 p-6 md:p-12 lg:p-20 max-w-4xl mx-auto w-full">
          {!currentPath ? (
            <div className="flex flex-col items-center justify-center h-[70vh] text-center p-6">
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                className="max-w-md space-y-6"
              >
                <div className="w-24 h-24 bg-primary/5 rounded-3xl flex items-center justify-center mx-auto mb-8 ring-1 ring-white/5 shadow-2xl shadow-primary/10">
                  <BookMarked className="w-12 h-12 text-primary" />
                </div>
                <h2 className="text-4xl md:text-5xl font-display font-bold text-foreground tracking-tight leading-tight">Your Digital Library</h2>
                <p className="text-muted-foreground/70 text-lg leading-relaxed font-medium">
                  Explore and search your technical knowledge base with a modern, distraction-free experience.
                </p>
                <div className="pt-6">
                  <Button 
                    size="lg" 
                    className="h-14 px-10 rounded-2xl text-lg font-bold shadow-2xl shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] transition-all"
                    onClick={() => {
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
                    Start Browsing
                  </Button>
                </div>
              </motion.div>
            </div>
          ) : isLoadingContent ? (
            <div className="space-y-12 max-w-2xl animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out">
              <Skeleton className="h-16 w-3/4 rounded-2xl bg-muted/20" />
              <div className="space-y-6">
                <Skeleton className="h-4 w-full rounded-full bg-muted/10" />
                <Skeleton className="h-4 w-full rounded-full bg-muted/10" />
                <Skeleton className="h-4 w-5/6 rounded-full bg-muted/10" />
              </div>
              <div className="space-y-6 pt-12">
                <Skeleton className="h-10 w-1/3 rounded-xl bg-muted/20" />
                <Skeleton className="h-64 w-full rounded-3xl bg-muted/10" />
              </div>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-[50vh] text-center p-6 bg-destructive/5 rounded-3xl border border-destructive/10">
              <AlertCircle className="w-12 h-12 text-destructive/50 mb-4" />
              <h3 className="text-2xl font-bold">Document Not Found</h3>
              <p className="text-muted-foreground mt-2 max-w-sm">
                We couldn't locate this file. It may have been moved, renamed, or deleted.
              </p>
              <Button variant="outline" className="mt-8 rounded-xl" onClick={() => setLocation('/')}>
                Back to Home
              </Button>
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
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out">
              <div className="mb-12 pb-8 border-b border-border/10">
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-display font-bold text-foreground tracking-tight leading-tight mb-6">
                  {currentPath.split('/').pop()?.replace('.md', '')}
                </h1>
                <div className="flex items-center gap-5 text-sm font-medium text-muted-foreground/60">
                   <div className="flex items-center gap-2">
                     <span className="w-1.5 h-1.5 rounded-full bg-primary/40" />
                     <span>{Math.ceil((contentData?.content.length || 0) / 1000)} min read</span>
                   </div>
                   <span className="opacity-20">|</span>
                   <span className="font-mono bg-muted/30 px-3 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider">
                     {contentData?.type?.split('/')[1] || 'DOC'}
                   </span>
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
