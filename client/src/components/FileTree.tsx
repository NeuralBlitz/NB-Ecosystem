import { useState } from 'react';
import { Folder, FolderOpen, FileText, ChevronRight, ChevronDown, FileCode, FileImage } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { FileNode } from '@/hooks/use-files';

interface FileTreeProps {
  nodes: FileNode[];
  selectedPath?: string;
  onSelect: (path: string) => void;
  depth?: number;
}

const getFileIcon = (name: string) => {
  if (name.endsWith('.md')) return <FileText className="w-4 h-4 text-blue-500" />;
  if (name.endsWith('.ts') || name.endsWith('.js') || name.endsWith('.json')) return <FileCode className="w-4 h-4 text-yellow-500" />;
  if (name.endsWith('.png') || name.endsWith('.jpg')) return <FileImage className="w-4 h-4 text-purple-500" />;
  return <FileText className="w-4 h-4 text-muted-foreground" />;
};

export function FileTree({ nodes, selectedPath, onSelect, depth = 0 }: FileTreeProps) {
  // Sort: Directories first, then files (alphabetical)
  const sortedNodes = [...nodes].sort((a, b) => {
    if (a.type === b.type) return a.name.localeCompare(b.name);
    return a.type === 'directory' ? -1 : 1;
  });

  return (
    <div className="select-none">
      {sortedNodes.map((node) => (
        <FileTreeNode 
          key={node.path} 
          node={node} 
          selectedPath={selectedPath} 
          onSelect={onSelect} 
          depth={depth} 
        />
      ))}
    </div>
  );
}

interface FileTreeNodeProps {
  node: FileNode;
  selectedPath?: string;
  onSelect: (path: string) => void;
  depth: number;
}

function FileTreeNode({ node, selectedPath, onSelect, depth }: FileTreeNodeProps) {
  const [isOpen, setIsOpen] = useState(false);
  const isSelected = selectedPath === node.path;

  // Auto-expand if child is selected
  // Note: In a real complex app we'd lift this state up, but simple recursion works for now
  
  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.type === 'directory') {
      setIsOpen(!isOpen);
    } else {
      onSelect(node.path);
    }
  };

  return (
    <div>
      <div 
        onClick={handleToggle}
        className={cn(
          "flex items-center py-1.5 px-2 cursor-pointer transition-colors duration-200 group rounded-md mx-2 text-sm",
          isSelected 
            ? "bg-primary/10 text-primary font-medium" 
            : "hover:bg-muted/50 text-foreground/80 hover:text-foreground"
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        <span className="mr-2 opacity-70 group-hover:opacity-100 transition-opacity">
          {node.type === 'directory' ? (
            isOpen ? <FolderOpen className="w-4 h-4 text-primary" /> : <Folder className="w-4 h-4" />
          ) : (
            getFileIcon(node.name)
          )}
        </span>
        
        <span className="truncate">{node.name}</span>
        
        {node.type === 'directory' && (
          <span className="ml-auto opacity-30">
            {isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </span>
        )}
      </div>

      <AnimatePresence>
        {node.type === 'directory' && isOpen && node.children && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <FileTree 
              nodes={node.children} 
              selectedPath={selectedPath} 
              onSelect={onSelect} 
              depth={depth + 1} 
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
