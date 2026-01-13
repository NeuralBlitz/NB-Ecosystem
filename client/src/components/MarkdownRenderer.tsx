import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import { motion } from 'framer-motion';

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="prose prose-slate dark:prose-invert max-w-none w-full pb-20"
    >
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          // Custom renderers if needed
          img: ({node, ...props}) => (
            <img 
              {...props} 
              className="rounded-xl shadow-md my-8 max-h-[500px] object-contain mx-auto bg-white/5" 
              loading="lazy"
            />
          ),
          table: ({node, ...props}) => (
            <div className="overflow-x-auto my-8 rounded-lg border border-border">
              <table {...props} className="w-full text-sm" />
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </motion.div>
  );
}
