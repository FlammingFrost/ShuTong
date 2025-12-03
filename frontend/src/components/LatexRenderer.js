import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

/**
 * Component to render text with LaTeX math expressions
 * Supports both inline ($...$) and display ($$...$$) math
 */
const LatexRenderer = ({ content, className = '' }) => {
  if (!content) return null;

  // Helper function to render text with markdown formatting (bold, line breaks)
  const renderTextWithMarkdown = (text, trimStart = false, trimEnd = false) => {
    const parts = [];
    let key = 0;

    // Trim text if needed
    let processedText = text;
    if (trimStart) {
      processedText = processedText.replace(/^\n+/, '');
    }
    if (trimEnd) {
      processedText = processedText.replace(/\n+$/, '');
    }

    // Split by newlines first
    const lines = processedText.split('\n');
    
    lines.forEach((line, lineIdx) => {
      const lineParts = [];
      // Handle bold text (**text**)
      const boldRegex = /\*\*(.*?)\*\*/g;
      let match;
      let lastIndex = 0;

      while ((match = boldRegex.exec(line)) !== null) {
        // Add text before bold
        if (match.index > lastIndex) {
          lineParts.push(line.slice(lastIndex, match.index));
        }
        // Add bold text
        lineParts.push(
          <strong key={`bold-${lineIdx}-${key++}`}>{match[1]}</strong>
        );
        lastIndex = match.index + match[0].length;
      }
      // Add remaining text
      if (lastIndex < line.length) {
        lineParts.push(line.slice(lastIndex));
      }

      // Add the line with a line break
      if (lineParts.length > 0) {
        parts.push(
          <span key={`line-${lineIdx}`}>
            {lineParts}
          </span>
        );
      } else if (line.length === 0) {
        // Empty line - add just a break
        parts.push(<br key={`br-${lineIdx}`} />);
      } else {
        parts.push(
          <span key={`line-${lineIdx}`}>{line}</span>
        );
      }
      
      // Add line break except for the last line
      if (lineIdx < lines.length - 1) {
        parts.push(<br key={`br-after-${lineIdx}`} />);
      }
    });

    return parts;
  };

  // Split content by display math ($$...$$), \begin{align}...\end{align}, and inline math ($...$)
  const renderContent = (text) => {
    const parts = [];
    let key = 0;

    // First, handle display math ($$...$$) and LaTeX environments like \begin{align}...\end{align}
    const displayMathRegex = /\$\$(.*?)\$\$/gs;
    const alignRegex = /\\begin\{(align|equation|gather|alignat|multline|flalign|split)\*?\}([\s\S]*?)\\end\{\1\*?\}/g;
    
    let segments = [];
    let lastIndex = 0;
    
    // Find all matches (both $$ and \begin{...})
    const allMatches = [];
    let match;
    
    // Find $$ matches
    while ((match = displayMathRegex.exec(text)) !== null) {
      allMatches.push({
        index: match.index,
        length: match[0].length,
        content: match[1].trim(),
        type: 'display'
      });
    }
    
    // Find \begin{...} matches
    while ((match = alignRegex.exec(text)) !== null) {
      allMatches.push({
        index: match.index,
        length: match[0].length,
        content: match[2].trim(),
        type: 'display'
      });
    }
    
    // Sort by index
    allMatches.sort((a, b) => a.index - b.index);
    
    // Build segments
    for (const m of allMatches) {
      // Add text before this match (trim trailing whitespace/newlines)
      if (m.index > lastIndex) {
        const textContent = text.slice(lastIndex, m.index).replace(/\n+$/, '');
        if (textContent) {
          segments.push({ type: 'text', content: textContent });
        }
      }
      // Add display math
      segments.push({ type: 'display', content: m.content });
      lastIndex = m.index + m.length;
    }
    // Add remaining text (trim leading whitespace/newlines)
    if (lastIndex < text.length) {
      const textContent = text.slice(lastIndex).replace(/^\n+/, '');
      if (textContent) {
        segments.push({ type: 'text', content: textContent });
      }
    }

    // Now handle inline math in text segments
    segments.forEach((segment, segmentIdx) => {
      if (segment.type === 'display') {
        parts.push(
          <div key={`display-${segmentIdx}`} style={{ margin: 0, padding: 0, lineHeight: 1 }}>
            <BlockMath math={segment.content} />
          </div>
        );
      } else {
        // Check if this text segment comes after a display math
        const prevSegment = segmentIdx > 0 ? segments[segmentIdx - 1] : null;
        const nextSegment = segmentIdx < segments.length - 1 ? segments[segmentIdx + 1] : null;
        const trimStart = prevSegment && prevSegment.type === 'display';
        const trimEnd = nextSegment && nextSegment.type === 'display';
        
        // Handle inline math in this text segment
        const inlineMathRegex = /\$([^$]+?)\$/g;
        let inlineMatch;
        let inlineLastIndex = 0;
        const inlineParts = [];

        while ((inlineMatch = inlineMathRegex.exec(segment.content)) !== null) {
          // Add text before inline math (with markdown support)
          if (inlineMatch.index > inlineLastIndex) {
            const textBefore = segment.content.slice(inlineLastIndex, inlineMatch.index);
            inlineParts.push(
              <span key={`text-${segmentIdx}-${key++}`}>
                {renderTextWithMarkdown(textBefore, trimStart && inlineLastIndex === 0, false)}
              </span>
            );
          }
          // Add inline math
          inlineParts.push(
            <InlineMath key={`inline-${segmentIdx}-${key++}`} math={inlineMatch[1].trim()} />
          );
          inlineLastIndex = inlineMatch.index + inlineMatch[0].length;
        }
        // Add remaining text (with markdown support)
        if (inlineLastIndex < segment.content.length) {
          const textAfter = segment.content.slice(inlineLastIndex);
          inlineParts.push(
            <span key={`text-${segmentIdx}-${key++}`}>
              {renderTextWithMarkdown(textAfter, trimStart && inlineLastIndex === 0, trimEnd)}
            </span>
          );
        }

        if (inlineParts.length > 0) {
          parts.push(...inlineParts);
        } else {
          parts.push(
            <span key={`text-${segmentIdx}`}>
              {renderTextWithMarkdown(segment.content, trimStart, trimEnd)}
            </span>
          );
        }
      }
    });

    return parts;
  };

  return (
    <div className={`latex-content ${className}`}>
      {renderContent(content)}
    </div>
  );
};

export default LatexRenderer;
