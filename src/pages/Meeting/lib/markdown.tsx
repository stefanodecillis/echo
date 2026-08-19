/**
 * A tiny, dependency-free markdown renderer for recap content.
 *
 * The backend already sanitizes `contentMd` before it reaches us (see
 * docs/DESIGN.md's connector section: "summary markdown sanitized before
 * render — no raw HTML, no remote resources"), but this renderer never
 * touches `dangerouslySetInnerHTML` regardless: every block and inline span
 * becomes a real React element, so there is no HTML string to sanitize in
 * the first place. Defense in depth costs nothing here.
 *
 * Supports the shapes a recap actually uses: headings (# / ## / ###),
 * paragraphs, bullet and numbered lists, blockquotes, a horizontal rule, and
 * inline bold, italic and inline code. Anything else renders as plain text
 * rather than breaking.
 */
import type { ReactNode } from "react";

type Block =
  | { type: "heading"; level: 1 | 2 | 3; text: string }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] }
  | { type: "quote"; text: string }
  | { type: "hr" }
  | { type: "p"; text: string };

const HEADING_RE = /^(#{1,3})\s+(.*)$/;
const HR_RE = /^(-{3,}|\*{3,})\s*$/;
const QUOTE_RE = /^>\s?(.*)$/;
const UL_RE = /^\s*[-*]\s+(.*)$/;
const OL_RE = /^\s*\d+[.)]\s+(.*)$/;

function isSpecialLine(line: string): boolean {
  return (
    HEADING_RE.test(line) ||
    HR_RE.test(line.trim()) ||
    QUOTE_RE.test(line) ||
    UL_RE.test(line) ||
    OL_RE.test(line)
  );
}

function parseBlocks(md: string): Block[] {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === "") {
      i += 1;
      continue;
    }

    const heading = HEADING_RE.exec(line);
    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1].length as 1 | 2 | 3,
        text: heading[2].trim(),
      });
      i += 1;
      continue;
    }

    if (HR_RE.test(line.trim())) {
      blocks.push({ type: "hr" });
      i += 1;
      continue;
    }

    const quote = QUOTE_RE.exec(line);
    if (quote) {
      const parts: string[] = [quote[1]];
      i += 1;
      while (i < lines.length) {
        const next = QUOTE_RE.exec(lines[i]);
        if (!next) break;
        parts.push(next[1]);
        i += 1;
      }
      blocks.push({ type: "quote", text: parts.join(" ") });
      continue;
    }

    if (UL_RE.test(line)) {
      const items: string[] = [];
      while (i < lines.length) {
        const item = UL_RE.exec(lines[i]);
        if (!item) break;
        items.push(item[1]);
        i += 1;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    if (OL_RE.test(line)) {
      const items: string[] = [];
      while (i < lines.length) {
        const item = OL_RE.exec(lines[i]);
        if (!item) break;
        items.push(item[1]);
        i += 1;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    const paraLines: string[] = [];
    while (i < lines.length && lines[i].trim() !== "" && !isSpecialLine(lines[i])) {
      paraLines.push(lines[i]);
      i += 1;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "p", text: paraLines.join(" ") });
    }
  }

  return blocks;
}

const INLINE_RE = /\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*|_([^_]+)_/g;

/** Bold, code and italic spans inside one line of text — plain strings for
 * everything else, so React escapes them the normal way. */
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let key = 0;
  let match: RegExpExecArray | null;

  INLINE_RE.lastIndex = 0;
  while ((match = INLINE_RE.exec(text))) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    if (match[1] !== undefined) {
      nodes.push(<strong key={`${keyPrefix}-${key++}`}>{match[1]}</strong>);
    } else if (match[2] !== undefined) {
      nodes.push(
        <code
          key={`${keyPrefix}-${key++}`}
          className="rounded bg-surface-sunken px-1 py-0.5 text-[0.85em] text-ink-soft"
        >
          {match[2]}
        </code>,
      );
    } else if (match[3] !== undefined) {
      nodes.push(<em key={`${keyPrefix}-${key++}`}>{match[3]}</em>);
    } else if (match[4] !== undefined) {
      nodes.push(<em key={`${keyPrefix}-${key++}`}>{match[4]}</em>);
    }
    lastIndex = INLINE_RE.lastIndex;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

const headingClasses: Record<1 | 2 | 3, string> = {
  1: "text-lg font-semibold text-ink",
  2: "text-base font-semibold text-ink",
  3: "text-sm font-semibold uppercase tracking-wide text-ink-faint",
};

export interface RecapMarkdownProps {
  content: string;
  className?: string;
}

/** Renders sanitized recap markdown as real React elements. */
export function RecapMarkdown({ content, className }: RecapMarkdownProps) {
  const blocks = parseBlocks(content);

  if (blocks.length === 0) return null;

  return (
    <div className={className}>
      {blocks.map((block, index) => {
        const key = `block-${index}`;
        switch (block.type) {
          case "heading": {
            const Tag = (`h${block.level}` as unknown) as "h1" | "h2" | "h3";
            return (
              <Tag key={key} className={cxHeading(block.level, index)}>
                {renderInline(block.text, key)}
              </Tag>
            );
          }
          case "ul":
            return (
              <ul key={key} className="mb-3 ml-5 list-disc space-y-1 text-sm text-ink-soft">
                {block.items.map((item, i) => (
                  <li key={i}>{renderInline(item, `${key}-${i}`)}</li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={key} className="mb-3 ml-5 list-decimal space-y-1 text-sm text-ink-soft">
                {block.items.map((item, i) => (
                  <li key={i}>{renderInline(item, `${key}-${i}`)}</li>
                ))}
              </ol>
            );
          case "quote":
            return (
              <blockquote
                key={key}
                className="mb-3 border-l-2 border-hairline pl-3 text-sm italic text-ink-faint"
              >
                {renderInline(block.text, key)}
              </blockquote>
            );
          case "hr":
            return <hr key={key} className="my-4 border-hairline" />;
          case "p":
            return (
              <p key={key} className="mb-3 text-sm leading-relaxed text-ink-soft">
                {renderInline(block.text, key)}
              </p>
            );
        }
      })}
    </div>
  );
}

function cxHeading(level: 1 | 2 | 3, index: number): string {
  const spacing = index === 0 ? "mb-2" : "mb-2 mt-5";
  return `${spacing} ${headingClasses[level]}`;
}
