import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { ForwardedRef, ReactElement, ReactNode, UIEvent } from "react";

import { cx } from "./lib/cx";

export interface VirtualListHandle {
  scrollToIndex: (index: number, opts?: { align?: "start" | "end" }) => void;
  scrollToBottom: () => void;
}

export interface VirtualListProps<T> {
  items: T[];
  /** Every row is this tall. Transcript lines and meeting rows are uniform
   * enough that a thin windowing wrapper doesn't need variable-height support. */
  itemHeight: number;
  renderItem: (item: T, index: number) => ReactNode;
  getKey?: (item: T, index: number) => string | number;
  overscan?: number;
  className?: string;
  emptyState?: ReactNode;
  /** Fires while scrolled near the top — for loading older history. */
  onNearStart?: () => void;
}

function VirtualListInner<T>(
  {
    items,
    itemHeight,
    renderItem,
    getKey,
    overscan = 4,
    className,
    emptyState,
    onNearStart,
  }: VirtualListProps<T>,
  ref: ForwardedRef<VirtualListHandle>,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setViewportHeight(entry.contentRect.height);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useImperativeHandle(
    ref,
    () => ({
      scrollToIndex(index, opts) {
        const el = containerRef.current;
        if (!el) return;
        const top = index * itemHeight;
        el.scrollTop = opts?.align === "end" ? top - viewportHeight + itemHeight : top;
      },
      scrollToBottom() {
        const el = containerRef.current;
        if (!el) return;
        el.scrollTop = items.length * itemHeight;
      },
    }),
    [itemHeight, items.length, viewportHeight],
  );

  if (items.length === 0) {
    return <div className={className}>{emptyState}</div>;
  }

  const totalHeight = items.length * itemHeight;
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(
    items.length,
    Math.ceil((scrollTop + viewportHeight) / itemHeight) + overscan,
  );

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const top = event.currentTarget.scrollTop;
    setScrollTop(top);
    if (onNearStart && top < itemHeight * 2) onNearStart();
  };

  return (
    <div ref={containerRef} className={cx("relative overflow-y-auto", className)} onScroll={handleScroll}>
      <div style={{ height: totalHeight, position: "relative" }}>
        {items.slice(startIndex, endIndex).map((item, i) => {
          const index = startIndex + i;
          return (
            <div
              key={getKey ? getKey(item, index) : index}
              style={{ position: "absolute", top: index * itemHeight, left: 0, right: 0, height: itemHeight }}
            >
              {renderItem(item, index)}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** A thin windowing wrapper for long transcripts and meeting lists — no
 * heavy dependency, just fixed-height rows and a `ResizeObserver`. */
export const VirtualList = forwardRef(VirtualListInner) as <T>(
  props: VirtualListProps<T> & { ref?: ForwardedRef<VirtualListHandle> },
) => ReactElement | null;
