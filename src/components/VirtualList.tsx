import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { ForwardedRef, ReactElement, ReactNode, UIEvent } from "react";

import { cx } from "./lib/cx";

/** What to assume a viewport holds before it has been measured. Generous on
 * purpose: too many rows is a wasted paint, too few is a hidden meeting. */
const UNMEASURED_VIEWPORT_PX = 2000;

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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  // Measured through a callback ref, not a mount effect, because the container
  // is not always there at mount: a transcript starts empty, the empty state
  // renders instead, and a `[]` effect would look once, find no element and
  // never look again. The height then stays 0 for the life of the meeting, and
  // 0 plus the overscan is exactly four rows — which is what a person saw
  // instead of their meeting on 2026-08-24 and again on 2026-08-25.
  const observerRef = useRef<ResizeObserver | null>(null);
  const measure = useCallback((el: HTMLDivElement | null) => {
    containerRef.current = el;
    observerRef.current?.disconnect();
    if (!el) return;
    // Read it once directly: a ResizeObserver reports the first size on its own
    // schedule, and until then the window would be empty.
    setViewportHeight(el.clientHeight);
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setViewportHeight(entry.contentRect.height);
    });
    observer.observe(el);
    observerRef.current = observer;
  }, []);

  useEffect(() => () => observerRef.current?.disconnect(), []);

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
        // The element's own height, not one worked out from the row count: the
        // count in this closure is whatever it was when the handle was last
        // built, and being one row short of the bottom is what leaves a live
        // transcript looking like it stopped following along.
        el.scrollTop = el.scrollHeight;
      },
    }),
    [itemHeight, items.length, viewportHeight],
  );

  if (items.length === 0) {
    return <div className={className}>{emptyState}</div>;
  }

  const totalHeight = items.length * itemHeight;
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  // A viewport that has not been measured yet draws a screenful rather than
  // nothing: being wrong by rendering a few too many rows costs a moment of
  // layout, while being wrong the other way hides the meeting.
  const window = viewportHeight > 0 ? viewportHeight : UNMEASURED_VIEWPORT_PX;
  const endIndex = Math.min(
    items.length,
    Math.ceil((scrollTop + window) / itemHeight) + overscan,
  );

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    const top = event.currentTarget.scrollTop;
    setScrollTop(top);
    if (onNearStart && top < itemHeight * 2) onNearStart();
  };

  return (
    <div
      ref={measure}
      // Deliberately no `relative` here. The rows position against the spacer
      // below, which is relative in its own right — and a caller that places
      // this list with `absolute inset-0` was losing that fight, because
      // Tailwind emits `.relative` after `.absolute` and the later rule wins.
      // The container then had no height of its own to scroll inside: the live
      // transcript could not be scrolled, and following the newest line did
      // nothing, because there was nowhere to scroll to.
      className={cx("overflow-y-auto", className)}
      onScroll={handleScroll}
    >
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
