import { EchoMark } from "../components";
import { cx } from "../components/lib/cx";

/**
 * `radiate` = a meeting is sitting there undecided (a ring leaves the tile every
 * 2.5s). `breathe` = Echo is listening (the mark itself pulses). `still` =
 * paused, where any motion at all would be a lie.
 */
export type PanelTileMotion = "still" | "radiate" | "breathe";

export interface PanelTileProps {
  motion?: PanelTileMotion;
}

/**
 * Echo's mark, white on an ink tile — the panel's anchor, and the one thing on
 * it that moves. Monochrome on purpose: the accent is spent elsewhere
 * (docs/DESIGN.md §4, one accent used sparingly), so the tile carries the
 * app's identity with contrast rather than colour.
 */
export function PanelTile({ motion = "still" }: PanelTileProps) {
  return (
    <span className="relative flex h-9 w-9 shrink-0 items-center justify-center">
      {/* 10px, not the card's 20px: a corner nested inside another one has to be
          tighter than its parent or the tile reads as a circle that missed. */}
      {motion === "radiate" && (
        <span
          aria-hidden
          className="animate-tile-radiate absolute inset-0 rounded-[10px] border border-ink"
        />
      )}
      <span className="relative flex h-9 w-9 items-center justify-center rounded-[10px] bg-ink text-white">
        <EchoMark
          className={cx("h-[19px] w-[19px]", motion === "breathe" && "animate-mark-breathe")}
        />
      </span>
    </span>
  );
}
