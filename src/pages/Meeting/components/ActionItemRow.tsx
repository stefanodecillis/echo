import { useState } from "react";

import { Chip } from "@/components";
import { CheckIcon } from "@/components/icons";
import { cx } from "@/components/lib/cx";
import { meeting as copy } from "@/lib/copy";
import type { ActionItem } from "@/lib/types";

export interface ActionItemRowProps {
  item: ActionItem;
  onToggleDone: (item: ActionItem) => void;
  onOwnerChange: (item: ActionItem, owner: string) => void;
}

/** One action item: a done toggle, the description, and an editable owner
 * chip — click it to type a name, Enter or blur to save. */
export function ActionItemRow({ item, onToggleDone, onOwnerChange }: ActionItemRowProps) {
  const [editingOwner, setEditingOwner] = useState(false);
  const [ownerDraft, setOwnerDraft] = useState(item.owner ?? "");

  const commitOwner = () => {
    setEditingOwner(false);
    const trimmed = ownerDraft.trim();
    if (trimmed !== (item.owner ?? "")) onOwnerChange(item, trimmed);
  };

  return (
    <li className="flex items-start gap-3 py-2.5">
      <button
        type="button"
        role="checkbox"
        aria-checked={item.done}
        aria-label={item.done ? copy.actionItemMarkNotDone : copy.actionItemMarkDone}
        onClick={() => onToggleDone(item)}
        className={cx(
          "mt-0.5 flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-md border transition-colors",
          item.done ? "border-ink bg-ink text-white" : "border-hairline text-transparent hover:border-ink-ghost",
        )}
      >
        <CheckIcon className="h-3 w-3" />
      </button>

      <div className="flex flex-1 flex-wrap items-center gap-2">
        <span className={cx("text-sm", item.done ? "text-ink-ghost line-through" : "text-ink")}>
          {item.description}
        </span>
        {item.dueHint && <Chip variant="outline">{item.dueHint}</Chip>}
        {editingOwner ? (
          <input
            autoFocus
            value={ownerDraft}
            onChange={(e) => setOwnerDraft(e.target.value)}
            onBlur={commitOwner}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitOwner();
              if (e.key === "Escape") {
                setOwnerDraft(item.owner ?? "");
                setEditingOwner(false);
              }
            }}
            placeholder={copy.ownerPlaceholder}
            className="w-32 rounded-full border border-hairline bg-surface px-2.5 py-0.5 text-xs text-ink focus:outline-none focus:ring-2 focus:ring-accent/30"
          />
        ) : (
          <button type="button" onClick={() => setEditingOwner(true)}>
            <Chip variant={item.owner ? "solid" : "neutral"}>{item.owner || copy.ownerPlaceholder}</Chip>
          </button>
        )}
      </div>
    </li>
  );
}
