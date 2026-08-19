import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

import { common } from "../lib/copy";
import { CloseIcon } from "./icons";
import { cx } from "./lib/cx";

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}

/** A centered dialog over a dimmed backdrop — confirmations, the template
 * picker, anything that has to interrupt. Escape and a backdrop click both
 * close it; nothing else on the page scrolls behind it. */
export function Modal({ open, onClose, title, children, footer, className }: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    dialogRef.current?.focus();

    const { overflow } = document.body.style;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = overflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/20 p-4">
      <div role="presentation" className="absolute inset-0" onClick={onClose} />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={cx(
          "echo-card relative z-10 w-full max-w-md p-6 shadow-lift focus:outline-none",
          className,
        )}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label={common.close}
          className="absolute right-4 top-4 rounded-full p-1 text-ink-ghost transition-colors hover:bg-surface-sunken hover:text-ink"
        >
          <CloseIcon className="h-4 w-4" />
        </button>
        {title && <h2 className="mb-3 pr-8 text-base font-semibold text-ink">{title}</h2>}
        <div className="text-sm text-ink-soft">{children}</div>
        {footer && <div className="mt-6 flex justify-end gap-2">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}
