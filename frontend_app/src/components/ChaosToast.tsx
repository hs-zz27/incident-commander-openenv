"use client";

import { useEffect, useState } from "react";

interface ChaosToastProps {
  /** Service name or label when ChaosAgent injects this step */
  message: string | null;
}

/**
 * Small fixed overlay — does not change main dashboard layout.
 */
export default function ChaosToast({ message }: ChaosToastProps) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");

  useEffect(() => {
    if (!message) return;
    setText(message);
    setOpen(true);
    const t = window.setTimeout(() => setOpen(false), 4500);
    return () => window.clearTimeout(t);
  }, [message]);

  if (!open || !text) return null;

  return (
    <div
      className="fixed top-24 right-8 z-[55] max-w-sm rounded-lg border border-fuchsia-500/40 bg-fuchsia-500/15 px-4 py-3 text-fuchsia-100 shadow-lg backdrop-blur-md font-body-sm"
      role="status"
    >
      <div className="flex items-start gap-2">
        <span className="material-symbols-outlined text-base shrink-0">bolt</span>
        <div>
          <p className="text-[10px] uppercase tracking-widest text-fuchsia-200/90 mb-0.5">
            Chaos injection
          </p>
          <p className="font-mono text-sm text-fuchsia-50">{text}</p>
        </div>
      </div>
    </div>
  );
}
