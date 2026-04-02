import type { Action } from "@/types";

interface Props {
  action: Action;
  size?: "sm" | "md" | "lg";
}

const classes: Record<Action, string> = {
  BUY: "badge-buy",
  SELL: "badge-sell",
  HOLD: "badge-hold",
  AVOID: "badge-avoid",
};

const sizes = {
  sm: "text-[10px] px-2 py-0.5",
  md: "text-xs px-2.5 py-1",
  lg: "text-sm px-3 py-1.5",
};

export default function ActionBadge({ action, size = "md" }: Props) {
  return (
    <span className={`${classes[action]} ${sizes[size]} inline-flex items-center gap-1`}>
      <span>{action}</span>
    </span>
  );
}
