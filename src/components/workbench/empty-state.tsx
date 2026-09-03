import type { ReactNode } from "react";

export default function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="workbench-empty">
      <div className="workbench-empty__icon" aria-hidden>
        {icon}
      </div>
      <h2>{title}</h2>
      <p>{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
