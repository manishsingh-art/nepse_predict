export default function SkeletonCard() {
  return (
    <div className="card p-5 space-y-4 animate-pulse">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <div className="skeleton h-5 w-20 rounded" />
          <div className="skeleton h-3.5 w-36 rounded" />
        </div>
        <div className="skeleton h-6 w-14 rounded-full" />
      </div>
      <div className="grid grid-cols-2 gap-3 pt-1">
        <div className="space-y-1.5">
          <div className="skeleton h-3 w-16 rounded" />
          <div className="skeleton h-5 w-24 rounded" />
        </div>
        <div className="space-y-1.5">
          <div className="skeleton h-3 w-20 rounded" />
          <div className="skeleton h-5 w-16 rounded" />
        </div>
        <div className="space-y-1.5">
          <div className="skeleton h-3 w-14 rounded" />
          <div className="skeleton h-5 w-20 rounded" />
        </div>
        <div className="space-y-1.5">
          <div className="skeleton h-3 w-18 rounded" />
          <div className="skeleton h-5 w-14 rounded" />
        </div>
      </div>
      <div className="skeleton h-8 w-full rounded-lg" />
    </div>
  );
}

export function SkeletonRow() {
  return (
    <tr className="border-b border-[#1e2a45] animate-pulse">
      {Array.from({ length: 6 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="skeleton h-4 rounded" style={{ width: `${60 + Math.random() * 40}%` }} />
        </td>
      ))}
    </tr>
  );
}
