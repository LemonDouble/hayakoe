interface Props {
  progress: number; // 0~1
  message?: string;
}

export default function ProgressBar({ progress, message }: Props) {
  const hasProgress = progress > 0;
  const pct = Math.round(progress * 100);

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{message || "처리 중..."}</span>
        {hasProgress && <span>{pct}%</span>}
      </div>
      {hasProgress ? (
        <div className="w-full bg-slate-700 rounded h-3">
          <div
            className="bg-blue-500 h-3 rounded transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : (
        <div className="w-full bg-slate-700 rounded h-3 overflow-hidden">
          <div className="h-3 rounded bg-blue-500/60 animate-indeterminate" />
        </div>
      )}
    </div>
  );
}
