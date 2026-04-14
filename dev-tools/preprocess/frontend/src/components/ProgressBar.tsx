interface Props {
  progress: number; // 0~1
  message?: string;
}

export default function ProgressBar({ progress, message }: Props) {
  const hasProgress = progress > 0;
  const pct = Math.round(progress * 100);

  return (
    <div>
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-fg-muted">{message || "처리 중..."}</span>
        {hasProgress && <span className="text-primary font-mono font-semibold">{pct}%</span>}
      </div>
      {hasProgress ? (
        <div className="w-full bg-line rounded-full h-1.5 overflow-hidden">
          <div
            className="progress-fill h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : (
        <div className="w-full bg-line rounded-full h-1.5 overflow-hidden">
          <div className="progress-fill h-1.5 rounded-full animate-indeterminate" />
        </div>
      )}
    </div>
  );
}
