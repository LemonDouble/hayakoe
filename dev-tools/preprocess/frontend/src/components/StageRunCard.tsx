import { isKnownStage, stageDescKeys, stageLabelOf } from "../utils/stages";
import { t } from "../i18n";

interface Props {
  stage: string;
  pendingStage: string | null;
  stageError: string;
  onRun: (stage: string) => void;
}

export default function StageRunCard({ stage, pendingStage, stageError, onRun }: Props) {
  return (
    <div className="bg-surface border border-line rounded-xl p-6">
      <div className="text-center mb-5">
        <p className="text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-2 font-display">NEXT STEP</p>
        <p className="font-display text-xl font-bold text-fg mb-2">
          {isKnownStage(stage) ? t(stageDescKeys(stage).titleKey) : stage}
        </p>
        <p className="text-fg-muted text-sm max-w-md mx-auto leading-relaxed">
          {isKnownStage(stage) ? t(stageDescKeys(stage).descKey) : ""}
        </p>
      </div>
      {pendingStage && (
        <div className="bg-warning/[0.06] border border-warning/25 rounded-lg p-4 mb-4 text-center">
          <p className="text-warning text-sm font-semibold mb-1">{t("detail.pending.title")}</p>
          <p className="text-fg-muted text-xs">{t("detail.pending.desc")}</p>
        </div>
      )}
      {stageError && !pendingStage && (
        <div className="bg-error/[0.06] border border-error/25 rounded-lg p-3 mb-4 text-error text-sm text-center">
          {stageError}
        </div>
      )}
      <div className="text-center">
        <button
          className={`px-8 py-2.5 rounded-lg font-semibold text-sm transition-colors ${
            pendingStage
              ? "bg-warning/20 text-warning cursor-wait animate-pulse"
              : "bg-primary hover:bg-primary-hover text-canvas"
          }`}
          onClick={() => !pendingStage && onRun(stage)}
          disabled={!!pendingStage}
        >
          {pendingStage
            ? t("detail.pending.waiting")
            : t("detail.run_stage", { stage: stageLabelOf(stage) })}
        </button>
      </div>
    </div>
  );
}
