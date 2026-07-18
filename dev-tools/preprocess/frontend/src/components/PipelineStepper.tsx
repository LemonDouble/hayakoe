import { Fragment } from "react";
import { STAGE_ORDER, stageLabelOf } from "../utils/stages";
import { t } from "../i18n";

interface Props {
  stage: string;
  currentIdx: number;
  onRollback: (stage: string) => void;
}

export default function PipelineStepper({ stage, currentIdx, onRollback }: Props) {
  const allStages = [...STAGE_ORDER, "done"];

  return (
    <div className="mb-2">
      <div className="flex items-start">
        {allStages.map((s, i) => {
          const isDone = s === "done";
          const stageIdx = isDone ? STAGE_ORDER.length : i;
          const isCompleted = isDone ? stage === "done" : stageIdx < currentIdx;
          const isCurrent = stageIdx === currentIdx && stage !== "done";
          const isClickable = !isDone && isCompleted;
          const stageLabel = isDone ? t("detail.stages.done") : stageLabelOf(s);

          return (
            <Fragment key={s}>
              {i > 0 && (
                <div
                  className={`flex-1 h-0.5 mt-4 transition-colors ${
                    stageIdx <= currentIdx || (isDone && stage === "done")
                      ? "bg-primary"
                      : "bg-line"
                  }`}
                />
              )}
              <div
                className={`flex flex-col items-center ${isClickable ? "cursor-pointer group" : ""}`}
                onClick={() => isClickable && onRollback(s)}
                title={isClickable ? t("detail.rollback_title", { stage: stageLabel }) : ""}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${
                    isCompleted
                      ? "bg-primary text-canvas group-hover:bg-primary-hover"
                      : isCurrent
                        ? "bg-surface border-2 border-primary text-primary"
                        : "bg-surface-2 border border-line text-fg-dim"
                  }`}
                >
                  {isCompleted ? "✓" : isDone ? "" : i + 1}
                </div>
                <span
                  className={`text-[10px] mt-1.5 whitespace-nowrap transition-colors font-semibold ${
                    isCompleted
                      ? "text-primary group-hover:text-primary-hover"
                      : isCurrent
                        ? "text-primary"
                        : "text-fg-dim"
                  }`}
                >
                  {stageLabel}
                </span>
              </div>
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}
