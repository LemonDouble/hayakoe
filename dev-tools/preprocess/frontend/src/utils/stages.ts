import { t } from "../i18n";

export const STAGE_ORDER = ["extract", "separate", "vad", "classify", "transcribe", "review"];

const KNOWN_STAGES = new Set(STAGE_ORDER);

export function isKnownStage(stage: string): boolean {
  return KNOWN_STAGES.has(stage);
}

export function stageLabelOf(stage: string): string {
  return isKnownStage(stage) ? t(`detail.stages.${stage}`) : stage;
}

export function stageDescKeys(stage: string): { titleKey: string; descKey: string } {
  return {
    titleKey: `detail.stage_desc.${stage}.title`,
    descKey: `detail.stage_desc.${stage}.desc`,
  };
}

// 대시보드 목록 배지는 파이프라인 단계 외에 classifying/done/empty 상태도 표시한다
const DASHBOARD_STAGES = new Set([...STAGE_ORDER, "classifying", "done", "empty"]);

export function dashboardStageLabel(stage: string): string {
  if (stage.startsWith("processing:")) {
    return t("dashboard.stages.processing", { stage: stage.split(":")[1] });
  }
  return DASHBOARD_STAGES.has(stage) ? t(`dashboard.stages.${stage}`) : stage;
}
