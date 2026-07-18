import { t } from "../i18n";

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? t("dashboard.format_duration.min_sec", { m, s }) : t("dashboard.format_duration.sec", { s });
}
