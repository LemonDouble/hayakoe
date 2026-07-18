import type { VideoStatus } from "../api/videos";
import { formatDuration } from "../utils/format";
import { t } from "../i18n";

interface Props {
  summary: VideoStatus["summary"];
}

export default function DoneSummary({ summary }: Props) {
  return (
    <div className="bg-success/[0.06] border border-success/25 rounded-xl p-8 text-center">
      <div className="w-14 h-14 rounded-full bg-success/15 border border-success/40 text-success flex items-center justify-center text-xl mx-auto mb-4">
        {"✓"}
      </div>
      <p className="font-display text-xl font-bold text-success mb-2">{t("detail.done.title")}</p>
      <p className="text-fg-muted text-sm mb-6">{t("detail.done.description")}</p>

      {summary && summary.length > 0 && (
        <div className="max-w-sm mx-auto space-y-2 mb-6">
          {summary.map((s) => (
            <div
              key={s.name}
              className="flex justify-between items-center bg-canvas border border-line rounded-lg px-4 py-2.5 text-sm"
            >
              <span className={s.name === "discarded" ? "text-fg-dim" : "text-fg font-semibold"}>
                {s.name === "discarded" ? t("detail.done.discarded") : s.name}
              </span>
              <span className="text-fg-muted font-mono text-xs">
                {t("common.count_duration", { count: s.count, duration: formatDuration(s.total_duration) })}
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="text-fg-dim text-sm">
        {t("detail.done.back_hint")}
      </p>
    </div>
  );
}
