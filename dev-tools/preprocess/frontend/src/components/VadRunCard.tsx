import type { VadParams } from "../api/videos";
import { t } from "../i18n";

const VAD_PRESETS: { labelKey: string; descKey: string; params: VadParams }[] = [
  {
    labelKey: "detail.vad.preset.default.label",
    descKey: "detail.vad.preset.default.desc",
    params: { min_segment_sec: 1.0, max_segment_sec: 8.0, threshold: 0.3, min_silence_ms: 50 },
  },
  {
    labelKey: "detail.vad.preset.noisy.label",
    descKey: "detail.vad.preset.noisy.desc",
    params: { min_segment_sec: 1.5, max_segment_sec: 8.0, threshold: 0.65, min_silence_ms: 40 },
  },
  {
    labelKey: "detail.vad.preset.monologue.label",
    descKey: "detail.vad.preset.monologue.desc",
    params: { min_segment_sec: 1.0, max_segment_sec: 12.0, threshold: 0.3, min_silence_ms: 80 },
  },
];

const VAD_FIELDS: {
  key: keyof VadParams;
  step: number;
  min?: number;
  max?: number;
  labelKey: string;
  hintKey: string;
}[] = [
  {
    key: "min_segment_sec",
    step: 0.1,
    labelKey: "detail.vad.min_segment",
    hintKey: "detail.vad.min_segment_hint",
  },
  {
    key: "max_segment_sec",
    step: 0.5,
    labelKey: "detail.vad.max_segment",
    hintKey: "detail.vad.max_segment_hint",
  },
  {
    key: "threshold",
    step: 0.05,
    min: 0,
    max: 1,
    labelKey: "detail.vad.threshold",
    hintKey: "detail.vad.threshold_hint",
  },
  {
    key: "min_silence_ms",
    step: 10,
    labelKey: "detail.vad.min_silence",
    hintKey: "detail.vad.min_silence_hint",
  },
];

interface Props {
  params: VadParams;
  onChange: (params: VadParams) => void;
  onRun: () => void;
}

export default function VadRunCard({ params, onChange, onRun }: Props) {
  return (
    <div className="bg-surface border border-line rounded-xl p-6">
      {/* 단계 설명 */}
      <div className="text-center mb-6">
        <p className="text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-2 font-display">NEXT STEP</p>
        <p className="font-display text-xl font-bold text-fg mb-2">
          {t("detail.stage_desc.vad.title")}
        </p>
        <p className="text-fg-muted text-sm max-w-lg mx-auto leading-relaxed">
          {t("detail.stage_desc.vad.desc")}
        </p>
      </div>

      {/* 프리셋 */}
      <div className="max-w-lg mx-auto mb-6">
        <p className="text-fg text-sm font-semibold mb-2">{t("detail.vad.quick_settings")}</p>
        <div className="grid grid-cols-3 gap-2">
          {VAD_PRESETS.map((preset) => (
            <button
              key={preset.labelKey}
              className="bg-canvas border border-line hover:border-primary/40 rounded-lg px-3 py-2.5 text-left transition-colors"
              onClick={() => onChange({ ...preset.params })}
            >
              <p className="text-fg text-xs font-semibold">{t(preset.labelKey)}</p>
              <p className="text-fg-dim text-[10px] mt-0.5">{t(preset.descKey)}</p>
            </button>
          ))}
        </div>
      </div>

      {/* 세부 파라미터 */}
      <div className="space-y-4 mb-6 max-w-lg mx-auto text-sm">
        {VAD_FIELDS.map((f) => (
          <label key={f.key} className="block text-fg-muted">
            <div className="flex items-center justify-between">
              <span>{t(f.labelKey)}</span>
              <input
                type="number"
                step={f.step}
                min={f.min}
                max={f.max}
                className="w-24 bg-canvas border border-line rounded-lg px-3 py-1.5 text-fg text-right focus:outline-none focus:border-primary/50 transition-colors"
                value={params[f.key]}
                onChange={(e) => onChange({ ...params, [f.key]: +e.target.value })}
              />
            </div>
            <p className="text-fg-dim text-xs mt-1">
              {t(f.hintKey)}
            </p>
          </label>
        ))}
      </div>

      {/* 상황별 팁 */}
      <details className="max-w-lg mx-auto mb-6">
        <summary className="text-fg-muted text-xs cursor-pointer hover:text-fg transition-colors">
          {t("detail.vad.tips_toggle")}
        </summary>
        <div className="bg-canvas border border-line rounded-lg p-4 mt-2 text-xs text-fg-muted space-y-1.5 leading-relaxed">
          <p>
            <span className="text-primary font-semibold">{t("detail.vad.tip_fast_dialogue_title")}</span>
            {t("detail.vad.tip_fast_dialogue_desc")}
          </p>
          <p>
            <span className="text-primary font-semibold">{t("detail.vad.tip_noisy_title")}</span>
            {t("detail.vad.tip_noisy_desc")}
          </p>
          <p>
            <span className="text-primary font-semibold">{t("detail.vad.tip_monologue_title")}</span>
            {t("detail.vad.tip_monologue_desc")}
          </p>
          <p>
            <span className="text-primary font-semibold">{t("detail.vad.tip_interjection_title")}</span>
            {t("detail.vad.tip_interjection_desc")}
          </p>
        </div>
      </details>

      <div className="text-center">
        <button
          className="bg-primary hover:bg-primary-hover text-canvas px-8 py-2.5 rounded-lg font-semibold text-sm transition-colors"
          onClick={onRun}
        >
          {t("detail.vad.run")}
        </button>
        <p className="text-fg-dim text-xs mt-3">
          {t("detail.run_after_hint")}
        </p>
      </div>
    </div>
  );
}
