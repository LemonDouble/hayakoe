import { getLang, setLang, LANG_LABELS } from "../i18n";

export default function LangSelector() {
  return (
    <div className="fixed top-4 right-4 z-50">
      <select
        value={getLang()}
        onChange={(e) => setLang(e.target.value)}
        className="text-xs px-2 py-1 rounded border border-line bg-surface text-fg-muted hover:text-fg cursor-pointer outline-none focus:border-primary"
      >
        {Object.entries(LANG_LABELS).map(([code, label]) => (
          <option key={code} value={code} className="bg-surface text-fg">
            {label}
          </option>
        ))}
      </select>
    </div>
  );
}
