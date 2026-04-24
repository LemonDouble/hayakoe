import ko from "./locales/ko.json";
import ja from "./locales/ja.json";
import zh from "./locales/zh.json";
import en from "./locales/en.json";

const LOCALES: Record<string, Record<string, string>> = { ko, ja, zh, en };
const STORAGE_KEY = "hayakoe-lang";
const SUPPORTED = Object.keys(LOCALES);

function detectLang(): string {
  // 1. URL param
  const url = new URL(window.location.href);
  const paramLang = url.searchParams.get("lang");
  if (paramLang && SUPPORTED.includes(paramLang)) return paramLang;

  // 2. localStorage
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && SUPPORTED.includes(stored)) return stored;

  // 3. navigator.language
  const nav = navigator.language.split("-")[0];
  if (SUPPORTED.includes(nav)) return nav;

  // 4. fallback
  return "en";
}

let currentLang = detectLang();

export function getLang(): string {
  return currentLang;
}

export const LANG_LABELS: Record<string, string> = {
  ko: "한국어",
  ja: "日本語",
  zh: "中文",
  en: "English",
};

export function setLang(lang: string): void {
  localStorage.setItem(STORAGE_KEY, lang);
  fetch("/api/lang", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lang }),
  }).catch(() => {});
  window.location.reload();
}

export function t(key: string, params?: Record<string, string | number>): string {
  const locale = LOCALES[currentLang] || LOCALES.en;
  let str = locale[key] ?? LOCALES.en[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      str = str.replace(new RegExp(`\\{${k}\\}`, "g"), String(v));
    }
  }
  return str;
}
