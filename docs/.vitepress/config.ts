import { defineConfig } from 'vitepress'

const koNav = [
  { text: '퀵스타트', link: '/quickstart/' },
  { text: '자체 화자 학습', link: '/training/' },
  { text: '서버로 배포', link: '/deploy/' },
  { text: '깊이 읽기', link: '/deep-dive/' },
  { text: 'FAQ', link: '/faq/' },
  {
    text: '링크',
    items: [
      { text: 'GitHub', link: 'https://github.com/LemonDouble/hayakoe' },
      { text: 'PyPI', link: 'https://pypi.org/project/hayakoe/' },
      { text: 'HuggingFace', link: 'https://huggingface.co/lemondouble/hayakoe' },
    ],
  },
]

const jaNav = [
  { text: 'クイックスタート', link: '/ja/quickstart/' },
  { text: '話者学習', link: '/ja/training/' },
  { text: 'サーバーデプロイ', link: '/ja/deploy/' },
  { text: '詳細解説', link: '/ja/deep-dive/' },
  { text: 'FAQ', link: '/ja/faq/' },
  {
    text: 'リンク',
    items: [
      { text: 'GitHub', link: 'https://github.com/LemonDouble/hayakoe' },
      { text: 'PyPI', link: 'https://pypi.org/project/hayakoe/' },
      { text: 'HuggingFace', link: 'https://huggingface.co/lemondouble/hayakoe' },
    ],
  },
]

const zhNav = [
  { text: '快速开始', link: '/zh/quickstart/' },
  { text: '话者训练', link: '/zh/training/' },
  { text: '服务器部署', link: '/zh/deploy/' },
  { text: '深入了解', link: '/zh/deep-dive/' },
  { text: 'FAQ', link: '/zh/faq/' },
  {
    text: '链接',
    items: [
      { text: 'GitHub', link: 'https://github.com/LemonDouble/hayakoe' },
      { text: 'PyPI', link: 'https://pypi.org/project/hayakoe/' },
      { text: 'HuggingFace', link: 'https://huggingface.co/lemondouble/hayakoe' },
    ],
  },
]

const enNav = [
  { text: 'Quickstart', link: '/en/quickstart/' },
  { text: 'Speaker Training', link: '/en/training/' },
  { text: 'Deploy', link: '/en/deploy/' },
  { text: 'Deep Dive', link: '/en/deep-dive/' },
  { text: 'FAQ', link: '/en/faq/' },
  {
    text: 'Links',
    items: [
      { text: 'GitHub', link: 'https://github.com/LemonDouble/hayakoe' },
      { text: 'PyPI', link: 'https://pypi.org/project/hayakoe/' },
      { text: 'HuggingFace', link: 'https://huggingface.co/lemondouble/hayakoe' },
    ],
  },
]

const koSidebar = {
  '/quickstart/': [
    {
      text: '퀵스타트',
      items: [
        { text: '시작하기', link: '/quickstart/' },
        { text: '설치 — CPU vs GPU', link: '/quickstart/install' },
        { text: '첫 음성 만들기', link: '/quickstart/first-voice' },
        { text: '속도·운율 조절', link: '/quickstart/parameters' },
        { text: '커스텀 단어 등록', link: '/quickstart/custom-words' },
        { text: '문장 단위 스트리밍', link: '/quickstart/streaming' },
        { text: '내 머신에서 벤치마크', link: '/quickstart/benchmark' },
      ],
    },
  ],
  '/training/': [
    {
      text: '자체 화자 학습',
      items: [
        { text: '전체 흐름', link: '/training/' },
        { text: '① 데이터 준비', link: '/training/data-prep' },
        { text: '② 전처리 & 학습', link: '/training/training' },
        { text: '③ 품질 리포트', link: '/training/quality-check' },
        { text: '④ 배포 (HF·S3·로컬)', link: '/training/publish' },
        { text: '트러블슈팅', link: '/training/troubleshooting' },
      ],
    },
  ],
  '/deploy/': [
    {
      text: '서버로 배포',
      items: [
        { text: '개요', link: '/deploy/' },
        { text: 'FastAPI 통합', link: '/deploy/fastapi' },
        { text: 'Docker 이미지', link: '/deploy/docker' },
        { text: '백엔드 선택 (CPU vs GPU)', link: '/deploy/backend' },
      ],
    },
  ],
  '/deep-dive/': [
    {
      text: '깊이 읽기',
      items: [
        { text: '왜 만들었나', link: '/deep-dive/' },
        { text: '용어 정리', link: '/deep-dive/glossary' },
        { text: '아키텍처 한눈에', link: '/deep-dive/architecture' },
        { text: 'ONNX 최적화 / 양자화', link: '/deep-dive/onnx-optimization' },
        { text: '문장 경계 pause — Duration Predictor', link: '/deep-dive/duration-predictor' },
        { text: 'BERT GPU 유지 & 배치 추론', link: '/deep-dive/bert-gpu' },
        { text: 'Source 추상화 (HF·S3·로컬)', link: '/deep-dive/source-abstraction' },
        { text: 'OpenJTalk 사전 번들링', link: '/deep-dive/openjtalk-dict' },
        { text: 'arm64 지원', link: '/deep-dive/arm64' },
        { text: '이슈 제보 & 라이선스', link: '/deep-dive/contributing' },
      ],
    },
  ],
  '/faq/': [
    { text: 'FAQ', items: [{ text: '전체', link: '/faq/' }] },
  ],
}

const jaSidebar = {
  '/ja/quickstart/': [
    {
      text: 'クイックスタート',
      items: [
        { text: 'はじめに', link: '/ja/quickstart/' },
        { text: 'インストール — CPU vs GPU', link: '/ja/quickstart/install' },
        { text: '初めての音声生成', link: '/ja/quickstart/first-voice' },
        { text: '速度・韻律の調整', link: '/ja/quickstart/parameters' },
        { text: 'カスタム単語登録', link: '/ja/quickstart/custom-words' },
        { text: '文単位ストリーミング', link: '/ja/quickstart/streaming' },
        { text: 'ベンチマーク', link: '/ja/quickstart/benchmark' },
      ],
    },
  ],
  '/ja/training/': [
    {
      text: '話者学習',
      items: [
        { text: '全体の流れ', link: '/ja/training/' },
        { text: '① データ準備', link: '/ja/training/data-prep' },
        { text: '② 前処理 & 学習', link: '/ja/training/training' },
        { text: '③ 品質レポート', link: '/ja/training/quality-check' },
        { text: '④ 配布 (HF·S3·ローカル)', link: '/ja/training/publish' },
        { text: 'トラブルシューティング', link: '/ja/training/troubleshooting' },
      ],
    },
  ],
  '/ja/deploy/': [
    {
      text: 'サーバーデプロイ',
      items: [
        { text: '概要', link: '/ja/deploy/' },
        { text: 'FastAPI 統合', link: '/ja/deploy/fastapi' },
        { text: 'Docker イメージ', link: '/ja/deploy/docker' },
        { text: 'バックエンド選択 (CPU vs GPU)', link: '/ja/deploy/backend' },
      ],
    },
  ],
  '/ja/deep-dive/': [
    {
      text: '詳細解説',
      items: [
        { text: 'なぜ作ったか', link: '/ja/deep-dive/' },
        { text: '用語集', link: '/ja/deep-dive/glossary' },
        { text: 'アーキテクチャ概観', link: '/ja/deep-dive/architecture' },
        { text: 'ONNX 最適化 / 量子化', link: '/ja/deep-dive/onnx-optimization' },
        { text: '文境界 pause — Duration Predictor', link: '/ja/deep-dive/duration-predictor' },
        { text: 'BERT GPU 常駐 & バッチ推論', link: '/ja/deep-dive/bert-gpu' },
        { text: 'Source 抽象化 (HF·S3·ローカル)', link: '/ja/deep-dive/source-abstraction' },
        { text: 'OpenJTalk 辞書バンドル', link: '/ja/deep-dive/openjtalk-dict' },
        { text: 'arm64 対応', link: '/ja/deep-dive/arm64' },
        { text: 'Issue 報告 & ライセンス', link: '/ja/deep-dive/contributing' },
      ],
    },
  ],
  '/ja/faq/': [
    { text: 'FAQ', items: [{ text: 'すべて', link: '/ja/faq/' }] },
  ],
}

const zhSidebar = {
  '/zh/quickstart/': [
    {
      text: '快速开始',
      items: [
        { text: '入门', link: '/zh/quickstart/' },
        { text: '安装 — CPU vs GPU', link: '/zh/quickstart/install' },
        { text: '第一个语音', link: '/zh/quickstart/first-voice' },
        { text: '速度·韵律调节', link: '/zh/quickstart/parameters' },
        { text: '自定义词汇注册', link: '/zh/quickstart/custom-words' },
        { text: '逐句流式生成', link: '/zh/quickstart/streaming' },
        { text: '基准测试', link: '/zh/quickstart/benchmark' },
      ],
    },
  ],
  '/zh/training/': [
    {
      text: '话者训练',
      items: [
        { text: '整体流程', link: '/zh/training/' },
        { text: '① 数据准备', link: '/zh/training/data-prep' },
        { text: '② 预处理 & 训练', link: '/zh/training/training' },
        { text: '③ 质量报告', link: '/zh/training/quality-check' },
        { text: '④ 发布 (HF·S3·本地)', link: '/zh/training/publish' },
        { text: '故障排除', link: '/zh/training/troubleshooting' },
      ],
    },
  ],
  '/zh/deploy/': [
    {
      text: '服务器部署',
      items: [
        { text: '概述', link: '/zh/deploy/' },
        { text: 'FastAPI 集成', link: '/zh/deploy/fastapi' },
        { text: 'Docker 镜像', link: '/zh/deploy/docker' },
        { text: '后端选择 (CPU vs GPU)', link: '/zh/deploy/backend' },
      ],
    },
  ],
  '/zh/deep-dive/': [
    {
      text: '深入了解',
      items: [
        { text: '为什么要做', link: '/zh/deep-dive/' },
        { text: '术语表', link: '/zh/deep-dive/glossary' },
        { text: '架构概览', link: '/zh/deep-dive/architecture' },
        { text: 'ONNX 优化 / 量化', link: '/zh/deep-dive/onnx-optimization' },
        { text: '句子边界 pause — Duration Predictor', link: '/zh/deep-dive/duration-predictor' },
        { text: 'BERT GPU 常驻 & 批量推理', link: '/zh/deep-dive/bert-gpu' },
        { text: 'Source 抽象化 (HF·S3·本地)', link: '/zh/deep-dive/source-abstraction' },
        { text: 'OpenJTalk 词典捆绑', link: '/zh/deep-dive/openjtalk-dict' },
        { text: 'arm64 支持', link: '/zh/deep-dive/arm64' },
        { text: 'Issue 提报 & 许可证', link: '/zh/deep-dive/contributing' },
      ],
    },
  ],
  '/zh/faq/': [
    { text: 'FAQ', items: [{ text: '全部', link: '/zh/faq/' }] },
  ],
}

const enSidebar = {
  '/en/quickstart/': [
    {
      text: 'Quickstart',
      items: [
        { text: 'Getting Started', link: '/en/quickstart/' },
        { text: 'Install — CPU vs GPU', link: '/en/quickstart/install' },
        { text: 'First Voice', link: '/en/quickstart/first-voice' },
        { text: 'Speed & Prosody', link: '/en/quickstart/parameters' },
        { text: 'Custom Words', link: '/en/quickstart/custom-words' },
        { text: 'Sentence Streaming', link: '/en/quickstart/streaming' },
        { text: 'Benchmark', link: '/en/quickstart/benchmark' },
      ],
    },
  ],
  '/en/training/': [
    {
      text: 'Speaker Training',
      items: [
        { text: 'Overview', link: '/en/training/' },
        { text: '① Data Preparation', link: '/en/training/data-prep' },
        { text: '② Preprocess & Train', link: '/en/training/training' },
        { text: '③ Quality Report', link: '/en/training/quality-check' },
        { text: '④ Publish (HF·S3·Local)', link: '/en/training/publish' },
        { text: 'Troubleshooting', link: '/en/training/troubleshooting' },
      ],
    },
  ],
  '/en/deploy/': [
    {
      text: 'Deploy',
      items: [
        { text: 'Overview', link: '/en/deploy/' },
        { text: 'FastAPI Integration', link: '/en/deploy/fastapi' },
        { text: 'Docker Image', link: '/en/deploy/docker' },
        { text: 'Backend Selection (CPU vs GPU)', link: '/en/deploy/backend' },
      ],
    },
  ],
  '/en/deep-dive/': [
    {
      text: 'Deep Dive',
      items: [
        { text: 'Why We Built This', link: '/en/deep-dive/' },
        { text: 'Glossary', link: '/en/deep-dive/glossary' },
        { text: 'Architecture Overview', link: '/en/deep-dive/architecture' },
        { text: 'ONNX Optimization / Quantization', link: '/en/deep-dive/onnx-optimization' },
        { text: 'Sentence Boundary Pause — Duration Predictor', link: '/en/deep-dive/duration-predictor' },
        { text: 'BERT GPU Residency & Batch Inference', link: '/en/deep-dive/bert-gpu' },
        { text: 'Source Abstraction (HF·S3·Local)', link: '/en/deep-dive/source-abstraction' },
        { text: 'OpenJTalk Dictionary Bundling', link: '/en/deep-dive/openjtalk-dict' },
        { text: 'arm64 Support', link: '/en/deep-dive/arm64' },
        { text: 'Issues & License', link: '/en/deep-dive/contributing' },
      ],
    },
  ],
  '/en/faq/': [
    { text: 'FAQ', items: [{ text: 'All', link: '/en/faq/' }] },
  ],
}

export default defineConfig({
  title: 'HayaKoe',
  description: '빠른 일본어 TTS 라이브러리 — Style-Bert-VITS2 기반, CPU 실시간 추론',

  base: '/hayakoe/',
  cleanUrls: true,
  lastUpdated: true,
  appearance: 'force-dark',

  vite: {
    ssr: {
      noExternal: ['@lemondouble/lemon-vitepress-theme'],
    },
    server: {
      allowedHosts: ['.ngrok-free.app'],
    },
  },

  head: [
    ['link', { rel: 'icon', href: '/hayakoe/favicon.ico', sizes: '48x48' }],
    ['link', { rel: 'icon', href: '/hayakoe/favicon-32.png', type: 'image/png', sizes: '32x32' }],
    ['link', { rel: 'icon', href: '/hayakoe/favicon-192.png', type: 'image/png', sizes: '192x192' }],
    ['link', { rel: 'apple-touch-icon', href: '/hayakoe/apple-touch-icon.png' }],
    ['meta', { name: 'theme-color', content: '#F0B90B' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'HayaKoe' }],
    ['meta', { property: 'og:description', content: '빠른 일본어 TTS 라이브러리' }],
  ],

  locales: {
    root: {
      label: '한국어',
      lang: 'ko-KR',
      themeConfig: {
        nav: koNav,
        sidebar: koSidebar,
        lastUpdated: { text: '마지막 업데이트', formatOptions: { dateStyle: 'medium' } },
        outline: { label: '목차', level: [2, 3] },
        docFooter: { prev: '이전', next: '다음' },
        darkModeSwitchLabel: '테마',
        returnToTopLabel: '맨 위로',
        sidebarMenuLabel: '메뉴',
      },
    },
    ja: {
      label: '日本語',
      lang: 'ja-JP',
      themeConfig: {
        nav: jaNav,
        sidebar: jaSidebar,
        lastUpdated: { text: '最終更新', formatOptions: { dateStyle: 'medium' } },
        outline: { label: '目次', level: [2, 3] },
        docFooter: { prev: '前へ', next: '次へ' },
        darkModeSwitchLabel: 'テーマ',
        returnToTopLabel: 'トップに戻る',
        sidebarMenuLabel: 'メニュー',
      },
    },
    zh: {
      label: '中文',
      lang: 'zh-CN',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        lastUpdated: { text: '最后更新', formatOptions: { dateStyle: 'medium' } },
        outline: { label: '目录', level: [2, 3] },
        docFooter: { prev: '上一页', next: '下一页' },
        darkModeSwitchLabel: '主题',
        returnToTopLabel: '回到顶部',
        sidebarMenuLabel: '菜单',
      },
    },
    en: {
      label: 'English',
      lang: 'en-US',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        lastUpdated: { text: 'Last Updated', formatOptions: { dateStyle: 'medium' } },
        outline: { label: 'On This Page', level: [2, 3] },
        docFooter: { prev: 'Previous', next: 'Next' },
        darkModeSwitchLabel: 'Theme',
        returnToTopLabel: 'Back to top',
        sidebarMenuLabel: 'Menu',
      },
    },
  },

  themeConfig: {
    logo: { src: '/lemon-logo.webp', alt: 'Lemon' },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LemonDouble/hayakoe' },
    ],
  },
})
