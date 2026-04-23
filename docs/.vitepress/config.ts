import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'ko-KR',
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

  themeConfig: {
    logo: { src: '/lemon-logo.webp', alt: 'Lemon' },

    nav: [
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
    ],

    sidebar: {
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
        {
          text: 'FAQ',
          items: [
            { text: '전체', link: '/faq/' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LemonDouble/hayakoe' },
    ],

    lastUpdated: {
      text: '마지막 업데이트',
      formatOptions: { dateStyle: 'medium' },
    },

    outline: {
      label: '목차',
      level: [2, 3],
    },

    docFooter: {
      prev: '이전',
      next: '다음',
    },

    darkModeSwitchLabel: '테마',
    returnToTopLabel: '맨 위로',
    sidebarMenuLabel: '메뉴',
  },
})
