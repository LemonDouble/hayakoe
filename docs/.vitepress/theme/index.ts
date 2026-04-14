import type { Theme } from 'vitepress'
import LemonTheme from '@lemondouble/lemon-vitepress-theme'
import SpeakerSample from './components/SpeakerSample.vue'
import SpeakerSampleGroup from './components/SpeakerSampleGroup.vue'
import './style.css'

const theme: Theme = {
  ...LemonTheme,
  enhanceApp(ctx) {
    LemonTheme.enhanceApp?.(ctx)
    ctx.app.component('SpeakerSample', SpeakerSample)
    ctx.app.component('SpeakerSampleGroup', SpeakerSampleGroup)
  },
}

export default theme
