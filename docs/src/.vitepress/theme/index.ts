// .vitepress/theme/index.ts
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'

import { 
  NolebaseEnhancedReadabilitiesMenu, 
  NolebaseEnhancedReadabilitiesScreenMenu, 
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
import AuthorBadge from '@/AuthorBadge.vue'
import Authors from '@/Authors.vue'
import Landing from '@/Landing.vue'
import TutorialGallery from '@/TutorialGallery.vue'
import LatteNav from '@/LatteNav.vue'
import LatteFooter from '@/LatteFooter.vue'
import LatteAwning from '@/LatteAwning.vue'
import Benchmarks from '@/Benchmarks.vue'
import Validation from '@/Validation.vue'
import PaperCite from '@/PaperCite.vue'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css' // You could setup your own, or else a default will be copied.
import './docstrings.css' // You could setup your own, or else a default will be copied.

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // Latte branding bookends — applied on every doc-layout page so
      // landing, tutorials/index, and the deep doc pages share one nav
      // and footer. Pages using layout: false (Landing.vue,
      // tutorials/index.md) bring their own LatteNav + LatteFooter and
      // these slot injections don't render for them.
      'layout-top': () => h(LatteNav),
      'layout-bottom': () => h(LatteFooter),
      'nav-bar-content-after': () => [
        h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
      ],
      // A enhanced readabilities menu for narrower screens (usually smaller than iPad Mini)
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
    app.component('AuthorBadge', AuthorBadge)
    app.component('Authors', Authors)
    app.component('Landing', Landing)
    app.component('TutorialGallery', TutorialGallery)
    app.component('LatteNav', LatteNav)
    app.component('LatteFooter', LatteFooter)
    app.component('LatteAwning', LatteAwning)
    app.component('Benchmarks', Benchmarks)
    app.component('Validation', Validation)
    app.component('PaperCite', PaperCite)
  }
}
export default Theme