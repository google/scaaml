// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeMathJax from 'rehype-mathjax';

// https://astro.build/config
export default defineConfig({
  site: 'https://google.github.io/scaaml/',
  base: '/scaaml',

  // Configure `remark-math` and `rehype-mathjax` plugins:
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathJax],
  },

  integrations: [
    starlight({
      title: 'SCAAML documentation',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/google/scaaml',
        }
      ],
      // Custom CSS to style MathJax equations
      customCss: ['./src/mathjax.css'],
      sidebar: [
        {
          label: 'Start Here',
          items: [
            // Each item here is one entry in the navigation menu.
            { label: 'Getting Started', slug: 'start_here/intro' },
            { label: 'Installation', slug: 'start_here/install' },
          ],
        },
        {
          label: 'Papers and Presentations',
          items: [
            { label: 'SCAAML (DEF CON 27, 2019)', slug: 'papers/scaaml_defcon_2019' },
            { label: 'SCALD (DEF CON 28 & Black Hat USA, 2020)', slug: 'papers/scald_defcon_2020' },
            { label: 'GPAM (CHES, 2024)', slug: 'papers/gpam_ches_2024' },
          ],
        },
        {
          label: 'Guides',
          items: [
            {
              label: 'Capture',
              items: [
                { label: 'Capture Overview', slug: 'guides/capture/overview' },
                { label: 'Attack Point Iterators', slug: 'guides/capture/attack_point_iterators' },
                { label: 'Capture Resume', slug: 'guides/capture/capture_resume' },
                { label: 'Capture Context-managers', slug: 'guides/capture/capture_context_managers' },
                { label: 'Oscilloscopes', slug: 'guides/capture/oscilloscopes' },
                { label: 'Saving a Dataset', slug: 'guides/capture/saving' },
                { label: 'Statistical Tools', slug: 'guides/capture/statistical_tools' },
              ],
            },
            {
              label: 'Deep learning',
              items: [
                { label: 'Overview of Components', slug: 'guides/deep_learning/overview' },
                { label: 'Deep Learning Models', slug: 'guides/deep_learning/deep_learning_models' },
                { label: 'Metrics', slug: 'guides/deep_learning/metrics' },
                { label: 'Intermediates of Algorithms', slug: 'guides/deep_learning/intermediates' },
                { label: 'Dataset Storage', slug: 'guides/deep_learning/dataset_storage' },
              ],
            },
          ],
        },
      ],
    }),
  ],
});
