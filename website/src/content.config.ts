import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/data';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
	docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
