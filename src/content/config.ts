import { defineCollection, z } from "astro:content";

const blog = defineCollection({
    type: "content",
    schema: z.object({
        title: z.string(),
        pubDate: z.coerce.date().optional(),
        onlylink: z.boolean().optional(),
        draft: z.boolean().optional(),
        series: z.string().optional(),            // e.g., "Diffusion 101"
        seriesOrder: z.number().optional(),
        description: z.string().optional(),
  }),
});

const projects = defineCollection({
    type: "content",
    schema: z.object({
        title: z.string(),
        pubDate: z.coerce.date().optional(),
        onlylink: z.boolean().optional(),
        draft: z.boolean().optional(),
        series: z.string().optional(),            // e.g., "Diffusion 101"
        seriesOrder: z.number().optional(),
        description: z.string().optional(),
        links: z.array(z.object({
            label: z.string(),                // e.g., "arXiv", "GitHub", "Site"
            url: z.string(),                  // we’ll normalize to https:// if needed
        })).optional(),
        authors: z.array(z.object({
            name: z.string(),
            url: z.string().optional(),  // allow missing urls too
        })).optional(),
  }),
});

const personal = defineCollection({
    type: "content",
    schema: z.object({
        title: z.string(),
        pubDate: z.coerce.date().optional(),
        onlylink: z.boolean().optional(),
        draft: z.boolean().optional(),
        series: z.string().optional(),            // e.g., "Diffusion 101"
        seriesOrder: z.number().optional(),
        description: z.string().optional(),
  }),
});

export const collections = { blog, projects, personal };