import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    pubDate: z.string().optional(),   // keep it minimal
    draft: z.boolean().optional(),
  }),
});

const projects = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    pubDate: z.string().optional(),
    draft: z.boolean().optional(),
  }),
});

export const collections = { blog, projects };
