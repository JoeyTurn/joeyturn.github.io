import { defineCollection, z } from "astro:content";

const blog = defineCollection({
    type: "content",
    schema: z.object({
        title: z.string(),
        pubDate: z.coerce.date().optional(),  
        onlylink: z.boolean().optional(),
        draft: z.boolean().optional(),
        series: z.string().optional(),
        seriesOrder: z.number().optional(),
        description: z.string().optional(),
        frontpagedescription: z.string().optional(),
        links: z.array(z.object({
            label: z.string(),
            url: z.string(),
        })).optional(),
        tagline: z.enum(["research", "notes"]).default("research"),
  }),
});

const projects = defineCollection({
    type: "content",
    schema: z.object({
        title: z.string(),
        pubDate: z.coerce.date().optional(),
        onlylink: z.boolean().optional(),
        draft: z.boolean().optional(),
        series: z.string().optional(),
        seriesOrder: z.number().optional(),
        description: z.string().optional(),
        frontpagedescription: z.string().optional(),
        links: z.array(z.object({
            label: z.string(),
            url: z.string(),
        })).optional(),
        authors: z.array(z.object({
            name: z.string(),
            url: z.string().optional(),
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
        series: z.string().optional(),
        seriesOrder: z.number().optional(),
        frontpagedescription: z.string().optional(),
        description: z.string().optional(),
  }),
});

export const collections = { blog, projects, personal };