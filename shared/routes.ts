import { z } from 'zod';
import { insertFavoriteSchema, favorites } from './schema';

export const api = {
  // File System Operations
  files: {
    tree: {
      method: 'GET' as const,
      path: '/api/files/tree',
      responses: {
        200: z.array(z.custom<any>()), // Recursive type difficult in Zod, using any for tree node
      },
    },
    content: {
      method: 'GET' as const,
      path: '/api/files/content',
      input: z.object({
        path: z.string(),
      }),
      responses: {
        200: z.object({
          content: z.string(),
          type: z.string(),
          path: z.string(),
        }),
        404: z.object({ message: z.string() }),
        400: z.object({ message: z.string() }),
      },
    },
  },
  // Favorites Operations
  favorites: {
    list: {
      method: 'GET' as const,
      path: '/api/favorites',
      responses: {
        200: z.array(z.custom<typeof favorites.$inferSelect>()),
      },
    },
    toggle: {
      method: 'POST' as const,
      path: '/api/favorites',
      input: insertFavoriteSchema,
      responses: {
        200: z.object({ added: z.boolean() }),
      },
    },
  },
};
