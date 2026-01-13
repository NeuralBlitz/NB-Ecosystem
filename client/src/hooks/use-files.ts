import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type InsertFavorite } from "@shared/routes"; // Assuming schema export matches

// Re-exporting types for frontend use
export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
}

export interface FileContent {
  content: string;
  type: string;
  path: string;
}

export function useFileTree() {
  return useQuery({
    queryKey: [api.files.tree.path],
    queryFn: async () => {
      const res = await fetch(api.files.tree.path);
      if (!res.ok) throw new Error("Failed to fetch file tree");
      // The API returns any[] because of recursive type difficulty, so we cast it
      return (await res.json()) as FileNode[];
    },
  });
}

export function useFileContent(path: string | null) {
  return useQuery({
    queryKey: [api.files.content.path, path],
    queryFn: async () => {
      if (!path) return null;
      // Construct URL with query param manually since buildUrl helper might be basic
      const url = `${api.files.content.path}?path=${encodeURIComponent(path)}`;
      const res = await fetch(url);
      
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("Failed to fetch content");
      
      return api.files.content.responses[200].parse(await res.json());
    },
    enabled: !!path, // Only run if path is provided
  });
}

export function useFavorites() {
  return useQuery({
    queryKey: [api.favorites.list.path],
    queryFn: async () => {
      const res = await fetch(api.favorites.list.path);
      if (!res.ok) throw new Error("Failed to fetch favorites");
      return api.favorites.list.responses[200].parse(await res.json());
    },
  });
}

export function useToggleFavorite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: InsertFavorite) => {
      const res = await fetch(api.favorites.toggle.path, {
        method: api.favorites.toggle.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to toggle favorite");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [api.favorites.list.path] });
    },
  });
}
