import { users, favorites, type Favorite, type InsertFavorite } from "@shared/schema";
import { db } from "./db";
import { eq } from "drizzle-orm";
import fs from "fs/promises";
import path from "path";
import mime from "mime-types";

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
}

export interface IStorage {
  // Favorites
  getFavorites(): Promise<Favorite[]>;
  addFavorite(favorite: InsertFavorite): Promise<Favorite>;
  removeFavorite(filePath: string): Promise<void>;
  
  // File System
  getFileTree(): Promise<FileNode[]>;
  getFileContent(filePath: string): Promise<{ content: string; type: string }>;
}

export class MemStorage implements IStorage {
  private dataDir = path.join(process.cwd(), "server/data");

  async getFavorites(): Promise<Favorite[]> {
    return await db.select().from(favorites);
  }

  async addFavorite(favorite: InsertFavorite): Promise<Favorite> {
    const [existing] = await db
      .select()
      .from(favorites)
      .where(eq(favorites.filePath, favorite.filePath));

    if (existing) return existing;

    const [newFavorite] = await db
      .insert(favorites)
      .values(favorite)
      .returning();
    return newFavorite;
  }

  async removeFavorite(filePath: string): Promise<void> {
    await db.delete(favorites).where(eq(favorites.filePath, filePath));
  }

  async getFileTree(): Promise<FileNode[]> {
    try {
      return await this.scanDirectory(this.dataDir, "");
    } catch (error) {
      console.error("Error scanning directory:", error);
      return [];
    }
  }

  private async scanDirectory(absolutePath: string, relativePath: string): Promise<FileNode[]> {
    const entries = await fs.readdir(absolutePath, { withFileTypes: true });
    const nodes: FileNode[] = [];

    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue; // Skip hidden files

      const entryRelativePath = path.join(relativePath, entry.name);
      const entryAbsolutePath = path.join(absolutePath, entry.name);

      if (entry.isDirectory()) {
        const children = await this.scanDirectory(entryAbsolutePath, entryRelativePath);
        nodes.push({
          name: entry.name,
          path: entryRelativePath,
          type: 'directory',
          children: children.sort((a, b) => {
            if (a.type === b.type) return a.name.localeCompare(b.name);
            return a.type === 'directory' ? -1 : 1;
          })
        });
      } else {
        nodes.push({
          name: entry.name,
          path: entryRelativePath,
          type: 'file'
        });
      }
    }

    return nodes.sort((a, b) => {
      if (a.type === b.type) return a.name.localeCompare(b.name);
      return a.type === 'directory' ? -1 : 1;
    });
  }

  async getFileContent(filePath: string): Promise<{ content: string; type: string }> {
    // Prevent directory traversal
    const cleanPath = path.normalize(filePath).replace(/^(\.\.[\/\\])+/, '');
    const absolutePath = path.join(this.dataDir, cleanPath);
    
    // Ensure path is within dataDir
    if (!absolutePath.startsWith(this.dataDir)) {
      throw new Error("Invalid file path");
    }

    try {
      const stats = await fs.stat(absolutePath);
      if (!stats.isFile()) throw new Error("Not a file");

      const content = await fs.readFile(absolutePath, 'utf-8');
      const type = mime.lookup(absolutePath) || 'text/plain';

      return { content, type };
    } catch (error) {
      throw new Error("File not found");
    }
  }
}

export const storage = new MemStorage();
