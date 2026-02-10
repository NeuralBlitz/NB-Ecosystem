import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import fs from "fs";
import path from "path";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // File Tree
  app.get(api.files.tree.path, async (req, res) => {
    try {
      const tree = await storage.getFileTree();
      res.json(tree);
    } catch (error) {
      res.status(500).json({ message: "Failed to load file tree" });
    }
  });

  // File Content
  app.get(api.files.content.path, async (req, res) => {
    try {
      const { path } = api.files.content.input.parse(req.query);
      const data = await storage.getFileContent(path);
      res.json({ ...data, path });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid path" });
      } else {
        res.status(404).json({ message: "File not found" });
      }
    }
  });

  // Favorites
  app.get(api.favorites.list.path, async (req, res) => {
    const favorites = await storage.getFavorites();
    res.json(favorites);
  });

  app.post(api.favorites.toggle.path, async (req, res) => {
    try {
      const input = api.favorites.toggle.input.parse(req.body);
      
      const favorites = await storage.getFavorites();
      const existing = favorites.find(f => f.filePath === input.filePath);

      if (existing) {
        await storage.removeFavorite(input.filePath);
        res.json({ added: false });
      } else {
        await storage.addFavorite(input);
        res.json({ added: true });
      }
    } catch (error) {
      res.status(400).json({ message: "Invalid request" });
    }
  });

  return httpServer;
}
