import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
import { ChromaClient } from "chromadb";
import { newMyEmbeddingFunction } from "./getEmbeddingFunction.js";

const client = new ChromaClient();

import fs from "fs/promises";
import path from "path";

const DATA_PATH = "../rag-node-v2/data"; // Adjust path as needed

async function main() {
  const documents = await loadDocuments();
  const chunks = await splitDocuments(documents);

  await addToChroma(chunks);
}

async function loadDocuments() {
  try {
    const files = await fs.readdir(DATA_PATH);
    const pdfFiles = files.filter((file) => file.endsWith(".pdf"));

    const loaders = await Promise.all(
      pdfFiles.map(async (file) => {
        const filePath = path.join(DATA_PATH, file);
        const buffer = await fs.readFile(filePath);
        const blob = new Blob([buffer], { type: "application/pdf" });

        return { loader: new WebPDFLoader(blob), source: filePath };
      })
    );

    // Load documents from each PDF

    const docsArray = await Promise.all(
      loaders.map(async ({ loader, source }) => {
        const docs = await loader.load();
        docs.forEach((doc) => (doc.metadata.source = source)); // Assign source
        return docs;
      })
    );

    return docsArray.flat(); // Flatten in case of multiple docs per PDF
  } catch (error) {
    console.error("Error loading documents:", error);
    return [];
  }
}

async function splitDocuments(documents) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 80,
    isSeparatorRegex: false,
  });

  const chuncks = await textSplitter.splitDocuments(documents);

  return chuncks;
}

function calculateChunkIds(chunks) {
  let lastPageId = null;
  let currentChunkIndex = 0;

  for (const chunk of chunks) {
    const source = chunk.metadata?.source;
    const page = chunk.metadata?.loc.pageNumber;
    const currentPageId = `${source}:${page}`;

    // If the page ID is the same as the last one, increment the index.
    if (currentPageId === lastPageId) {
      currentChunkIndex += 1;
    } else {
      currentChunkIndex = 0;
    }

    // Calculate the chunk ID.
    const chunkId = `${currentPageId}:${currentChunkIndex}`;
    lastPageId = currentPageId;

    // Add it to the chunk's metadata.
    chunk.metadata.id = chunkId;
  }

  return chunks;
}

async function addToChroma(chunks) {
  const chuncksWithIds = calculateChunkIds(chunks);

  const collection = await client.getOrCreateCollection({
    name: "my_third_collection",
    embeddingFunction: newMyEmbeddingFunction,
  });

  const existingItems = await collection.get({ include: [] });

  const existingIds = new Set(existingItems.ids || []);
  console.log(`Number of existing documents in DB: ${existingIds.size}`);

  // Filter out already existing documents
  const newChunks = chuncksWithIds.filter(
    (chunk) => !existingIds.has(chunk.metadata.id)
  );

  if (newChunks.length > 0) {
    console.log(`👉 Adding new documents: ${newChunks.length}`);
    const newChunkIds = newChunks.map((chunk) => chunk.metadata.id);

    await collection.upsert({
      documents: newChunks.map((doc) => doc.pageContent),
      ids: newChunkIds,
    });
  } else {
    console.log("✅ No new documents to add");
  }
}

main();
