import * as transformers from "@xenova/transformers";

class MyEmbeddingFunction {
  constructor() {
    this.modelName = "Xenova/all-mpnet-base-v2";
  }

  async queryEmbedding(text) {
    this.pipeline = await transformers.pipeline(
      "feature-extraction",
      this.modelName
    );

    const result = await this.pipeline(text, {
      pooling: "mean",
      normalize: true,
    });

    return result;
  }

  async generate(texts) {
    this.pipeline = await transformers.pipeline(
      "feature-extraction",
      this.modelName
    );

    return await Promise.all(
      texts.map(async (t) => {
        const singleResult = await this.pipeline(t, {
          pooling: "mean",
          normalize: true,
        });
        return Array.from(singleResult.data);
      })
    );
  }
}

export const newMyEmbeddingFunction = new MyEmbeddingFunction();

/* import { pipeline } from "@huggingface/transformers";
export const getEmbeddingFunction = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
  {
    quantized: false,
  }
);

let result = await getEmbeddingFunction("This is a simple test", {
  pooling: "mean",
  normalize: true,
});

console.log(result); */
