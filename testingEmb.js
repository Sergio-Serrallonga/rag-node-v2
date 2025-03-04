import { ChromaClient } from "chromadb";
import { newMyEmbeddingFunction } from "./getEmbeddingFunction.js";
const client = new ChromaClient();

const newMyEmbeddingQuery = await newMyEmbeddingFunction.queryEmbedding(
  "How much money do the player start with in Monopoly?"
);

// switch `createCollection` to `getOrCreateCollection` to avoid creating a new collection every time
const collection = await client.getOrCreateCollection({
  name: "my_third_collection",
  embeddingFunction: newMyEmbeddingFunction,
});

const results = await collection.query({
  queryTexts: `(5) That the Borrower will during the continuance of this security at all times keep the property hereby assigned, insured in the name of the Bank
against the hazards of fire, earthquake, windstorm, riot and fire arising therefrom respectively and in the case of any motor vehicles, on a
comprehensive motor vehicle, policy and such other risks as the Bank may require to their full insurable value to the satisfaction of the Bank in
such Insurance Oﬃce as the Bank may from time to time direct and on demand to deliver to the Bank all such policies of Insurance and all
receipts and vouchers for the payment of premiums. If default at any time be made by the Borrower in eﬀecting and keeping such Insurance it
shall be lawful for the Bank to insure and keep insured the property and to charge the amount of the premium therefore to the Borrower. The
proceeds of any insurance on the Property shall at the option of the Bank be applied toward the replacement of the Property or toward the
payment of the said promissory note and all amounts charged to the Borrower hereunder.`, // Chroma will embed this for you
  nResults: 5, // how many results to return
});

console.log(results);
