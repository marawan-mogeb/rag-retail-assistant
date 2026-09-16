"""
Rebuild the Chroma vector store locally.

Use this if the vector store exported from Kaggle fails to load due to a
chromadb version mismatch. Produces an identical store using whatever
chromadb version is installed in this environment.

Run from the backend/ directory:
    python rebuild_vector_store.py
"""
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "data/vector_store"
COLLECTION_NAME = "retail_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Same corpus as the notebook
RETAIL_DOCS = {
    "return_policy_electronics.txt": """Electronics Return Policy: Items in the Electronics category may be returned within 30 days
of purchase with original packaging and proof of purchase. Opened software, opened headphones, and items marked
final sale are not eligible for return. A 15% restocking fee applies to opened electronics larger than $200.
Defective items may be exchanged within 90 days at no charge.""",

    "return_policy_apparel.txt": """Apparel Return Policy: Clothing and footwear may be returned within 60 days of purchase if
unworn, unwashed, and with tags attached. Swimwear and undergarments are final sale for hygiene reasons.
Sale items marked with a red tag are final sale and cannot be returned or exchanged.""",

    "warranty_kitchen_appliances.txt": """Kitchen Appliance Warranty: All kitchen appliances (blenders, toasters, coffee makers)
carry a 1-year manufacturer warranty covering defects in materials and workmanship. Damage from misuse,
water damage, or unauthorized repair voids the warranty. Extended 3-year warranty plans are available at
checkout for an additional fee.""",

    "warranty_electronics.txt": """Electronics Warranty: Laptops, tablets, and phones include a 1-year limited warranty from the
manufacturer. Screen cracks and liquid damage are not covered under standard warranty. AppleCare-style extended
protection plans can be purchased separately within 60 days of the original purchase.""",

    "shipping_policy.txt": """Shipping Policy: Standard shipping takes 5-7 business days and is free on orders over $35.
Expedited shipping (2-3 business days) costs $9.99. Orders placed before 2pm EST ship the same business day.
International shipping is available to select countries and may incur customs fees.""",

    "restocking_policy.txt": """Shelf Restocking Policy: Store shelves are audited twice daily for out-of-stock items.
Products with fewer than 3 units remaining on a shelf are flagged for restocking within 4 hours. Shelves
showing large gaps or empty facings should be reported to the floor manager immediately for priority restocking.""",

    "planogram_guidelines.txt": """Planogram Guidelines: Each shelf section should maintain full-face product presentation,
meaning products are pulled forward to fill gaps as they sell. A shelf is considered understocked if visible
empty space exceeds 20% of the section width. Overstocked shelves that block signage should also be adjusted.""",

    "grocery_category_guide.txt": """Grocery Category Guide: Perishable items (dairy, produce, meat) are restocked daily and
rotated using first-in-first-out (FIFO) principles. Non-perishable dry goods follow a weekly restocking cycle
unless a shelf audit flags low stock earlier.""",

    "loyalty_program.txt": """Loyalty Program: Members earn 1 point per $1 spent, redeemable at 100 points for a $5 reward.
Points expire after 12 months of account inactivity. Birthday month purchases earn double points automatically.""",

    "price_match_policy.txt": """Price Match Policy: We match prices from major competitors on identical, in-stock items
within 14 days of purchase. Clearance, closeout, and marketplace seller listings are excluded from price matching."""
}


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if end >= len(text):
            break
    return chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []
    chunk_counter = 0
    for source, text in RETAIL_DOCS.items():
        for c in chunk_text(text):
            all_chunks.append({"text": c, "source": source, "chunk_id": chunk_counter})
            chunk_counter += 1
    print(f"Created {len(all_chunks)} chunks from {len(RETAIL_DOCS)} documents")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=OUTPUT_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in all_chunks],
        ids=[f"chunk_{c['chunk_id']}" for c in all_chunks],
    )
    print(f"Inserted {collection.count()} chunks into collection '{COLLECTION_NAME}'")

    config = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBED_MODEL_NAME,
        "vector_store": "chromadb",
        "collection_name": COLLECTION_NAME,
        "chromadb_version": chromadb.__version__,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Vector store rebuilt at: {OUTPUT_DIR}")
    print(f"chromadb version used: {chromadb.__version__}")


if __name__ == "__main__":
    main()
