import os

from sentence_transformers import SentenceTransformer

from ANN_Offline import build_index, save_template_map
from encode_templates import encode_templates, get_templates_from_file
from lasr import LASRRetriever

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    model = SentenceTransformer(MODEL_NAME)
    templates, labels = get_templates_from_file("data/example_temp.txt")
    embeddings, labels = encode_templates(templates, labels, model)
    index_folder = "index"
    os.makedirs(index_folder, exist_ok=True)
    index = build_index(embeddings, labels)
    import faiss

    faiss.write_index(index, f"{index_folder}/reflection_templates.index")
    save_template_map()

    retriever = LASRRetriever(embeddings, index=index, k=min(15, len(templates)), n_components=3)
    demo = "Two cyclists 84 km apart ride toward each other at 18 km/h and 24 km/h. When do they meet?"
    q = model.encode(demo)
    result = retriever.retrieve(q)
    print(f"\nLASR demo query: {demo}")
    print(f"Top template id={int(result.indices[0])}  score={result.scores[0]:.4f}  "
          f"energy={result.energy_retained:.3f}")
    print("Run `python evaluate.py` for MRR / nDCG / ANN latency numbers.")


if __name__ == "__main__":
    main()
