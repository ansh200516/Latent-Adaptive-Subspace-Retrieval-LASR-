import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def save_template_map():
    with open('data/example_temp.txt', 'r') as f:
        content = f.read()

    templates_raw = content.split('---')
    first_template_content = templates_raw[0]
    start_index = first_template_content.find("###")
    if start_index != -1:
        templates_raw[0] = first_template_content[start_index:]

    templates = [t.strip() for t in templates_raw if t.strip()]

    id_to_template = {i: template for i, template in enumerate(templates)}

    output_path = "data/id_to_template.json"
    with open(output_path, "w") as f:
        json.dump(id_to_template, f, indent=4)

    print(f"Mapping from ID to template saved successfully in {output_path}.")

def build_index(embeddings, labels):
    M = 3 
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(embeddings)
    print("Faiss index created.")
    print(f"Index is trained: {index.is_trained}")
    index.add(embeddings)
    print(f"Total vectors in index: {index.ntotal}")
    return index

def load_index():
    print("Loading pre-built index and resources...")
    index = faiss.read_index("index/reflection_templates.index")
    with open("data/id_to_template.json", "r") as f:
        id_to_template = {int(k): v for k, v in json.load(f).items()}
    print("Ready to receive queries.")
    return index, id_to_template

def find_k_candidates(index, query, model,id_to_template,k=5):
    query_embedding = model.encode(query)
    query_embedding=np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances,indices=index.search(query_embedding,k)
    results=[]
    print(f"\nSearching for '{query}':")
    print("-" * 30)
    for i in range(k):
        retrieved_id=indices[0][i]
        similarity_score=distances[0][i]
        retrieved_template=id_to_template[retrieved_id]
        results.append({
            'id':retrieved_id,
            'similarity_score':similarity_score,
            'template':retrieved_template
        })
        print(f"Rank {i+1}: (ID: {retrieved_id}, Score: {similarity_score:.4f})")
        print(f"   L> {retrieved_template}")
    return results

