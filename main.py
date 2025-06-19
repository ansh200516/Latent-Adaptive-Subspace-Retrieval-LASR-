from encode_templates import*
from umap_visualizer import*
from ANN_Offline import*
import os

model=SentenceTransformer('all-MiniLM-L6-v2')

def main():
    templates,labels=get_templates_from_file('data/example_temp.txt')
    embeddings, labels=encode_templates(templates,labels,model)
    umap_embeddings=umap_embed(embeddings)
    draw_umap(umap_embeddings,labels, n_components=3, title='UMAP projection')
    index = build_index(embeddings, labels)
    
    index_folder = 'index'
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
    faiss.write_index(index, f"{index_folder}/reflection_templates.index")

    save_template_map()
    # index,id_to_template=load_index()
    # find_k_candidates(index,model,id_to_template,k=5)

if __name__ == '__main__':
    main()