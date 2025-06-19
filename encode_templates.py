from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re

def get_templates_from_file(filepath):
    """
    Reads a file and extracts thought templates and their labels.
    Templates are separated by '---'.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start_str = "## Thought templates:"
    start_index = content.find(start_str)
    if start_index != -1:
        content = content[start_index + len(start_str):]
    
    templates = [t.strip() for t in content.split('\n---\n') if t.strip()]
    
    processed_templates = []
    labels = []
    for t in templates:
        match = re.search(r'### Problem Type (\d+):', t)
        if match:
            labels.append(int(match.group(1)))
        else:
            labels.append(0) # Default label if not found
            
        processed_t = re.sub(r'### Problem Type \d+: ', '', t, count=1)
        processed_templates.append(processed_t.strip())
        
    return processed_templates, labels

def encode_templates(templates, labels,model):
    """
    Function to extract templates, encode them, and save the embeddings and labels.
    """
    if templates:
        print("Encoding templates...")
        embeddings = model.encode(templates)
        print("Encoding complete.")
        print("Shape of embeddings:", embeddings.shape)
        
        output_dir = 'embeddings'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path_embeddings = os.path.join(output_dir, 'thought_templates_embeddings.npy')
        print(f"Saving embeddings to {output_path_embeddings}")
        np.save(output_path_embeddings, embeddings)
        print("Embeddings saved successfully.")
        
        output_path_labels = os.path.join(output_dir, 'thought_templates_labels.npy')
        print(f"Saving labels to {output_path_labels}")
        np.save(output_path_labels, np.array(labels))
        print("Labels saved successfully.")
    else:
        print("No templates found in the file.")
    return embeddings, labels

