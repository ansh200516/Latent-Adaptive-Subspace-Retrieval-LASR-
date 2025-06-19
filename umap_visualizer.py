import umap
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def umap_embed(embeddings):
    reducer = umap.UMAP(n_components=3, random_state=42)
    return reducer.fit_transform(embeddings)

def draw_umap(embeddings, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    u = fit.fit_transform(embeddings)
    
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=labels)
    elif n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=labels)
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=labels, s=20)
        for i, txt in enumerate(labels):
            ax.text(u[i,0], u[i,1], u[i,2], f' {txt}', size=9, zorder=1, color='k')
    
    plt.title(title, fontsize=18)
    
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, f'umap_plot_{n_components}d.png'))
    plt.close()

# if __name__ == '__main__':
#     embeddings = np.load('embeddings/thought_templates_embeddings.npy')
#     labels = np.load('embeddings/thought_templates_labels.npy')
#     draw_umap(embeddings, labels, n_components=3, title='UMAP projection')
    # draw_umap(embeddings, n_components=2, title='2D UMAP projection')
    # draw_umap(embeddings, n_components=1, title='1D UMAP projection')