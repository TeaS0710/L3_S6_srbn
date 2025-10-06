import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import matplotlib.cm as cm


def build_labels_from_clusters(tokens, clusters):
    """
    Construit :
      - labels : array(N) où labels[i] = cluster_id du token i
      - cluster_sizes : dict {cid: nb_membres}
      - centroids_idx : dict {cid: index_token_centroide}
    """
    token_to_cid = {}
    cluster_sizes = {}
    centroids_idx = {}

    for cid_str, cdata in clusters.items():
        cid = int(cid_str)
        members = cdata["members"]
        cluster_sizes[cid] = len(members)
        centroid_token = cdata["centroid"]

        for mem in members:
            token_to_cid[mem] = cid

        if centroid_token in tokens:
            idx_centroid = tokens.index(centroid_token)
            centroids_idx[cid] = idx_centroid
        else:
            centroids_idx[cid] = -1

    labels = []
    for t in tokens:
        cid = token_to_cid.get(t, -1)
        labels.append(cid)

    return np.array(labels), cluster_sizes, centroids_idx


def parse_pos_from_token(token):
    """
    Si token = "maison_NOUN", renvoie "NOUN", sinon None.
    """
    if "_" in token:
        return token.split("_")[-1]
    return None


def pos_to_marker(pos):
    """
    Associe une forme de marqueur à chaque POS.
    Personnalisez selon vos besoins.
    """
    marker_map = {
        "NOUN": "o",
        "PROPN": "D",
        "VERB": "s",
        "ADJ": "^",
        "ADV": "v"
    }
    return marker_map.get(pos, 'x')  # par défaut 'x'


def plot_mds_2d(ax, tokens, sim_mat, clusters, title):
    """
    - Convertit la matrice de similarité en distance => MDS => scatter 2D.
    - Couleur = cluster, Forme = POS, Centroïdes en plus gros.
    - Dessine le tout sur l'axe ax.
    """
    # Style "seaborn-v0_8" pour la grille, etc.
    plt.style.use('seaborn-v0_8')

    # 1) Build labels
    labels, cluster_sizes, centroids_idx = build_labels_from_clusters(tokens, clusters)

    # 2) Similarité -> distance
    dist_mat = 1.0 - sim_mat

    # 3) MDS 2D
    mds_2d = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_2d = mds_2d.fit_transform(dist_mat)

    # 4) Préparation du scatter plot

    cmap = cm.get_cmap("tab20")

    # 5) Tracé des points
    for i, token in enumerate(tokens):
        cid = labels[i]
        color = "gray" if cid < 0 else cmap(cid % 20)

        # Forme selon la POS
        pos_tag = parse_pos_from_token(token)
        marker = pos_to_marker(pos_tag)

        x, y = pos_2d[i, 0], pos_2d[i, 1]

        # Vérifier si c'est un centroïde
        is_centroid = any(idx == i for idx in centroids_idx.values())
        if is_centroid:
            c_found = [c for c, c_idx in centroids_idx.items() if c_idx == i]
            if c_found:
                c_id = c_found[0]
                size = 20 + cluster_sizes[c_id] * 2
            else:
                size = 30

            # Marqueur 'x' => pas d'edgecolor
            if marker == 'x':
                ax.scatter(x, y, c=[color], marker=marker, s=size, alpha=0.8, linewidth=0.7)
            else:
                # Marqueurs pleins => edgecolor noir pour que le centroïde se détache un peu
                ax.scatter(x, y, c=[color], marker=marker, s=size, edgecolors='black', alpha=0.8, linewidth=0.7)

        else:
            # Points non-centroïdes, plus petits encore
            if marker == 'x':
                ax.scatter(x, y, c=[color], marker=marker, s=15, alpha=0.7, linewidth=0.7)
            else:
                ax.scatter(x, y, c=[color], marker=marker, s=15, alpha=0.7, edgecolors='none')

    ax.set_title(title, fontsize=10)
    ax.grid(True)

    # ==> Pas de légendes pour alléger.
