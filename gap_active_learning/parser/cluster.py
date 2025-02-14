import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from dscribe.descriptors import SOAP
from ase.io import read
import pandas as pd

from gap_active_learning.al.similarity import *

def build_SOAP_vector(traj,
                      species=[],
                      soap_info = {
                                   'rcut':5,
                                   'nmax':8,
                                   'lmax':4,
                                   'sigma': 1.0,
                                   'average': 'inner',
                                  }
                      ):
    
    # Setting up the SOAP descriptor	
    soap = SOAP(
                species=species,
                periodic=True,
                r_cut=soap_info['rcut'],
                n_max=soap_info['nmax'],
                l_max=soap_info['lmax'],
                sigma=soap_info['sigma'],
                average=soap_info['average'],
                )
    
    return soap.create(traj)


def kpca_kmeans(list_of_atoms, kmeans_clusters=10, kernel="poly"):
    """
    Perform Kernel PCA and K-means clustering on SOAP descriptors.
    """
    if len(list_of_atoms) <= kmeans_clusters:
        print("Total number of atoms are smaller than the number of KMeans clusters!")
        return list_of_atoms
    else:
        species = []
        if type(list_of_atoms) == list:
            for t in list_of_atoms:
                elements = np.unique(t.get_chemical_symbols())
                for e in elements:
                    if e not in species:
                        species.append(e)
        elif type(list_of_atoms) == ase.atoms.Atoms:
            elements = np.unique(list_of_atoms.get_chemical_symbols())
            for e in elements:
                if e not in species:
                    species.append(e)

        # Generate SOAP descriptors
        soap_vectors = build_SOAP_vector(list_of_atoms, species=species)

        # Kernel PCA
        kernel_pca = KernelPCA(n_components=2, kernel=kernel)
        kpca_vectors = kernel_pca.fit_transform(soap_vectors)
    
        # K-means clustering
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=1)
        labels = kmeans.fit_predict(kpca_vectors)

        FE_present = all('FE' in atoms.info for atoms in list_of_atoms)

        atoms_dict = {}
        for i, atoms in enumerate(list_of_atoms):
            atoms_dict[i] = {"atoms": atoms, "label": labels[i]}
            if FE_present:
                atoms_dict[i]["FE"] = atoms.info['FE']

        df = pd.DataFrame.from_dict(atoms_dict, orient="index")
        df["label"] = df["label"].astype(int)

        if FE_present:
            df["FE"] = pd.to_numeric(df["FE"], errors="coerce")
            # Sample representative points
            representatives = [df.loc[df.label == g].FE.idxmin() for g in np.unique(labels)] 

        else:
            cluster_centers = kmeans.cluster_centers_
            representatives = []
            for g in np.unique(labels):
                cluster_points = df[df.label == g].index.to_list()
                cluster_vectors = kpca_vectors[cluster_points]

                # Find the point closest to the cluster center (Euclidean distance)
                centroid = cluster_centers[g]
                closest_idx = np.argmin(np.linalg.norm(cluster_vectors - centroid, axis=1))
                representatives.append(cluster_points[closest_idx])

        return [list_of_atoms[i] for i in representatives]


def plot_pca_clusters(kpca_vectors, labels, representatives):
    """
    Plots the Kernel PCA projection of SOAP descriptors with KMeans clusters.

    Parameters:
    - kpca_vectors: np.ndarray, PCA-reduced feature vectors (2D).
    - labels: np.ndarray, Cluster labels from KMeans.
    - representatives: list, Indices of selected representative structures.
    """
    plt.figure(figsize=(8, 6))

    # Get unique cluster labels and colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Generate distinct colors

    # Plot all points with default size, color-coded by cluster
    for cluster, color in zip(unique_labels, colors):
        cluster_mask = labels == cluster
        plt.scatter(kpca_vectors[cluster_mask, 0], kpca_vectors[cluster_mask, 1], 
                    color=color, alpha=0.6, edgecolors='k', label=f"Cluster {cluster}")

    # Compute cluster centroids (mean position of each cluster in PCA space)
    cluster_centers_pca = np.array([kpca_vectors[labels == g].mean(axis=0) for g in unique_labels])

    # Plot cluster centroids with "X" markers
    plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                c='black', marker='X', s=150, label="Centroids", edgecolors="black")

    # Plot representatives with the same color as their cluster, making them slightly larger
    for rep_idx, cluster in zip(representatives, labels[representatives]):
        plt.scatter(kpca_vectors[rep_idx, 0], kpca_vectors[rep_idx, 1], 
                    color=colors[cluster], edgecolors='black', marker='D', 
                    s=100, label=f"Rep {cluster}" if f"Rep {cluster}" not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot settings
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Kernel PCA Projection with KMeans Clustering")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Workflow ---
if __name__ == "__main__":
    # Step 1: Read structures
    structures = read("original.xyz", ":")

    # Step 2: Perform Kernel PCA and K-means clustering
    kmeans_clusters = 10

    # Re-run Kernel PCA & KMeans to get additional variables (without modifying function behavior)
    species = list({e for atoms in structures for e in atoms.get_chemical_symbols()})
    soap_vectors = build_SOAP_vector(structures, species=species)

    kernel_pca = KernelPCA(n_components=2, kernel="poly")
    kpca_vectors = kernel_pca.fit_transform(soap_vectors)

    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=1)
    labels = kmeans.fit_predict(kpca_vectors)

    # Identify representatives based on clustering (same logic as kpca_kmeans)
    df = pd.DataFrame({"label": labels})
    representatives = [df[df.label == g].index[0] for g in np.unique(labels)]  # Select first in cluster

    # Extract representative structures
    representative_atoms = [structures[i] for i in representatives]

    # Step 3: Plot PCA clusters
    plot_pca_clusters(kpca_vectors, labels, representatives)

    # Step 4: Save representative structures
    ase.io.write("representatives.xyz", representative_atoms)
    print(f"Saved {len(representative_atoms)} representative structures to 'representatives.xyz'")


