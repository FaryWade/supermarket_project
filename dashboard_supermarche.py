import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('supermarket_sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("Segmentation des Ventes Supermarché")
st.write("Dashboard d'analyse commerciale — Algorithme K-Means")

# ─────────────────────────────
# SIDEBAR — FILTRES
# ─────────────────────────────
st.sidebar.header("Filtres")

branches = st.sidebar.multiselect(
    "Sélectionner une branche",
    options=df['Branch'].unique(),
    default=df['Branch'].unique()
)

product_lines = st.sidebar.multiselect(
    "Sélectionner une ligne de produit",
    options=df['Product line'].unique(),
    default=df['Product line'].unique()
)

n_clusters = st.sidebar.slider("Nombre de clusters (K)", min_value=2, max_value=8, value=3)

# Appliquer les filtres
df_filtered = df[
    (df['Branch'].isin(branches)) &
    (df['Product line'].isin(product_lines))
]

# ─────────────────────────────
# ONGLETS
# ─────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Indicateurs Commerciaux",
    "🔍 Exploration des Ventes",
    "🧩 Segmentation K-Means",
    "💡 Recommandations"
])

# ══════════════════════════
# TAB 1 — INDICATEURS
# ══════════════════════════
with tab1:
    st.header("Indicateurs Commerciaux")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Chiffre d'affaires Sales", f"{df_filtered['Sales'].sum():,.0f} $")
    with col2:
        st.metric("Nombre de transactions", f"{len(df_filtered):,}")
    with col3:
        st.metric("Panier moyen", f"{df_filtered['Sales'].mean():.2f} $")
    with col4:
        st.metric("Quantités vendues", f"{df_filtered['Quantity'].sum():,}")

    st.subheader("Chiffre d'affaires par ligne de produit")
    ca_par_produit = df_filtered.groupby('Product line')['Sales'].sum().sort_values(ascending=False)
    st.bar_chart(ca_par_produit)

    st.subheader("Chiffre d'affaires par branche")
    ca_par_branche = df_filtered.groupby('Branch')['Sales'].sum()
    st.bar_chart(ca_par_branche)

    st.subheader("Évolution des ventes dans le temps")
    ventes_par_jour = df_filtered.groupby('Date')['Sales'].sum()
    st.line_chart(ventes_par_jour)

# ══════════════════════════
# TAB 2 — EXPLORATION
# ══════════════════════════
with tab2:
    st.header("Exploration des Ventes")

    st.subheader("Distribution des montants de vente")
    fig, ax = plt.subplots()
    ax.hist(df_filtered['Sales'], bins=30, color='orange', edgecolor='black')
    ax.set_xlabel("Montant Sales ($)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des totaux de vente")
    st.pyplot(fig)

    st.subheader("Distribution des quantités achetées")
    fig2, ax2 = plt.subplots()
    ax2.hist(df_filtered['Quantity'], bins=10, color='steelblue', edgecolor='black')
    ax2.set_xlabel("Quantité")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution des quantités")
    st.pyplot(fig2)

    st.subheader("Quantités vendues par ligne de produit")
    qty_produit = df_filtered.groupby('Product line')['Quantity'].sum().sort_values(ascending=False)
    st.bar_chart(qty_produit)

    st.subheader("Panier moyen par mode de paiement")
    panier_paiement = df_filtered.groupby('Payment')['Sales'].mean().sort_values(ascending=False)
    st.bar_chart(panier_paiement)

    st.subheader("Panier moyen par type de client")
    panier_client = df_filtered.groupby('Customer type')['Sales'].mean()
    st.bar_chart(panier_client)

# ══════════════════════════
# TAB 3 — SEGMENTATION
# ══════════════════════════
with tab3:
    st.header("Segmentation K-Means")

    # Variables utilisées pour le clustering
    features = ['Unit price', 'Quantity', 'Sales', 'Rating']
    st.write(f"**Variables utilisées :** {', '.join(features)}")

    X = df_filtered[features].dropna()

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Méthode du coude
    st.subheader("Méthode du coude (choix optimal de K)")
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig3, ax3 = plt.subplots()
    ax3.plot(list(K_range), inertias, marker='o', color='orange')
    ax3.set_xlabel("Nombre de clusters K")
    ax3.set_ylabel("Inertie")
    ax3.set_title("Méthode du coude")
    st.pyplot(fig3)

    # Clustering final
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_result = X.copy()
    df_result['Cluster'] = labels

    st.subheader(f"Résultats : {n_clusters} segments identifiés")

    # Profil moyen par cluster
    st.write("**Profil moyen par cluster :**")
    profil = df_result.groupby('Cluster')[features].mean().round(2)
    st.dataframe(profil)

    # Visualisation scatter
    st.subheader("Visualisation des clusters (Sales vs Quantity)")
    fig4, ax4 = plt.subplots()
    colors = ['orange', 'steelblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    for i in range(n_clusters):
        cluster_data = df_result[df_result['Cluster'] == i]
        ax4.scatter(cluster_data['Sales'], cluster_data['Quantity'],
                    label=f"Cluster {i}", color=colors[i], alpha=0.6)
    ax4.set_xlabel("Sales ($)")
    ax4.set_ylabel("Quantité")
    ax4.set_title("Clusters : Sales vs Quantité")
    ax4.legend()
    st.pyplot(fig4)

    # Taille de chaque cluster
    st.subheader("Nombre de transactions par cluster")
    taille_cluster = df_result['Cluster'].value_counts().sort_index()
    st.bar_chart(taille_cluster)

# ══════════════════════════
# TAB 4 — RECOMMANDATIONS
# ══════════════════════════
with tab4:
    st.header("Recommandations Décisionnelles")

    # Recalcul des clusters pour cette page
    X2 = df_filtered[features].dropna()
    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)
    kmeans2 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels2 = kmeans2.fit_predict(X2_scaled)
    df_reco = X2.copy()
    df_reco['Cluster'] = labels2

    profil2 = df_reco.groupby('Cluster')['Sales'].mean()
    Sales_moyen = profil2.mean()

    for c in sorted(df_reco['Cluster'].unique()):
        ca_cluster = profil2[c]
        n_transactions = (df_reco['Cluster'] == c).sum()

        st.subheader(f"Cluster {c} — {n_transactions} transactions | Panier moyen : {ca_cluster:.2f} $")

        if ca_cluster >= Sales_moyen * 1.2:
            st.success("🏆 Segment Premium : clients à fort panier. Recommandation : fidélisation, offres exclusives.")
        elif ca_cluster <= Sales_moyen * 0.8:
            st.error("⚡ Segment à Activer : panier faible. Recommandation : promotions, coupons de réduction.")
        else:
            st.warning("🔄 Segment Intermédiaire : comportement moyen. Recommandation : cross-selling, personnalisation.")

    st.divider()
    st.subheader("Synthèse")
    st.write("""
    La segmentation K-Means permet d'identifier des profils distincts de transactions.
    Chaque cluster correspond à un comportement d'achat différent basé sur le prix unitaire,
    la quantité achetée, le Sales dépensé et la note de satisfaction.

    Actions recommandées :
    - Concentrer les efforts marketing sur les segments à fort potentiel
    - Mettre en place des offres personnalisées par segment
    - Surveiller l'évolution des clusters dans le temps
    - Optimiser le stock selon les produits préférés de chaque segment
    """)

    # Export CSV
    st.divider()
    csv = df_reco.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Télécharger les données segmentées (CSV)",
        data=csv,
        file_name="supermarche_segmente.csv",
        mime="text/csv"
    )
