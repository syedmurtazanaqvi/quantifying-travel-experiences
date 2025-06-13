import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#data findings
data = {
    'City': ['Islamabad', 'Wuhan', 'Chongqing', 'Chengdu', 'Tokyo', 'Osaka', 'Kyoto', 'Kamakura', 'Bangkok', 'Lahore'],
    'Natural Beauty': [9, 6, 7, 8, 6, 7, 9, 8, 7, 8],
    'Food': [9, 9, 9, 10, 10, 9, 10, 8, 10, 10],
    'Cultural Vibe': [8, 8, 9, 9, 9, 9, 10, 8, 9, 10],
    'Affordability': [9, 7, 6, 7, 5, 6, 5, 6, 8, 9],
    'Cleanliness': [7, 8, 6, 7, 9, 9, 9, 9, 6, 5],
    'Hospitality': [9, 8, 8, 8, 7, 8, 8, 9, 8, 9],
    'Safety': [7, 7, 6, 7, 9, 9, 9, 9, 7, 5],
    'Revisit Likelihood': [9, 8, 8, 9, 9, 9, 10, 9, 10, 9]
}

df = pd.DataFrame(data)

# Sidebar weight sliders
st.sidebar.header("Adjust Weights")
categories = ['Natural Beauty', 'Food', 'Cultural Vibe', 'Affordability', 'Cleanliness', 'Hospitality', 'Safety', 'Revisit Likelihood']
weights = {cat: st.sidebar.slider(cat, 0.0, 1.0, 0.1 if cat != 'Food' else 0.15) for cat in categories}
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}  # normalize

# Weighted score calculation
def calculate_weighted_score(df, weights):
    categories = list(weights.keys())
    weight_vector = np.array(list(weights.values()))
    df['Weighted Score'] = df[categories].values @ weight_vector
    return df

df = calculate_weighted_score(df, weights)

st.title("Quantifying Travel Experiences")
st.markdown("This dashboard ranks cities based on personal travel factors. Adjust the sliders to customize weightings.")

# City dropdown and radar chart
city = st.selectbox("Select a city to view its radar chart:", df['City'])

# Radar chart
def plot_radar_chart(city_row):
    values = city_row[categories].tolist()
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"Radar Chart for {city_row['City']}")
    return fig

st.pyplot(plot_radar_chart(df[df['City'] == city].iloc[0]))

# Bubble chart
fig1 = px.scatter(df, x='Affordability', y='Natural Beauty', size='Revisit Likelihood', color='City', hover_name='City', title='Affordability vs Natural Beauty')
st.plotly_chart(fig1)

# PCA and Clustering
X = StandardScaler().fit_transform(df[categories])
pca = PCA(n_components=2)
components = pca.fit_transform(X)
df['PC1'] = components[:, 0]
df['PC2'] = components[:, 1]
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

fig2 = px.scatter(df, x='PC1', y='PC2', color='Cluster', text='City', title='City Clusters Based on Experience Factors')
st.plotly_chart(fig2)

# Dataframe
st.dataframe(df[['City', 'Weighted Score'] + categories])

