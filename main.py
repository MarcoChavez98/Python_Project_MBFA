import streamlit as st
header = st.container()
texte = st.container()
with header:
    st.title("HELLO LES AMIS!!!! :duck:")

with texte:
    st.subheader("Cela ressemble à ça d'écrire avec streamlit")

    st.write("C'est là où toute la magie va se passer")