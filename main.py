import streamlit as st
header = st.container()
texte = st.container()
with header:
    st.title("HELLO LES AMIS!!!! :duck: :pizza:")

with texte:
    st.subheader("Cela ressemble à ça d'écrire avec streamlit")

    st.write("C'est là où toute la magie va se passer")
    
cool = st.slider("C'est pas cool?",0,100)
