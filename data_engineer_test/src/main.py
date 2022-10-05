import streamlit as st
import pandas as pd

liste_dvds_availables = ["Back To The Future 1", "Back To The Future 2", "Back To The Future 3", "La chèvre", "Harry Potter à l'école des sorciers"]

st.title("Boutique DVDs")

st.write("""
## DVDs disponibles
""")

def panier_prix(panier_dict):

    liste_dvds = []
    prix_bttf = 0
    prix_autres = 0
    compteur = 0
    mod_prix = 1

    for dvd, nb_dvds in panier_dict.items():
        for i in range(nb_dvds):
            liste_dvds.append(dvd)

    for dvd in liste_dvds:
        if "back to the future" in dvd.lower():
            prix_bttf = prix_bttf + 15

        else:
            prix_autres = prix_autres + 20 

    liste_dvds_no_duplicate = list(dict.fromkeys(liste_dvds))
    
    for dvd in liste_dvds_no_duplicate:
        if "back to the future" in dvd.lower():
            compteur = compteur + 1

    if compteur == 2:
        mod_prix = 0.9
    
    if compteur >= 3:
        mod_prix = 0.8

    prix_tot = mod_prix*prix_bttf + prix_autres

    return prix_tot

for dvd in liste_dvds_availables:
    if dvd not in st.session_state:
        st.session_state[dvd] = 0

for dvd in liste_dvds_availables:
    if st.button(dvd):
        st.session_state[dvd] += 1

st.write("""
## Panier
""")

prix = 0
for dvd in liste_dvds_availables:
    st.write(dvd, st.session_state[dvd])

if st.button('Reset panier'):
    for dvd in liste_dvds_availables:
        st.session_state[dvd] = 0

st.write("## Prix à payer")
st.write(panier_prix(st.session_state))
# s = ''

# for i in liste_dvds:
#     s += "- " + i + "\n"

# st.markdown(s)


