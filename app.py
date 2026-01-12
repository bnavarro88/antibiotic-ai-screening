import streamlit as st
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np

# --- CONFIGURACIÓN DE LA RED NEURONAL ---
class CerebroHibrido(nn.Module):
    def __init__(self):
        super(CerebroHibrido, self).__init__()
        self.fc1 = nn.Linear(1025, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x_quimica, x_biologia):
        x = torch.cat((x_quimica, x_biologia), dim=0)
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

# --- FUNCIONES DE APOYO ---
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    # Esta es la forma universal compatible con todas las versiones de RDKit
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return torch.tensor(list(fp), dtype=torch.float)

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="AI Antibiotic Predictor", layout="wide")
st.title("🔬 AI-Driven Antibiotic Screening Tool")
st.markdown("Basado en el modelo de Red Neuronal Híbrida para la crisis de resistencia.")

# Cargar Modelo
@st.cache_resource
def load_model():
    model = CerebroHibrido()
    model.load_state_dict(torch.load('modelo_antibioticos_final.pth'))
    model.eval()
    return model

modelo = load_model()

# Entradas del Usuario
col1, col2 = st.columns(2)

with col1:
    st.subheader("Configuración Química")
    smiles_input = st.text_input("SMILES de la molécula", "CCCC...") # Ej: Amoxicilina
    tipo_gram = st.radio("Tipo de Bacteria", ("Gram Positiva", "Gram Negativa"))

with col2:
    st.subheader("Visualización Molecular")
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            img = Draw.MolToImage(mol)
            st.image(img)

# Predicción
if st.button("🚀 Ejecutar Screening"):
    fp = smiles_to_fp(smiles_input)
    if fp is not None:
        biologia = torch.tensor([1.0] if tipo_gram == "Gram Positiva" else [0.0])
        with torch.no_grad():
            prob = modelo(fp, biologia).item()
        
        st.divider()
        st.metric(label="Probabilidad de Efectividad", value=f"{prob:.2%}")
        
        if prob > 0.7:
            st.success("Candidato Prometedor: Alta probabilidad de inhibición.")
        elif prob > 0.4:
            st.warning("Eficacia Moderada: Requiere mayor investigación.")
        else:

            st.error("Baja Probabilidad: Es probable que la bacteria sea resistente.")
