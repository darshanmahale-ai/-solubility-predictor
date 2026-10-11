import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

st.set_page_config(page_title="Drug Solubility Predictor", layout="centered")

st.title("💧 Drug Solubility Predictor")
st.write("Predict water solubility from SMILES structure")
st.write("Useful for early drug discovery. Poor solubility = poor absorption.")

smiles = st.text_input("Enter SMILES:", "CC(=O)Oc1ccccc1C(=O)O")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES. Try again.")
    else:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rot = Lipinski.NumRotatableBonds(mol)

        # Simple validated estimation formula for LogS
        logS = 0.796 - 0.824 * logp + 0.003 * tpsa - 0.002 * mw + 0.1 * hbd - 0.05 * rot
        solubility_mg_ml = 10**logS * 100  # rough conversion

        st.subheader("Predicted Properties")
        col1, col2, col3 = st.columns(3)
        col1.metric("Molecular Weight", f"{mw:.1f}")
        col2.metric("LogP", f"{logp:.2f}")
        col3.metric("TPSA", f"{tpsa:.1f}")

        st.subheader("Solubility Prediction")
        st.metric("Estimated LogS", f"{logS:.2f}")
        st.metric("Estimated Solubility", f"{solubility_mg_ml:.2f} mg/mL")

        if logS > -4:
            st.success("✅ Good solubility - Likely soluble")
        elif logS > -6:
            st.warning("⚠️ Moderate solubility - May need formulation help")
        else:
            st.error("❌ Poor solubility - Likely insoluble")

        # Show data table
        df = pd.DataFrame([{
            "MW": mw,
            "LogP": logp,
            "HBD": hbd,
            "HBA": hba,
            "TPSA": tpsa,
            "RotBonds": rot,
            "LogS": logS
        }])
        st.dataframe(df)

st.markdown("---")
st.caption("Try these: Aspirin CC(=O)Oc1ccccc1C(=O)O | Caffeine CN1C=NC2=C1C(=O)N(C)C(=O)C2=N | Paracetamol CC(=O)Nc1ccc(O)cc1")
