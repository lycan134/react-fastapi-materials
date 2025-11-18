from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementFraction
import sys
import os

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Import neural network class
# ----------------------------
# Define the class directly here to avoid import issues
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Make it available for unpickling
sys.modules['__main__'].NeuralNetwork = NeuralNetwork

# ----------------------------
# Load model and normalization stats
# ----------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "models/best_model_full.pt")
NORM_STATS_PATH = os.path.join(BACKEND_DIR, "models/normalization_stats.pth")

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

stats = torch.load(NORM_STATS_PATH, map_location=device)
X_mean, X_std = stats["X_mean"], stats["X_std"]
y_mean, y_std = stats["y_mean"], stats["y_std"]
EPS = 1e-8

# ----------------------------
# Target elements and constants
# ----------------------------
TARGET_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr'
]
EXTRA_ELEMENTS = ['Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

# 228 spacegroups (exclude 168 and 207)
USED_SPACEGROUPS = [sg for sg in range(1, 231) if sg not in [168, 207]]
N_SPACEGROUPS = len(USED_SPACEGROUPS)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------
# Pydantic input
# ----------------------------
class FormulaInput(BaseModel):
    formula: str

# ----------------------------
# Featurization function
# ----------------------------
def featurize_formula(formula: str) -> pd.DataFrame:
    comp = Composition(formula)
    ef = ElementFraction()
    ef_features = ef.featurize(comp)
    df_base = pd.DataFrame([ef_features], columns=ef.feature_labels())

    # Ensure all target elements
    for el in TARGET_ELEMENTS:
        if el not in df_base.columns:
            df_base[el] = 0.0
    df_base = df_base[TARGET_ELEMENTS]

    # Drop superheavy elements
    df_base = df_base.drop(columns=[el for el in EXTRA_ELEMENTS if el in df_base.columns])

    # Element-based descriptors
    el_objs = [Element(e) for e in comp.elements]
    fractions = np.array([comp.get_atomic_fraction(e) for e in comp.elements])

    def safe_attr(el, attr):
        try:
            val = getattr(el, attr)
            return float(val) if val is not None else np.nan
        except Exception:
            return np.nan

    atomic_masses = np.array([safe_attr(e, "atomic_mass") for e in el_objs], dtype=np.float32)
    en_values = np.array([safe_attr(e, "X") for e in el_objs], dtype=np.float32)
    cov_radii = np.array([safe_attr(e, "atomic_radius_calculated") for e in el_objs], dtype=np.float32)
    ea_values = np.array([safe_attr(e, "electron_affinity") for e in el_objs], dtype=np.float32)

    df_base["n_atoms"] = comp.num_atoms
    df_base["n_elements"] = len(comp.elements)
    df_base["avg_atomic_mass"] = np.nansum(atomic_masses * fractions)
    df_base["en_mean"] = np.nanmean(en_values)
    df_base["en_max"] = np.nanmax(en_values)
    df_base["en_min"] = np.nanmin(en_values)
    df_base["en_range"] = df_base["en_max"] - df_base["en_min"]
    df_base["avg_covalent_radius"] = np.nanmean(cov_radii)
    df_base["ea_mean"] = np.nanmean(ea_values)
    df_base["ea_max"] = np.nanmax(ea_values)
    df_base["ea_min"] = np.nanmin(ea_values)
    df_base["ea_range"] = df_base["ea_max"] - df_base["ea_min"]

    # Vectorized spacegroups
    df_vectorized = pd.concat([df_base]*N_SPACEGROUPS, ignore_index=True)
    spacegroup_cols = [f"spacegroup_{sg}" for sg in USED_SPACEGROUPS]
    df_vectorized[spacegroup_cols] = np.eye(N_SPACEGROUPS, dtype=np.float32)

    # Column order
    final_columns = TARGET_ELEMENTS + [
        "n_atoms", "n_elements", "avg_atomic_mass", "en_mean", "en_max",
        "en_min", "en_range", "avg_covalent_radius", "ea_mean",
        "ea_max", "ea_min", "ea_range"
    ] + spacegroup_cols

    df_final = df_vectorized[final_columns].astype(np.float32)
    return df_final

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_energy(input: FormulaInput):
    try:
        df_features = featurize_formula(input.formula)
        X_tensor = torch.tensor(df_features.values, dtype=torch.float32).to(device)
        X_tensor = (X_tensor - X_mean) / (X_std + EPS)
        with torch.no_grad():
            y_pred = model(X_tensor).cpu()
        y_pred = y_pred * y_std + y_mean
        results = [{"spacegroup": sg, "formation_energy": float(fe)}
                   for sg, fe in zip(USED_SPACEGROUPS, y_pred.flatten())]
        return {"predictions": results}
    except Exception as e:
        return {"error": str(e)}
