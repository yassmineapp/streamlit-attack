# -*- coding: utf-8 -*-
"""
Prédiction des attaques classiques en cyber — tableau de bord Streamlit

Cette application reprend la logique du notebook « atelier_10_cyberattaques_corrige » :
  1. charger le CSV et supprimer les doublons / traiter les valeurs manquantes ;
  2. préparer X (variables) et y (type d'attaque) ;
  3. découper en train 70 % / validation 15 % / test 15 % (stratifié) ;
  4. entraîner plusieurs modèles (Régression logistique, Random Forest, XGBoost) ;
  5. choisir le meilleur sur la validation (F1 macro) puis l'évaluer sur le test ;
  6. permettre la prédiction d'un trafic (manuelle ou par fichier CSV).

Lancement :  streamlit run app.py
Connexion  :  identifiant « admin »  /  mot de passe « yassmine »
"""

import os
import io
import hmac
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# 0. PARAMÈTRES GÉNÉRAUX
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Prédiction des attaques en cyber", page_icon="🛡️", layout="wide")

AUTH_ACTIVE = True           # True = réactive la page de connexion au démarrage
UTILISATEUR = "admin"          # identifiant de connexion
MOT_DE_PASSE = "yassmine"      # mot de passe de connexion
RANDOM_STATE = 42
DOSSIER = os.path.dirname(os.path.abspath(__file__))
CHEMIN_CSV = os.path.join(DOSSIER, "dataset.csv")
CHEMIN_ENTETE = os.path.join(DOSSIER, "entete.png")   # bannière de l'en-tête

# Colonnes non utilisées comme variables explicatives (comme dans le notebook)
COLONNES_EXCLUES = ["attack_type", "is_attack", "timestamp_utc", "src_ip", "dst_ip", "src_port"]

# Description des 10 attaques (connaissances générales de cybersécurité)
ATTAQUES = {
    "DDoS": ("🌊", "Saturation d'un service par un très grand nombre de requêtes ou de paquets."),
    "Phishing": ("🎣", "Courriels frauduleux avec liens piégés pour voler des identifiants."),
    "SQL_Injection": ("💉", "Injection de code SQL dans une requête pour lire ou modifier la base de données."),
    "Brute_Force": ("🔨", "Essais répétés de mots de passe pour forcer l'accès à un compte."),
    "XSS": ("📜", "Injection de scripts malveillants dans une page web vue par d'autres utilisateurs."),
    "Malware": ("🦠", "Logiciel malveillant exécuté sur une machine (processus et fichiers suspects)."),
    "Ransomware": ("🔒", "Chiffrement massif de fichiers pour exiger une rançon."),
    "Port_Scanning": ("📡", "Exploration de nombreux ports pour repérer les services ouverts."),
    "Man_in_the_Middle": ("🕵️", "Interception du trafic entre deux machines (ex. usurpation ARP)."),
    "Botnet": ("🤖", "Machines infectées contrôlées à distance (communications régulières vers un serveur C2)."),
    "Normal": ("✅", "Trafic légitime (disponible uniquement si des données normales sont chargées)."),
}

# Palette de couleurs : une couleur fixe par classe
PALETTE = ["#2563eb", "#7c3aed", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444",
           "#ec4899", "#14b8a6", "#8b5cf6", "#f97316", "#64748b"]


MARQUE = "Yassmine NAAOUMI"
SLOGAN = "Prédiction des attaques"
LOGO_SVG = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 380 92'> <defs> <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#050f24'/><stop offset='.55' stop-color='#0a2352'/><stop offset='1' stop-color='#061431'/></linearGradient> <linearGradient id='tx' x1='0' y1='0' x2='1' y2='0'><stop offset='0' stop-color='#8fdcff'/><stop offset='1' stop-color='#3b9cff'/></linearGradient> <linearGradient id='hd' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#123a7a'/><stop offset='1' stop-color='#0a2352'/></linearGradient> </defs> <rect width='380' height='92' rx='10' fill='url(#bg)'/> <g stroke='#35d0ff' stroke-width='1' fill='none' opacity='.55'><path d='M0 20 H40'/><path d='M0 72 H34'/><path d='M380 20 H350 L340 30 H320'/><path d='M380 74 H356 L346 64 H328'/></g> <g transform='translate(12 6) scale(.8)'> <path d='M 22 92 C 22 80 14 72 12 58 C 8 40 18 14 48 10 C 72 7 88 24 86 42 C 87 46 90 52 93 58 L 85 60 C 86 66 84 70 80 72 L 80 80 C 80 84 74 86 66 88 L 66 92 Z' fill='url(#hd)' stroke='#35d0ff' stroke-width='2.4' stroke-linejoin='round'/> <g stroke='#35d0ff' stroke-width='1.6' fill='none' stroke-linecap='round'> <path d='M30 44 H42 L48 34 H60 L66 26'/><path d='M34 58 H46 L52 48 H70 L76 40'/><path d='M40 30 L46 22 H58'/><path d='M52 62 H64 L70 54'/><path d='M26 52 L32 44'/> </g> <g fill='#8fdcff'><circle cx='30' cy='44' r='2.6'/><circle cx='60' cy='34' r='2.6'/><circle cx='66' cy='26' r='2.6'/><circle cx='70' cy='48' r='2.6'/><circle cx='76' cy='40' r='2.6'/><circle cx='58' cy='22' r='2.6'/><circle cx='64' cy='62' r='2.6'/><circle cx='70' cy='54' r='2.6'/><circle cx='34' cy='58' r='2.6'/></g> </g> <text x='108' y='46' font-family='Segoe UI,Arial,Helvetica,sans-serif' font-size='31' font-weight='800' font-style='italic'><tspan fill='#ffffff'>Attack </tspan><tspan fill='url(#tx)'>Predict</tspan></text> <path d='M110 62 H128' stroke='#3b9cff' stroke-width='1' opacity='.8'/><path d='M300 62 H318' stroke='#3b9cff' stroke-width='1' opacity='.8'/> <text x='214' y='66' text-anchor='middle' font-family='Segoe UI,Arial,Helvetica,sans-serif' font-size='10.5' fill='#c3d4ee' font-style='italic'>Prédiction des Attaques</text> </svg>"


def fr_nombre(n):
    """2845317 -> « 2 845 317 »."""
    return f"{int(n):,}".replace(",", " ")


def fr_pct(x, d=2):
    """0.9841 -> « 98,41 % »."""
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def joli(nom):
    """Affiche « Man_in_the_Middle » sous la forme « Man in the Middle »."""
    return str(nom).replace("_", " ")


# ----------------------------------------------------------------------------
# 1. STYLE (CSS) ET LOGO
# ----------------------------------------------------------------------------
CSS = """
<style>
.stApp { background: #f4f7fc; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }
h1, h2, h3 { color: #0f2a5c; letter-spacing: -0.3px; }

/* Barre latérale */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1f4d 0%, #123a8f 100%); }
[data-testid="stSidebar"] * { color: #e8efff !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.05); border-radius: 10px; padding: 8px 12px;
    margin-bottom: 4px; transition: all .15s;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.16); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #2563eb; box-shadow: 0 4px 12px rgba(37,99,235,.5);
}

[data-testid="stSidebar"] button, [data-testid="stSidebar"] button * { color: #0f2a5c !important; font-weight: 700; }

/* Cartes */
.carte { background: #fff; border-radius: 16px; padding: 16px 18px; box-shadow: 0 2px 14px rgba(15,42,92,.08);
         border: 1px solid #e6edf8; height: 100%; min-height: 118px; }
.kpi-titre { color: #64748b; font-size: .82rem; font-weight: 600; }
.kpi-valeur { color: #0f2a5c; font-size: 1.9rem; font-weight: 800; line-height: 1.2; }
.kpi-sous { color: #64748b; font-size: .75rem; }
.attaque-nom { font-weight: 800; color: #0f2a5c; font-size: 1.05rem; }
.attaque-desc { color: #475569; font-size: .85rem; min-height: 2.6rem; }
.badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700; }
.badge-att { background:#fee2e2; color:#b91c1c; } .badge-ok { background:#dcfce7; color:#15803d; }
.resultat { border-radius: 16px; padding: 18px 22px; border: 2px solid; }
.resultat-att { background:#fff1f1; border-color:#ef4444; } .resultat-ok { background:#effcf3; border-color:#22c55e; }
.titre-page { display:flex; align-items:center; gap:12px; margin-bottom: .4rem; }
.sous-titre { color:#64748b; margin-top:-6px; margin-bottom: 14px; }
div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg,#2563eb,#1d4ed8); border: 0; color: white; font-weight: 700; border-radius: 10px; }

/* ===== Version compacte, proche de la maquette ===== */
.block-container { padding: .7rem 1.1rem 1rem; max-width: 1700px; }
[data-testid="stVerticalBlock"] { gap: .55rem; }
[data-testid="stHorizontalBlock"] { gap: .6rem; }
[data-testid="stSidebar"] { width: 215px !important; min-width: 215px !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { padding: 6px 10px; font-size: .85rem; }
.page-titre { font-size: 1.25rem; font-weight: 800; color: #0f2a5c; margin: 0; }
.barre { display: grid; grid-template-columns: 1fr auto; align-items: center; background: #fff; border-radius: 14px;
         padding: 8px 16px; box-shadow: 0 2px 10px rgba(15,42,92,.07); border: 1px solid #e3ebf7; margin-bottom: .4rem; }
.b-g { display: flex; align-items: center; gap: 10px; }
.marque { font-weight: 800; color: #1e3a8a; font-size: 1.05rem; line-height: 1.1; }
.slogan { font-size: .6rem; color: #64748b; letter-spacing: .4px; }
.b-ico { width: 44px; height: 44px; border-radius: 12px; background: #eaf1ff; display: flex; align-items: center; justify-content: center; margin-right: 12px; }
.b-titre { font-size: 1.25rem; font-weight: 800; color: #0f2a5c; line-height: 1.2; }
.b-sous { font-size: .7rem; color: #64748b; }
.b-d { display: flex; justify-content: flex-end; align-items: center; gap: 14px; font-size: 1rem; }
.profil { font-size: .78rem; background: #f1f5fb; border-radius: 999px; padding: 4px 12px; color: #334155; font-weight: 600; }
[class*="st-key-carte_"] { background: #fff; border-radius: 14px; padding: 10px 14px 8px; border: 1px solid #e3ebf7;
                           box-shadow: 0 2px 10px rgba(15,42,92,.07); }
.ct { font-weight: 700; color: #0f2a5c; font-size: .88rem; margin-bottom: 2px; }
.kpi { box-sizing: border-box; margin-bottom: 4px; display: flex; gap: 10px; align-items: center; background: #fff; border-radius: 14px; padding: 10px 12px; height: 78px;
       box-shadow: 0 2px 10px rgba(15,42,92,.07); border: 1px solid #e3ebf7; overflow: hidden; }
.kpi-ic { min-width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; }
.kpi-t { white-space: nowrap; font-size: .7rem; color: #64748b; font-weight: 600; }
.kpi-v { white-space: nowrap; font-size: 1.2rem; font-weight: 800; color: #0f2a5c; line-height: 1.15; }
.kpi-s { font-size: .62rem; color: #64748b; line-height: 1.2; }
.pl b { white-space: nowrap; margin-left: 6px; }
.pl { display: flex; justify-content: space-between; align-items: center; font-size: .78rem; color: #334155; padding: 6px 0; border-bottom: 1px solid #eef2f8; }
.pl i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
.lg { display: flex; justify-content: space-between; font-size: .72rem; color: #334155; padding: 2.5px 0; }
.lg i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.tinc { width: 100%; border-collapse: collapse; font-size: .72rem; }
.tinc th { text-align: left; color: #64748b; font-weight: 700; padding: 3px 6px; }
.tinc td { padding: 4px 6px; border-top: 1px solid #eef2f8; color: #1e293b; }
.badge-info { background: #dbeafe; color: #1d4ed8; }
[data-testid="stWidgetLabel"] p { font-size: .72rem; font-weight: 600; color: #334155; margin-bottom: 0; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { min-height: 34px; font-size: .8rem; }
.res { display: flex; justify-content: space-between; align-items: center; border-radius: 12px; padding: 10px 14px; border: 2px solid; margin-top: 6px; }
.res-att { background: #fff1f1; border-color: #ef4444; } .res-ok { background: #effcf3; border-color: #22c55e; }
.res-v { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
.res-c { font-size: 1.5rem; font-weight: 800; color: #0f2a5c; text-align: right; line-height: 1.1; }
.barre-p { width: 120px; height: 7px; border-radius: 6px; background: #e5e7eb; overflow: hidden; margin-top: 4px; }
.barre-p > div { height: 100%; border-radius: 6px; }
.res-bas { display: grid; grid-template-columns: 1fr 1.6fr; gap: 8px; margin-top: 8px; font-size: .75rem; color: #334155; }
.res-bas > div { background: #f6f9fe; border-radius: 10px; padding: 6px 10px; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none; }
/* La liste (barre latérale) reste toujours accessible : bouton de réouverture toujours visible,
   et bouton « réduire » masqué sur grand écran pour ne jamais la perdre. */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { display: flex !important; visibility: visible !important; opacity: 1 !important; z-index: 999991; }
@media (min-width: 992px) { [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] button { display: none !important; } }
.note { font-size: .65rem; color: #64748b; line-height: 1.3; margin-top: 4px; }
/* Barre latérale claire comme la maquette */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3ebf7; }
[data-testid="stSidebar"] * { color: #334155 !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { background: transparent; box-shadow: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #eef3fb; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: #2563eb; box-shadow: 0 3px 10px rgba(37,99,235,.35); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * { color: #ffffff !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] button { border: 1px solid #cbd5e1; background: #fff; }
/* Valeurs min / max toujours visibles sous les sliders */
[data-testid="stSliderTickBar"] { opacity: 1 !important; visibility: visible !important; display: flex !important; }
[data-testid="stSliderTickBar"] p { font-size: .62rem !important; color: #64748b !important; }
.res-warn { background: #fffbeb; border-color: #f59e0b; }
/* ===== Mode compact : tout tient dans un seul écran, sans défilement ===== */
.block-container { padding: .35rem .9rem 0 .9rem !important; }
[data-testid="stVerticalBlock"] { gap: .35rem !important; }
[data-testid="stHorizontalBlock"] { gap: .45rem !important; }
[data-testid="stSidebar"] { width: 190px !important; min-width: 190px !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { padding: 5px 10px; font-size: .8rem; margin-bottom: 1px; }
[data-testid="stSidebarContent"] { padding-top: .5rem; }
.barre { padding: 4px 14px; margin-bottom: 8px; }
.b-ico { width: 34px; height: 34px; margin-right: 10px; }
.b-titre { font-size: 1.1rem; }
.b-sous { font-size: .64rem; }
.profil { padding: 3px 10px; font-size: .7rem; }
.kpi { height: 60px; padding: 5px 10px; gap: 9px; margin-bottom: 6px; }
.kpi-ic { min-width: 32px; height: 32px; }
.kpi-v { font-size: 1.15rem; } .kpi-t { font-size: .64rem; } .kpi-s { font-size: .56rem; }
[class*="st-key-carte_"] { padding: 6px 10px 2px; border-radius: 12px; }
.ct { font-size: .78rem; margin-bottom: 8px; line-height: 1.2; }
.pl { font-size: .7rem; padding: 3px 0; }
.lg { font-size: .64rem; padding: 0; line-height: 1.35; }
.tinc { font-size: .66rem; } .tinc th { padding: 1px 6px; } .tinc td { padding: 2px 6px; }
.badge { padding: 0 8px; font-size: .62rem; }
[data-testid="stWidgetLabel"] p { font-size: .66rem; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { min-height: 28px; font-size: .74rem; }
[data-testid="stNumberInput"] button { min-height: 28px; }
div.stButton > button { min-height: 30px; padding: 0 8px; font-size: .78rem; }
[data-testid="stSlider"] { padding-bottom: 0; }
[data-testid="stSliderThumbValue"] p, [data-testid="stSliderThumbValue"] { font-size: .7rem; }
[data-testid="stSliderTickBar"] { padding-bottom: 0 !important; }
[data-testid="stToggle"] label p { font-size: .72rem; }
.res { padding: 5px 12px; margin-top: 2px; } .res-v { font-size: 1.3rem; } .res-c { font-size: 1.3rem; }
.res-bas { margin-top: 6px; font-size: .68rem; } .res-bas > div { padding: 4px 9px; }
.note { white-space: nowrap; font-size: .56rem; margin-top: 8px; line-height: 1.25; }
/* Bannière d'en-tête (image fournie) */
.bandeau { position: relative; height: 64px; border-radius: 14px; overflow: hidden; display: flex; align-items: center; justify-content: center;
           background: radial-gradient(ellipse at 50% 50%, #0b2a57 0%, #061631 62%, #040d20 100%); margin-bottom: 8px;
           box-shadow: 0 2px 10px rgba(15,42,92,.18); }
.bandeau-i { height: 64px; width: auto; -webkit-mask-image: linear-gradient(90deg, transparent, #000 14%, #000 86%, transparent);
             mask-image: linear-gradient(90deg, transparent, #000 14%, #000 86%, transparent); }
.bandeau-d { position: absolute; right: 14px; top: 50%; transform: translateY(-50%); display: flex; align-items: center; gap: 12px; }
.bandeau .profil { background: rgba(255,255,255,.12); color: #e2e8f0; }
</style>
"""


def logo_html(largeur=190):
    """Logo « Attack Predict » généré par le code (SVG intégré) : aucun fichier image nécessaire."""
    src = "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode("utf-8")).decode()
    return f"<img src='{src}' style='width:{largeur}px;max-width:100%;height:auto;display:block;border-radius:12px'/>"


# Icônes « Lucide » (traits fins, même style partout)
ICONES = {
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "shield-alert": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "share-2": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "bars": '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
    "pie": '<path d="M21 12c.552 0 1.005-.449.95-.998a10 10 0 0 0-8.953-8.951c-.55-.055-.998.398-.998.95v8a1 1 0 0 0 1 1z"/><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "flask": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
}


def ic(nom, couleur="#2563eb", taille=18):
    """Retourne une icône SVG en ligne."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{taille}' height='{taille}' viewBox='0 0 24 24' fill='none' "
            f"stroke='{couleur}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' "
            f"style='vertical-align:-3px'>{ICONES[nom]}</svg>")


def entete(titre, sous_titre=""):
    """Titre de page compact (les pages secondaires)."""
    st.markdown(f"<div class='page-titre'>{titre}</div><div class='sous-titre'>{sous_titre}</div>", unsafe_allow_html=True)


def barre_entete():
    """En-tête : bannière « Attack Predict » (entete.png) + notifications / aide / profil à droite."""
    if os.path.exists(CHEMIN_ENTETE):
        with open(CHEMIN_ENTETE, "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        st.markdown(
            f"<div class='bandeau'><img class='bandeau-i' src='{src}'/>"
            f"<div class='bandeau-d'>{ic('bell', '#cbd5e1')}{ic('help', '#cbd5e1')}"
            f"<span class='profil'>{ic('user', '#cbd5e1', 15)} {MARQUE} ▾</span></div></div>", unsafe_allow_html=True)
        return
    st.markdown(
        f"<div class='barre'><div class='b-g'><div class='b-ico'>{ic('shield-check', '#2563eb', 26)}</div><div>"
        "<div class='b-titre'>Prédiction des attaques classiques en cyber</div>"
        "<div class='b-sous'>Analyse du trafic • Prédiction des attaques • Évaluation du modèle • Test de scénarios</div></div></div>"
        f"<div class='b-d'>{ic('bell', '#475569')}{ic('help', '#475569')}"
        f"<span class='profil'>{ic('user', '#475569', 15)} {MARQUE} ▾</span></div></div>",
        unsafe_allow_html=True)


def carte_kpi(titre, valeur, sous=""):
    st.markdown(
        f"<div class='carte'><div class='kpi-titre'>{titre}</div>"
        f"<div class='kpi-valeur'>{valeur}</div><div class='kpi-sous'>{sous}</div></div>",
        unsafe_allow_html=True,
    )


def style_figure(fig, hauteur=330):
    fig.update_layout(template="plotly_white", height=hauteur, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif"))
    return fig


# ----------------------------------------------------------------------------
# 2. CONNEXION (login / mot de passe)
# ----------------------------------------------------------------------------
def page_connexion():
    st.markdown(CSS, unsafe_allow_html=True)
    _, centre, _ = st.columns([1, 1.1, 1])
    with centre:
        st.markdown(
            f"<div style='text-align:center;margin-top:6vh'>{logo_html(260)}"
            f"<h1 style='margin-bottom:0'>Attack Predict</h1>"
            f"<div class='sous-titre'>{SLOGAN}</div></div>",
            unsafe_allow_html=True,
        )
        with st.form("connexion"):
            identifiant = st.text_input("Nom d'utilisateur", placeholder="admin")
            mdp = st.text_input("Mot de passe", type="password")
            ok = st.form_submit_button("Se connecter", width="stretch")
        if ok:
            # compare_digest évite les comparaisons « fuites de temps »
            bon_id = hmac.compare_digest(identifiant.strip().lower().encode(), UTILISATEUR.lower().encode())
            bon_mdp = hmac.compare_digest(mdp.strip().lower().encode(), MOT_DE_PASSE.lower().encode())
            if bon_id and bon_mdp:
                st.session_state["connecte"] = True
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")


# ----------------------------------------------------------------------------
# 3. DONNÉES : chargement et nettoyage (logique du notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_donnees(chemin, normal_csv=None):
    """Charge dataset.csv (+ un CSV optionnel de trafic normal), supprime les doublons
    et remplace les valeurs manquantes. Retourne (DataFrame, dictionnaire d'infos)."""
    df = pd.read_csv(chemin, low_memory=False)
    infos = {"lignes_brutes": len(df)}

    # Si des lignes ont is_attack = 0, elles représentent du trafic normal.
    if "is_attack" in df.columns:
        df.loc[df["is_attack"] == 0, "attack_type"] = "Normal"

    # CSV de trafic normal fourni par l'utilisateur (aucune donnée n'est inventée)
    if normal_csv is not None:
        n = pd.read_csv(io.BytesIO(normal_csv), low_memory=False)
        n["attack_type"], n["is_attack"] = "Normal", 0
        if len(n) > 20000:
            n = n.sample(20000, random_state=RANDOM_STATE)
        df = pd.concat([df, n], ignore_index=True, sort=False)

    df = df.dropna(subset=["attack_type"])
    if "sample_id" in df.columns:                      # identifiant technique inutile
        df = df.drop(columns=["sample_id"])
    infos["doublons"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Valeurs manquantes : 0 pour le numérique, valeur la plus fréquente pour le texte
    infos["manquants"] = df.isna().sum()[lambda s: s > 0].to_dict()
    num = df.select_dtypes(include=np.number).columns
    df[num] = df[num].fillna(0)
    for col in df.columns.difference(num):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df, infos


def vectoriser(df, colonnes=None, medianes=None, num_cols=None):
    """Transforme un tableau en variables numériques (get_dummies pour le texte).
    Si `colonnes` est donné, aligne le résultat sur les colonnes du modèle."""
    X = df.drop(columns=COLONNES_EXCLUES, errors="ignore").copy()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    X = pd.get_dummies(X, columns=cat, dtype=int)
    X = X.apply(pd.to_numeric, errors="coerce")
    if colonnes is not None:                           # prédiction sur de nouvelles données
        for c in colonnes:
            if c not in X.columns:                     # variable absente : médiane (numérique) ou 0
                X[c] = medianes[c] if c in num_cols else 0
        X = X[colonnes].fillna(pd.Series(medianes))
    return X.fillna(0)


# ----------------------------------------------------------------------------
# 4. ENTRAÎNEMENT DES MODÈLES (mis en cache : fait une seule fois)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entraînement des modèles en cours…")
def entrainer(_df, signature):
    """Entraîne 3 modèles, choisit le meilleur sur la validation, l'évalue sur le test."""
    df = _df
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["attack_type"]), index=df.index)
    classes = list(le.classes_)
    X = vectoriser(df)

    # Découpage stratifié 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp)

    modeles = {
        # La régression logistique a besoin de variables standardisées (pipeline)
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, objective="multi:softprob",
                                 eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
    }

    def metriques(y_vrai, y_pred):
        return {
            "Accuracy": accuracy_score(y_vrai, y_pred),
            "Précision (macro)": precision_score(y_vrai, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_vrai, y_pred, average="macro", zero_division=0),
            "F1-score (macro)": f1_score(y_vrai, y_pred, average="macro", zero_division=0),
        }

    lignes = []
    for nom, m in modeles.items():
        m.fit(X_tr, y_tr)
        lignes.append({"Modèle": nom, **metriques(y_va, m.predict(X_va))})
    comparaison = pd.DataFrame(lignes).sort_values("F1-score (macro)", ascending=False).reset_index(drop=True)

    meilleur_nom = comparaison.loc[0, "Modèle"]
    meilleur = modeles[meilleur_nom]
    pred_te = meilleur.predict(X_te)
    proba_te = meilleur.predict_proba(X_te)
    labels = np.arange(len(classes))

    # Tableau des « incidents » = vraies prédictions du modèle sur le jeu de test
    meta = df.loc[X_te.index, ["timestamp_utc", "src_ip", "dst_ip", "attack_type"]].copy()
    incidents = pd.DataFrame({
        "Horodatage": meta["timestamp_utc"].values,
        "Adresse source": meta["src_ip"].values,
        "Adresse destination": meta["dst_ip"].values,
        "Type réel": meta["attack_type"].values,
        "Classe prédite": [classes[i] for i in pred_te],
        "Probabilité": proba_te.max(axis=1),
    })
    incidents["Résultat"] = np.where(incidents["Classe prédite"] == "Normal", "Normal", "Attaque")
    incidents["Correct"] = incidents["Type réel"] == incidents["Classe prédite"]
    incidents = incidents.sort_values("Horodatage", ascending=False).reset_index(drop=True)

    # AUC (ROC) macro « un contre tous » sur le test
    try:
        auc = roc_auc_score(y_te, proba_te, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")

    # Prédictions du meilleur modèle sur tout l'échantillon chargé (pour le tableau de bord)
    pred_all = meilleur.predict(X)
    est_attaque = np.array([classes[i] != "Normal" for i in pred_all])
    heures = pd.to_datetime(df["timestamp_utc"], errors="coerce").dt.hour
    horaire = (pd.DataFrame({"h": heures.values, "att": est_attaque}).dropna(subset=["h"])
               .groupby("h").agg(total=("att", "size"), attaques=("att", "sum")).reindex(range(24), fill_value=0))
    par_classe_predite = pd.Series([classes[i] for i in pred_all]).value_counts()

    cm = confusion_matrix(y_te, pred_te, labels=labels)
    rappel = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(len(classes)), where=cm.sum(axis=1) != 0)
    rapport = pd.DataFrame(classification_report(y_te, pred_te, labels=labels, target_names=classes,
                                                 zero_division=0, output_dict=True)).T

    return {
        "classes": classes, "colonnes": X.columns.tolist(), "medianes": X_tr.median().to_dict(),
        "num_cols": [c for c in X.columns if not c.startswith(("protocol_", "service_"))],
        "modeles": modeles, "meilleur_nom": meilleur_nom, "meilleur": meilleur,
        "comparaison": comparaison, "test": metriques(y_te, pred_te),
        "cm": cm, "rappel": pd.Series(rappel, index=classes), "rapport": rapport,
        "importances": pd.Series(modeles["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False),
        "tailles": (len(X_tr), len(X_va), len(X_te)), "incidents": incidents,
        "auc": auc, "horaire": horaire, "par_classe_predite": par_classe_predite,
        "nb_attaques_predites": int(est_attaque.sum()),
    }


def predire(res, df_lignes):
    """Prédit le type d'attaque de nouvelles lignes. Retourne (classes prédites, probabilités)."""
    X = vectoriser(df_lignes, res["colonnes"], res["medianes"], res["num_cols"])
    proba = res["meilleur"].predict_proba(X)
    return [res["classes"][i] for i in proba.argmax(axis=1)], proba


# ----------------------------------------------------------------------------
# 5. PAGES
# ----------------------------------------------------------------------------
def afficher(fig, hauteur, haut=2):
    """Affiche un graphique Plotly compact, sans barre d'outils."""
    fig.update_layout(template="plotly_white", height=hauteur, margin=dict(l=4, r=4, t=haut, b=2),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      font=dict(family="Inter, Segoe UI, sans-serif", size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def titre_carte(texte):
    st.markdown(f"<div class='ct'>{texte}</div>", unsafe_allow_html=True)


def table_incidents_html(inc, n=5):
    """Petit tableau HTML des derniers incidents (vraies prédictions sur le jeu de test)."""
    lignes = ""
    for i, r in inc.head(n).iterrows():
        badge = "badge-ok" if r["Résultat"] == "Normal" else "badge-att"
        lignes += (f"<tr><td>{i + 1}</td><td>{str(r['Horodatage']).replace('T', ' ')}</td><td>{r['Adresse source']}</td>"
                   f"<td>{r['Adresse destination']}</td><td>{joli(r['Classe prédite'])}</td>"
                   f"<td>{fr_pct(r['Probabilité'])}</td><td><span class='badge {badge}'>{r['Résultat']}</span></td></tr>")
    return ("<table class='tinc'><thead><tr><th>#</th><th>Horodatage</th><th>Adresse source</th><th>Adresse destination</th>"
            "<th>Classe prédite</th><th>Probabilité</th><th>Résultat</th></tr></thead><tbody>" + lignes + "</tbody></table>")


# --- Panneau « Partie de test » ---------------------------------------------
def bornes(df, col):
    return float(df[col].min()), float(df[col].max())


def curseur(df, col, libelle, key):
    """Curseur (ligne) borné par les valeurs réelles du dataset, à la place des champs +/-."""
    lo, hi = bornes(df, col)
    if hi <= lo:
        st.session_state[key] = lo
        st.caption(f"{libelle} : {lo:g} (constante)")
        return
    v = min(max(float(st.session_state.get(key, lo) or 0), lo), hi)
    serie = df[col].dropna().head(2000)
    if pd.api.types.is_integer_dtype(df[col]) or bool((serie % 1 == 0).all()):
        lo_i, hi_i = int(lo), int(hi)
        pas = 1 if hi_i - lo_i <= 200000 else (hi_i - lo_i) // 1000
        v = lo_i + round((v - lo_i) / pas) * pas if pas > 1 else round(v)
        st.session_state[key] = int(min(max(v, lo_i), hi_i))
        st.slider(libelle, lo_i, hi_i, step=pas, key=key)
    else:
        st.session_state[key] = v
        st.slider(libelle, lo, hi, step=(hi - lo) / 200, format="%.2f", key=key)


def remplir_panneau(ligne, df):
    """Copie une ligne (médianes ou vrai exemple) dans les champs du panneau."""
    def borne(col):
        lo, hi = bornes(df, col)
        return float(min(max(float(ligne.get(col, 0) or 0), lo), hi))
    st.session_state["d_protocol"] = ligne["protocol"]
    st.session_state["d_service"] = ligne["service"]
    st.session_state["d_port"] = int(ligne.get("dst_port", 0) or 0)
    st.session_state["d_taille"] = borne("avg_packet_size_bytes")
    st.session_state["d_debit"] = borne("packet_rate_pps")
    st.session_state["d_duree"] = borne("flow_duration_ms")
    st.session_state["d_echecs"] = int(ligne.get("failed_logins_5m", 0) or 0)
    st.session_state["d_entropie"] = borne("payload_entropy")


def init_panneau(df):
    if "base_d" not in st.session_state:
        # Départ : une vraie ligne du jeu de données (1re classe de la liste)
        classe0 = sorted(df["attack_type"].unique())[0]
        base = df[df["attack_type"] == classe0].sample(1, random_state=RANDOM_STATE).iloc[0].to_dict()
        base = {k: (0 if pd.isna(v) else v) for k, v in base.items()}
        st.session_state["base_d"] = base
        remplir_panneau(base, df)
        st.session_state.update({"d_susp": 0.70, "d_sim": False, "d_journal": True})


def charger_exemple_panneau(df):
    """Callback : charge un vrai exemple du jeu de données de la classe choisie."""
    ligne = df[df["attack_type"] == st.session_state["d_ex"]].sample(1).iloc[0].to_dict()
    ligne = {k: (0 if pd.isna(v) else v) for k, v in ligne.items()}
    st.session_state["base_d"] = ligne
    remplir_panneau(ligne, df)
    st.session_state.pop("resultat", None)


def panneau_test(df, res):
    with st.container(key="carte_test"):
        st.markdown(f"<div class='ct' style='font-size:1rem'>{ic('flask', '#2563eb', 20)} Partie de test</div>"
                    "<div class='kpi-s' style='margin-bottom:8px'>Modifiez les paramètres du trafic et testez la prédiction.</div>", unsafe_allow_html=True)
        e1, e2 = st.columns([1.5, 1])
        e1.selectbox("Exemple réel", res["classes"], key="d_ex", format_func=joli, label_visibility="collapsed")
        e2.button("Charger", on_click=charger_exemple_panneau, args=(df,), key="d_charger", width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Protocole", sorted(df["protocol"].unique()), key="d_protocol")
            curseur(df, "avg_packet_size_bytes", "Taille du paquet (octets)", "d_taille")
            lo, hi = bornes(df, "flow_duration_ms")
            st.slider("Durée (ms)", lo, hi, step=max((hi - lo) / 200, 0.01), format="%.0f", key="d_duree")
            st.selectbox("Service de destination", sorted(df["service"].unique()), key="d_service")
            st.slider("Niveau de suspicion", 0.0, 1.0, step=0.01, format="%.2f", key="d_susp")
        with c2:
            curseur(df, "dst_port", "Port", "d_port")
            lo, hi = bornes(df, "packet_rate_pps")
            st.slider("Taux de paquets (pps)", lo, hi, step=max((hi - lo) / 200, 0.01), format="%.0f", key="d_debit")
            curseur(df, "failed_logins_5m", "Tentatives échouées", "d_echecs")
            lo, hi = bornes(df, "payload_entropy")
            st.slider("Entropie de la charge utile", lo, hi, step=0.01, key="d_entropie")
            st.toggle("Mode simulation", key="d_sim")
            st.toggle("Journaliser le test", key="d_journal")

        if st.button("▷ Prédire", type="primary", width="stretch", key="d_predire"):
            # Les variables non affichées viennent de la ligne chargée (médianes par défaut)
            ligne = dict(st.session_state["base_d"])
            ligne.update({
                "protocol": st.session_state["d_protocol"], "service": st.session_state["d_service"],
                "dst_port": st.session_state["d_port"], "avg_packet_size_bytes": st.session_state["d_taille"],
                "packet_rate_pps": st.session_state["d_debit"], "flow_duration_ms": st.session_state["d_duree"],
                "failed_logins_5m": st.session_state["d_echecs"], "payload_entropy": st.session_state["d_entropie"],
            })
            classes_pred, proba = predire(res, pd.DataFrame([ligne]))
            p = proba[0]
            top = np.argsort(p)[::-1][:3]
            st.session_state["resultat"] = {
                "classe": classes_pred[0], "conf": float(p.max()), "seuil": st.session_state["d_susp"],
                "simulation": st.session_state["d_sim"],
                "top3": [(res["classes"][i], float(p[i])) for i in top],
            }
            if st.session_state.get("d_journal") and not st.session_state.get("d_sim"):
                st.session_state.setdefault("journal", []).append({
                    "Horodatage": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Protocole": ligne["protocol"], "Service": ligne["service"], "Port": ligne["dst_port"],
                    "Classe prédite": classes_pred[0], "Confiance": round(float(p.max()), 4),
                })

        r = st.session_state.get("resultat")
        if r:
            # Normal si le modèle le prédit ; sinon « Attaque » si la confiance atteint le seuil d'alerte
            if r["classe"] == "Normal":
                etat, css, couleur = "Normal", "res-ok", "#22c55e"
            elif r["conf"] >= r["seuil"]:
                etat, css, couleur = "Attaque", "res-att", "#ef4444"
            else:
                etat, css, couleur = "À vérifier", "res-warn", "#f59e0b"
            icone_res = "shield-check" if etat == "Normal" else "shield-alert"
            sim = " <span class='badge badge-info'>simulation</span>" if r["simulation"] else ""
            top3 = " • ".join(f"{joli(k)} ({fr_pct(v)})" for k, v in r["top3"])
            st.markdown(
                f"<div class='res {css}'><div style='display:flex;align-items:center;gap:10px'>{ic(icone_res, couleur, 34)}<div>"
                f"<div class='kpi-t'>Résultat de la prédiction</div>"
                f"<div class='res-v' style='color:{couleur}'>{etat}</div>{sim}</div></div>"
                f"<div><div class='kpi-t' style='text-align:right'>Confiance</div><div class='res-c'>{fr_pct(r['conf'])}</div>"
                f"<div class='barre-p'><div style='width:{r['conf'] * 100:.1f}%;background:{couleur}'></div></div></div></div>"
                f"<div class='res-bas'><div><span class='kpi-t'>Classe prédite</span><br>"
                f"<span class='badge badge-info'>{joli(r['classe'])}</span></div>"
                f"<div><span class='kpi-t'>Probabilités (Top 3)</span><br>{top3}</div></div>",
                unsafe_allow_html=True)
        else:
            st.markdown("<div class='note'>Cliquez sur « Prédire » pour lancer le modèle.</div>", unsafe_allow_html=True)
        note = "Calculé par le modèle • Suspicion = seuil d'alerte • Simulation = non journalisé"
        if "Normal" not in res["classes"]:
            pass
        st.markdown(f"<div class='note'>{note}</div>", unsafe_allow_html=True)


# --- Tableau de bord ---------------------------------------------------------
def page_dashboard(df, res):
    init_panneau(df)
    principal, droite = st.columns([2.1, 1], gap="small")
    couleurs = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(res["classes"])}
    t = res["test"]
    nb_att = res["nb_attaques_predites"]

    with principal:
        # 1) Quatre cartes KPI (valeurs calculées, pas de variation inventée)
        kpis = [("shield-check", "#dbeafe", "Trafic analysé", fr_nombre(len(df)), "lignes chargées"),
                ("triangle-alert", "#fee2e2", "Attaques prédites", fr_nombre(nb_att), "lignes classées en attaque"),
                ("target", "#ede9fe", "Taux d'attaque", fr_pct(nb_att / len(df)), "part prédite en attaque"),
                ("share-2", "#cffafe", "Précision du modèle", fr_pct(t["Accuracy"]), "Accuracy (jeu de test)")]
        for col, (icone, bg, titre, val, sous) in zip(st.columns(4, gap="small"), kpis):
            col.markdown(f"<div class='kpi'><div class='kpi-ic' style='background:{bg}'>{ic(icone, {'#dbeafe':'#2563eb','#fee2e2':'#dc2626','#ede9fe':'#7c3aed','#cffafe':'#0891b2'}[bg], 20)}</div><div>"
                         f"<div class='kpi-t'>{titre}</div><div class='kpi-v'>{val}</div><div class='kpi-s'>{sous}</div></div></div>",
                         unsafe_allow_html=True)

        # 2) Tendances du trafic + performances du modèle
        a, b = st.columns([1.3, 1], gap="small")
        with a:
            with st.container(key="carte_trafic"):
                titre_carte(f"{ic('trending-up')} Tendances du trafic réseau (par heure de la journée)")
                h = res["horaire"]
                etiq = [f"{i:02d}h" for i in h.index]
                fig = go.Figure()
                fig.add_scatter(x=etiq, y=h["total"], name="Trafic total", line=dict(color="#2563eb", width=2),
                                fill="tozeroy", fillcolor="rgba(37,99,235,.10)")
                fig.add_scatter(x=etiq, y=h["attaques"], name="Attaques prédites", line=dict(color="#7c3aed", width=2))
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14, x=0, font_size=10),
                                  xaxis=dict(tickvals=etiq[::3], tickfont_size=9),
                                  yaxis=dict(title="Nombre de lignes", title_font_size=9, tickfont_size=9))
                afficher(fig, 130, haut=14)
        with b:
            with st.container(key="carte_perf"):
                titre_carte(f"{ic('bars')} Performances du modèle <span style='font-weight:500;color:#64748b;font-size:.7rem'>(test)</span>")
                g, lst = st.columns([1.25, 1.15])
                with g:
                    acc = t["Accuracy"]
                    fig = go.Figure(go.Pie(values=[acc, max(1 - acc, 0)], hole=0.74, sort=False, textinfo="none",
                                           marker_colors=["#4f46e5", "#e5e7eb"], hoverinfo="skip"))
                    fig.add_annotation(text=f"<b>{fr_pct(acc)}</b><br><span style='font-size:10px'>Accuracy</span>",
                                       showarrow=False, font=dict(size=14, color="#0f2a5c"))
                    afficher(fig, 112)
                with lst:
                    auc = res["auc"]
                    lignes = [("Précision", fr_pct(t["Précision (macro)"]), "#2563eb"),
                              ("Recall", fr_pct(t["Recall (macro)"]), "#7c3aed"),
                              ("F1-score", fr_pct(t["F1-score (macro)"]), "#0ea5e9"),
                              ("AUC (ROC)", fr_pct(auc) if auc == auc else "n/d", "#10b981")]
                    st.markdown("".join(f"<div class='pl'><span><i style='background:{c}'></i>{n}</span><b>{v}</b></div>"
                                        for n, v, c in lignes), unsafe_allow_html=True)

        # 3) Fréquence par classe + répartition (attaques seulement)
        cnt = res["par_classe_predite"].drop(labels=["Normal"], errors="ignore").sort_values(ascending=False)
        c, d = st.columns([1.15, 1.2], gap="small")
        with c:
            with st.container(key="carte_freq"):
                titre_carte(f"{ic('bars')} Fréquence des attaques par classe")
                asc = cnt.sort_values()
                fig = go.Figure(go.Bar(x=asc.values, y=[joli(k) for k in asc.index], orientation="h", marker_color="#2563eb",
                                       text=[fr_nombre(v) for v in asc.values], textposition="outside", cliponaxis=False))
                fig.update_layout(xaxis=dict(range=[0, cnt.max() * 1.2], tickfont_size=8), yaxis=dict(tickfont_size=8.5))
                afficher(fig, 158)
        with d:
            with st.container(key="carte_repart"):
                titre_carte(f"{ic('pie')} Répartition des attaques ({len(cnt)} classes)")
                g, lg = st.columns([1.1, 1])
                with g:
                    fig = go.Figure(go.Pie(labels=[joli(k) for k in cnt.index], values=cnt.values, hole=0.6, sort=False,
                                           marker_colors=[couleurs[k] for k in cnt.index], textinfo="percent", textfont_size=8))
                    fig.add_annotation(text=f"<b>{fr_nombre(cnt.sum())}</b><br><span style='font-size:9px'>attaques</span>",
                                       showarrow=False, font=dict(size=13, color="#0f2a5c"))
                    afficher(fig, 150)
                with lg:
                    total = cnt.sum()
                    st.markdown("".join(
                        f"<div class='lg'><span><i style='background:{couleurs[k]}'></i>{joli(k)}</span><b>{fr_pct(v / total, 1)}</b></div>"
                        for k, v in cnt.items()), unsafe_allow_html=True)

        # 4) Incidents récents (5 lignes seulement)
        with st.container(key="carte_inc"):
            titre_carte(f"{ic('database')} Incidents récents (prédictions)")
            st.markdown(table_incidents_html(res["incidents"], 5), unsafe_allow_html=True)

    with droite:
        panneau_test(df, res)


# Variables du formulaire de test : (colonne, libellé, pas)
CHAMPS = [
    ("dst_port", "Port destination", 1.0), ("flow_duration_ms", "Durée du flux (ms)", 100.0),
    ("packet_count", "Nombre de paquets", 100.0), ("packet_rate_pps", "Débit (paquets/s)", 100.0),
    ("syn_count", "Paquets SYN", 10.0), ("unique_dst_ports_5m", "Ports distincts (5 min)", 1.0),
    ("failed_logins_5m", "Échecs de connexion (5 min)", 1.0), ("http_requests_5m", "Requêtes HTTP (5 min)", 10.0),
    ("sql_keyword_hits_5m", "Mots-clés SQL (5 min)", 1.0), ("xss_pattern_hits_5m", "Motifs XSS (5 min)", 1.0),
    ("email_links_count", "Liens dans l'e-mail", 1.0), ("file_modifications_5m", "Fichiers modifiés (5 min)", 10.0),
    ("file_entropy", "Entropie des fichiers", 0.1), ("payload_entropy", "Entropie de la charge utile", 0.1),
    ("beacon_interval_cv", "Régularité des balises (CV)", 0.05), ("arp_spoof_events_5m", "Événements ARP spoof (5 min)", 1.0),
    ("dns_queries_5m", "Requêtes DNS (5 min)", 1.0),
]


def initialiser_formulaire(df):
    """Valeurs par défaut du formulaire = médianes du jeu de données."""
    if "ligne_base" not in st.session_state:
        base = df.drop(columns=COLONNES_EXCLUES, errors="ignore").median(numeric_only=True).to_dict()
        base["protocol"] = df["protocol"].mode().iloc[0]
        base["service"] = df["service"].mode().iloc[0]
        st.session_state["ligne_base"] = base
        remplir_formulaire(base)


def remplir_formulaire(ligne):
    for col, _, _ in CHAMPS:
        st.session_state[f"f_{col}"] = float(ligne.get(col, 0) or 0)
    st.session_state["f_protocol"] = ligne["protocol"]
    st.session_state["f_service"] = ligne["service"]


def charger_exemple(df):
    """Charge une vraie ligne du dataset (de la classe choisie) dans le formulaire."""
    classe = st.session_state["ex_classe"]
    ligne = df[df["attack_type"] == classe].sample(1).iloc[0].to_dict()
    ligne = {k: (0 if pd.isna(v) else v) for k, v in ligne.items()}
    st.session_state["ligne_base"] = ligne
    remplir_formulaire(ligne)


def page_prediction(df, res):
    entete("Prédiction / Test", "Testez un trafic réseau et obtenez le type d'attaque prédit")
    if "Normal" not in res["classes"]:
        st.warning("Le modèle n'a appris que des attaques : il ne peut pas répondre « Normal ». "
                   "Ajoutez du trafic normal (is_attack = 0) dans **Données & Modèle** pour activer cette classe.")
    initialiser_formulaire(df)
    onglet1, onglet2 = st.tabs(["🧪 Test manuel", "📂 Prédiction sur un fichier CSV"])

    with onglet1:
        a, b = st.columns([3, 1])
        with a:
            st.selectbox("Charger un vrai exemple du jeu de données (optionnel)", res["classes"],
                         key="ex_classe", format_func=joli)
        with b:
            st.write(""); st.write("")
            st.button("Charger l'exemple", on_click=charger_exemple, args=(df,), width="stretch")
        st.caption("Les variables non affichées ci-dessous prennent la valeur de la ligne chargée "
                   "(ou la médiane du jeu de données par défaut).")

        p1, p2 = st.columns(2)
        p1.selectbox("Protocole", sorted(df["protocol"].unique()), key="f_protocol")
        p2.selectbox("Service de destination", sorted(df["service"].unique()), key="f_service")
        cols = st.columns(3)
        for i, (col, libelle, pas) in enumerate(CHAMPS):
            with cols[i % 3]:
                curseur(df, col, libelle, f"f_{col}")

        if st.button("🔍 Prédire", type="primary", width="stretch"):
            ligne = dict(st.session_state["ligne_base"])
            for col, _, _ in CHAMPS:
                ligne[col] = st.session_state[f"f_{col}"]
            ligne["protocol"], ligne["service"] = st.session_state["f_protocol"], st.session_state["f_service"]
            classes_pred, proba = predire(res, pd.DataFrame([ligne]))
            pred, p = classes_pred[0], proba[0]
            est_attaque = pred != "Normal"
            css = "resultat-att" if est_attaque else "resultat-ok"
            titre = "⚠️ Attaque détectée" if est_attaque else "✅ Trafic normal"
            st.markdown(
                f"<div class='resultat {css}'><div class='kpi-titre'>Résultat de la prédiction</div>"
                f"<div class='kpi-valeur'>{titre}</div>"
                f"<div>Classe prédite : <b>{ATTAQUES.get(pred, ('', ''))[0]} {joli(pred)}</b> — "
                f"confiance : <b>{p.max():.2%}</b></div></div>", unsafe_allow_html=True)
            top = np.argsort(p)[::-1][:3]
            st.write("**Top 3 des probabilités :** " + " • ".join(f"{joli(res['classes'][i])} ({p[i]:.2%})" for i in top))
            fig = px.bar(x=[joli(k) for k in res["classes"]], y=p, labels={"x": "", "y": "Probabilité"},
                         title="Probabilité par classe")
            fig.update_traces(marker_color="#2563eb")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(style_figure(fig, 300), width="stretch")

    with onglet2:
        st.write("Importez un CSV de trafic (mêmes colonnes que le dataset ; les colonnes manquantes sont remplacées par la médiane).")
        fichier = st.file_uploader("Fichier CSV", type="csv", key="csv_pred")
        if fichier is not None:
            nouveau = pd.read_csv(fichier, low_memory=False)
            classes_pred, proba = predire(res, nouveau)
            sortie = nouveau.copy()
            sortie["classe_predite"] = classes_pred
            sortie["confiance"] = proba.max(axis=1).round(4)
            st.success(f"{len(sortie)} lignes analysées.")
            st.dataframe(sortie.head(500), width="stretch", hide_index=True)
            comptes = sortie["classe_predite"].value_counts()
            fig = px.bar(x=[joli(k) for k in comptes.index], y=comptes.values, title="Classes prédites",
                         labels={"x": "", "y": "Nombre"})
            st.plotly_chart(style_figure(fig, 280), width="stretch")
            st.download_button("⬇️ Télécharger les prédictions (CSV)", sortie.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")


@st.cache_data(show_spinner=False)
def signatures_attaques(_df, signature):
    """Pour chaque classe : les 3 variables dont la moyenne s'écarte le plus de la moyenne générale."""
    num = _df.drop(columns=COLONNES_EXCLUES, errors="ignore").select_dtypes(include=np.number)
    moy_g, ecart = num.mean(), num.std().replace(0, np.nan)
    sortie = {}
    for classe, groupe in _df.groupby("attack_type"):
        z = ((groupe[num.columns].mean() - moy_g) / ecart).dropna().sort_values(ascending=False).head(3)
        sortie[classe] = [(v, groupe[v].mean()) for v in z.index]
    return sortie


def page_attaques(df, res):
    entete("Catalogue des attaques", "Toutes les attaques détectées par le modèle, avec leurs variables les plus distinctives")
    sig = signatures_attaques(df, f"{len(df)}")
    rappel = res["rappel"]
    effectifs = df["attack_type"].value_counts()
    classes = res["classes"]
    for i in range(0, len(classes), 3):
        cols = st.columns(3)
        for col, classe in zip(cols, classes[i:i + 3]):
            icone, desc = ATTAQUES.get(classe, ("⚠️", ""))
            variables = "".join(f"<li><code>{v}</code> : moy. {m:,.2f}</li>" for v, m in sig.get(classe, []))
            with col:
                st.markdown(
                    f"<div class='carte'><div class='attaque-nom'>{icone} {joli(classe)}</div>"
                    f"<div class='attaque-desc'>{desc}</div>"
                    f"<span class='badge badge-att'>{effectifs.get(classe, 0):,} lignes</span> "
                    f"<span class='badge badge-ok'>Recall test : {rappel[classe]:.1%}</span>"
                    f"<div class='kpi-sous' style='margin-top:8px'>Variables les plus distinctives :</div>"
                    f"<ul style='margin:0;padding-left:18px;font-size:.8rem'>{variables}</ul></div>",
                    unsafe_allow_html=True)
        st.write("")


def page_analyses(df, res):
    entete("Analyses", "Signatures des attaques et importance des variables")
    cles = [c for c, _, _ in CHAMPS if c in df.columns and c != "dst_port"]
    moyennes = df.groupby("attack_type")[cles].mean()
    z = (moyennes - moyennes.mean()) / moyennes.std().replace(0, np.nan)
    fig = px.imshow(z.fillna(0).round(2), color_continuous_scale="RdBu_r", aspect="auto", zmin=-3, zmax=3,
                    title="Signature des attaques (moyenne par classe, standardisée par variable)")
    st.plotly_chart(style_figure(fig, 430), width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        top = res["importances"].head(15).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", title="Top 15 des variables (Random Forest)",
                     labels={"x": "Importance", "y": ""})
        fig.update_traces(marker_color="#2563eb")
        st.plotly_chart(style_figure(fig, 430), width="stretch")
    with g2:
        variable = st.selectbox("Distribution d'une variable par type d'attaque", cles, index=0)
        echantillon = df.sample(min(6000, len(df)), random_state=RANDOM_STATE)
        fig = px.box(echantillon, x="attack_type", y=variable, color="attack_type", points=False)
        fig.update_layout(showlegend=False, xaxis_title="", xaxis_tickangle=-35)
        st.plotly_chart(style_figure(fig, 380), width="stretch")
    st.caption("L'importance indique les variables utilisées par les arbres ; elle ne prouve pas une relation de cause à effet.")


def page_incidents(df, res):
    entete("Incidents", "Prédictions du modèle sur le jeu de test (données jamais vues à l'entraînement)")
    inc = res["incidents"]
    c1, c2 = st.columns([2, 1])
    choix = c1.multiselect("Filtrer par classe prédite", res["classes"], format_func=joli)
    seulement_erreurs = c2.checkbox("Afficher seulement les erreurs de prédiction")
    vue = inc.copy()
    if choix:
        vue = vue[vue["Classe prédite"].isin(choix)]
    if seulement_erreurs:
        vue = vue[~vue["Correct"]]
    st.write(f"**{len(vue):,}** incidents affichés — erreurs de prédiction : **{int((~inc['Correct']).sum())}** sur {len(inc):,}.")
    aff = vue.head(1000).copy()
    aff["Probabilité"] = (aff["Probabilité"] * 100).round(2)
    st.dataframe(aff, width="stretch", hide_index=True, height=340,
                 column_config={"Probabilité": st.column_config.ProgressColumn("Probabilité (%)", min_value=0, max_value=100, format="%.2f")})
    st.download_button("⬇️ Télécharger (CSV)", vue.to_csv(index=False).encode("utf-8"), "incidents.csv", "text/csv")
    if st.session_state.get("journal"):
        with st.expander(f"Tests manuels journalisés ({len(st.session_state['journal'])})"):
            st.dataframe(pd.DataFrame(st.session_state["journal"]), width="stretch", hide_index=True, height=200)


def page_donnees(df, res, infos):
    entete("Données & Modèle", "Chargement du CSV, nettoyage et configuration des modèles")
    c = st.columns(4)
    with c[0]: carte_kpi("Lignes brutes", f"{infos['lignes_brutes']:,}".replace(",", " "))
    with c[1]: carte_kpi("Doublons supprimés", str(infos["doublons"]))
    with c[2]: carte_kpi("Lignes utilisées", f"{len(df):,}".replace(",", " "))
    with c[3]: carte_kpi("Variables du modèle", str(len(res["colonnes"])))
    st.write("")

    with st.expander("Aperçu du dataset (8 premières lignes)"):
        st.dataframe(df.head(8), width="stretch", hide_index=True, height=250)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Valeurs manquantes (avant traitement)")
        if infos["manquants"]:
            st.dataframe(pd.Series(infos["manquants"], name="Nombre de NaN"), width="stretch")
            st.caption("Numérique → remplacé par 0 ; texte → valeur la plus fréquente (comme dans le notebook).")
        else:
            st.success("Aucune valeur manquante.")
    with g2:
        st.markdown("##### Découpage et modèles")
        tr, va, te = res["tailles"]
        st.write(f"Entraînement : **{tr:,}** • Validation : **{va:,}** • Test : **{te:,}** (stratifié 70/15/15)")
        st.write("Modèles : Régression logistique (standardisée), Random Forest (100 arbres), "
                 "XGBoost (100 arbres, profondeur 4).")
        st.write(f"Meilleur modèle (validation) : **{res['meilleur_nom']}**")

    st.markdown("##### Ajouter du trafic normal (active la classe « Normal »)")
    st.caption("Importez un CSV ayant les mêmes colonnes que le dataset et contenant uniquement du trafic légitime. "
               "Les lignes seront étiquetées « Normal » et les modèles seront ré-entraînés.")
    fichier = st.file_uploader("CSV de trafic normal", type="csv", key="csv_normal")
    a, b = st.columns(2)
    if fichier is not None and a.button("Ajouter et ré-entraîner", type="primary"):
        st.session_state["normal_csv"] = fichier.getvalue()
        st.rerun()
    if "normal_csv" in st.session_state and b.button("Retirer le trafic normal"):
        del st.session_state["normal_csv"]
        st.rerun()


def page_evaluation(df, res):
    entete("Évaluation du modèle", "Comparaison des algorithmes (validation) et résultats finaux (test)")
    st.markdown("##### Comparaison sur le jeu de validation")
    comp = res["comparaison"]
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns[1:]}), width="stretch", hide_index=True)
    fig = px.bar(comp, x="Modèle", y="F1-score (macro)", text=comp["F1-score (macro)"].map("{:.4f}".format))
    fig.update_traces(marker_color="#2563eb", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1.1], title="F1-score macro (validation)")
    st.plotly_chart(style_figure(fig, 300), width="stretch")

    st.markdown(f"##### Résultats finaux sur le test — {res['meilleur_nom']}")
    cols = st.columns(4)
    for col, (nom, val) in zip(cols, res["test"].items()):
        with col: carte_kpi(nom, f"{val:.2%}")
    st.write("")

    classes = [joli(k) for k in res["classes"]]
    cm = res["cm"].astype(float)
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) != 0)
    g1, g2 = st.columns([1.3, 1])
    with g1:
        fig = px.imshow(cm_norm, x=classes, y=classes, text_auto=".2f", color_continuous_scale="Blues",
                        labels=dict(x="Classe prédite", y="Classe réelle"), title="Matrice de confusion normalisée (test)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(fig, 520), width="stretch")
    with g2:
        rap = res["rapport"].iloc[:len(classes)].copy()
        rap.index = classes
        st.markdown("**Rapport par classe (test)**")
        st.dataframe(rap[["precision", "recall", "f1-score", "support"]].round(4), width="stretch", height=300)

    st.warning("⚠️ Des scores proches de 100 % sur des données synthétiques peuvent venir de classes très faciles à "
               "séparer. Ils ne garantissent pas les mêmes performances sur un réseau réel : il faudrait des données "
               "indépendantes, représentatives, avec du trafic normal et malveillant.")


def page_parametres(df, res):
    entete("Paramètres", "Compte, journal des tests et modèle")
    st.write(f"Utilisateur connecté : **{UTILISATEUR}**")
    st.write(f"Modèle utilisé : **{res['meilleur_nom']}** — {len(res['colonnes'])} variables, {len(res['classes'])} classes.")
    c1, c2 = st.columns(2)
    if c1.button("Vider le journal des tests", width="stretch"):
        st.session_state["journal"] = []
        st.success("Journal vidé.")
    if c2.button("Vider le cache et ré-entraîner", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


# ----------------------------------------------------------------------------
# 6. PROGRAMME PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    if AUTH_ACTIVE and not st.session_state.get("connecte"):
        page_connexion()
        return

    st.markdown(CSS, unsafe_allow_html=True)

    if not os.path.exists(CHEMIN_CSV):
        st.error("Fichier dataset.csv introuvable : placez-le à côté de app.py.")
        st.stop()

    # Données + modèles (calculés une seule fois grâce au cache)
    df, infos = charger_donnees(CHEMIN_CSV, st.session_state.get("normal_csv"))
    signature = f"{len(df)}-{df['attack_type'].nunique()}-{int('normal_csv' in st.session_state)}"
    res = entrainer(df, signature)

    # Barre latérale : menu + déconnexion
    with st.sidebar:
        st.markdown(f"<div style='padding:4px 2px 14px'>{logo_html(150)}</div>", unsafe_allow_html=True)
        page = st.radio("Navigation", ["Tableau de bord", "Prédiction / Test", "Analyses", "Attaques", "Incidents",
                                       "Données & Modèle", "Évaluation", "Paramètres"],
                        label_visibility="collapsed")
        if AUTH_ACTIVE and st.button("Se déconnecter", width="stretch"):
            st.session_state.clear()
            st.rerun()

    barre_entete()
    if page.endswith("Tableau de bord"): page_dashboard(df, res)
    elif page.endswith("Prédiction / Test"): page_prediction(df, res)
    elif page.endswith("Analyses"): page_analyses(df, res)
    elif page.endswith("Attaques"): page_attaques(df, res)
    elif page.endswith("Incidents"): page_incidents(df, res)
    elif page.endswith("Données & Modèle"): page_donnees(df, res, infos)
    elif page.endswith("Paramètres"): page_parametres(df, res)
    else: page_evaluation(df, res)


main()
