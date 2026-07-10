"""
Génération du graphe de connaissances multi-relationnel
Domaine : Bibliothèque — version "positions et tailles EXACTES"
--------------------------------------------------------------
Contrairement aux tentatives précédentes (où les positions étaient
estimées à l'œil), CETTE version utilise des coordonnées et des
rayons de cercles mesurés AUTOMATIQUEMENT sur l'image de référence,
grâce à une détection de cercles par transformée de Hough (OpenCV,
fonction cv2.HoughCircles). Chaque nœud reprend donc :
    - sa position (x, y) exacte en pixels
    - son rayon exact en pixels
    - sa couleur de bordure exacte (mesurée séparément par
      échantillonnage de pixels)

Le canevas matplotlib est configuré à la MÊME résolution que
l'image d'origine (1536 x 1024 px) et les coordonnées sont
utilisées telles quelles (1 unité de données = 1 pixel), ce qui
garantit que la disposition finale colle au plus près de l'image.

Dépendances :
    pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patheffects as pe

# =================================================================
# 1. DIMENSIONS DE L'IMAGE DE RÉFÉRENCE (pour un canevas identique)
# =================================================================
LARGEUR_PX = 1536
HAUTEUR_PX = 1024


def conv_y(y_image):
    """Convertit une coordonnée Y mesurée sur l'image (axe vers le
    bas) en coordonnée Y matplotlib (axe vers le haut)."""
    return HAUTEUR_PX - y_image


# =================================================================
# 2. PALETTE DE COULEURS (mesurée par échantillonnage de pixels
#    sur les bordures des cercles de l'image de référence)
# =================================================================
COLOR_BY_TYPE = {
    "Membre":         "#0D620C",
    "Auteur":         "#6A9F17",
    "Livre":          "#EA5C09",
    "Éditeur":        "#148B84",
    "Catégorie":      "#DD1514",
    "Rayon":          "#DAB914",
    "Bibliothèque":   "#0E217B",
    "Ville":          "#0E660E",
    "Bibliothécaire": "#9663AF",
    "Type":           "#9663AF",   # tous les nœuds "type" (ontologie) sont violets
}

COULEUR_FLECHE_METIER = "#2451D6"
COULEUR_FLECHE_TYPE = "#8C8C8C"

# =================================================================
# 3. NŒUDS : (x_image, y_image, rayon_px, catégorie, texte affiché)
# =================================================================
# x_image / y_image / rayon_px proviennent directement de la
# détection automatique de cercles (cv2.HoughCircles) sur l'image
# fournie ; ce sont donc les positions et tailles RÉELLES des
# cercles dans l'image de référence, pas des estimations à l'œil.

NODES = {
    "dr_arthur":        (916, 64, 60, "Bibliothécaire", "DR ARTHUR\nSAWADOGO\n(Bibliothécaire)"),
    "type_biblio_caire": (1362, 212, 54, "Type", "BIBLIOTHÉCAIRE"),
    "biblio_centrale":  (818, 218, 58, "Bibliothèque", "BIBLIOTHÈQUE\nCENTRALE"),
    "biblio_univ":      (1080, 218, 60, "Bibliothèque", "BIBLIOTHÈQUE\nUNIVERSITAIRE"),
    "zongo":            (140, 324, 56, "Membre", "ZONGO WENDESO\nADELINE\n(Membre)"),
    "gnambre":          (364, 326, 48, "Membre", "GNAMBRE\nMOHAMINE\n(Membre)"),
    "nazi_boni":        (508, 328, 43, "Auteur", "NAZI BONI\n(Auteur)"),
    "norbert_zongo":    (634, 328, 44, "Auteur", "NORBERT\nZONGO\n(Auteur)"),
    "bobo":             (1198, 414, 51, "Ville", "BOBO\nDIOULASSO\n(Ville)"),
    "koudougou":        (1054, 416, 52, "Ville", "KOUDOUGOU\n(Ville)"),
    "ouaga":            (884, 420, 57, "Ville", "OUAGADOUGOU\n(Ville)"),
    "type_biblio":      (1426, 424, 48, "Type", "BIBLIOTHÈQUE"),
    "type_personne":    (40, 498, 36, "Type", "PERSONNE"),
    "type_membre":      (120, 498, 33, "Type", "MEMBRE"),
    "parachutage":      (228, 520, 52, "Livre", "LE\nPARACHUTAGE"),
    "sentiers":         (366, 520, 53, "Livre", "LES SENTIERS\nD'ABRAHAM"),
    "crepuscule":       (508, 522, 55, "Livre", "CRÉPUSCULE DES\nTEMPS ANCIENS"),
    "type_auteur":      (706, 522, 36, "Type", "AUTEUR"),
    "type_ville":       (936, 598, 38, "Type", "VILLE"),
    "rayon_a":          (648, 718, 46, "Rayon", "RAYON A"),
    "rayon_b":          (788, 720, 46, "Rayon", "RAYON B"),
    "histoire":         (160, 734, 38, "Catégorie", "HISTOIRE"),
    "roman":            (52, 736, 37, "Catégorie", "ROMAN"),
    "presses":          (430, 738, 47, "Éditeur", "PRESSES UNIV.\nDE OUAGA"),
    "editions_faso":    (292, 740, 48, "Éditeur", "LES ÉDITIONS\nDU FASO"),
    "type_rayon":       (706, 890, 38, "Type", "RAYON"),
    "type_editeur":     (346, 896, 42, "Type", "ÉDITEUR"),
    "type_categorie":   (96, 898, 43, "Type", "CATÉGORIE"),
}

# =================================================================
# 4. ARÊTES "MÉTIER" (flèches bleues pleines + libellé bleu)
# =================================================================
ARETES_METIER = [
    ("dr_arthur", "biblio_centrale",   "TRAVAILLE DANS"),
    ("dr_arthur", "biblio_univ",       "TRAVAILLE DANS"),

    ("biblio_centrale", "zongo",       "INSCRIT"),
    ("biblio_centrale", "gnambre",     "INSCRIT"),

    ("biblio_centrale", "ouaga",       "APPARTIENT À"),
    ("biblio_univ", "bobo",            "EST SITUÉE À"),
    ("biblio_univ", "koudougou",       "EST SITUÉE À"),

    ("biblio_centrale", "rayon_b",     "POSSÈDE"),
    ("biblio_univ", "rayon_a",         "POSSÈDE"),

    ("zongo", "parachutage",           "EMPRUNTE"),
    ("gnambre", "sentiers",            "EMPRUNTE"),

    ("nazi_boni", "crepuscule",        "ÉCRIT"),
    ("norbert_zongo", "parachutage",   "ÉCRIT"),

    ("parachutage", "roman",           "APPARTIENT À"),
    ("parachutage", "histoire",        "APPARTIENT À"),
    ("sentiers", "editions_faso",      "EST PUBLIÉ PAR"),
    ("sentiers", "presses",            "EST PUBLIÉ PAR"),

    ("parachutage", "rayon_b",         "EST RANGÉ DANS"),
    ("sentiers", "rayon_a",            "EST RANGÉ DANS"),
]

# =================================================================
# 5. ARÊTES D'ONTOLOGIE (flèches grises pointillées "EST DE TYPE")
# =================================================================
ARETES_TYPE = [
    ("dr_arthur", "type_biblio_caire"),
    ("zongo", "type_personne"),
    ("zongo", "type_membre"),
    ("gnambre", "type_personne"),
    ("gnambre", "type_membre"),
    ("norbert_zongo", "type_auteur"),
    ("crepuscule", "type_auteur"),
    ("ouaga", "type_ville"),
    ("koudougou", "type_ville"),
    ("bobo", "type_ville"),
    ("biblio_centrale", "type_biblio"),
    ("biblio_univ", "type_biblio"),
    ("roman", "type_categorie"),
    ("histoire", "type_categorie"),
    ("editions_faso", "type_editeur"),
    ("presses", "type_editeur"),
    ("rayon_a", "type_rayon"),
    ("rayon_b", "type_rayon"),
]

# =================================================================
# 6. FONCTIONS DE DESSIN
# =================================================================
def dessiner_noeud(ax, cle):
    x, y, r, categorie, texte = NODES[cle]
    y = conv_y(y)
    couleur = COLOR_BY_TYPE[categorie]
    cercle = Circle(
        (x, y), r,
        facecolor="white", edgecolor=couleur, linewidth=3.0, zorder=3,
    )
    ax.add_patch(cercle)
    ax.text(
        x, y, texte,
        ha="center", va="center",
        fontsize=max(6.5, r / 6.5), fontweight="bold",
        linespacing=1.25, zorder=4,
    )


def point(cle):
    x, y, r, _, _ = NODES[cle]
    return x, conv_y(y), r


def dessiner_fleche(ax, cle_a, cle_b, label, couleur, style, rad=0.08):
    xa, ya, ra = point(cle_a)
    xb, yb, rb = point(cle_b)
    fleche = FancyArrowPatch(
        (xa, ya), (xb, yb),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0 if style == "metier" else 1.3,
        linestyle="-" if style == "metier" else (0, (5, 4)),
        color=couleur,
        shrinkA=ra + 4, shrinkB=rb + 4,
        zorder=2,
    )
    ax.add_patch(fleche)
    if label:
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(
            mx, my, label,
            fontsize=8.5 if style == "metier" else 8,
            color=couleur, ha="center", va="center",
            fontweight="bold" if style == "metier" else "normal",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5,
        )


# =================================================================
# 7. CONSTRUCTION DE LA FIGURE (même résolution que l'image d'origine)
# =================================================================
DPI = 100
fig, ax = plt.subplots(figsize=(LARGEUR_PX / DPI, HAUTEUR_PX / DPI), dpi=DPI)
ax.set_xlim(0, LARGEUR_PX)
ax.set_ylim(0, HAUTEUR_PX)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

for source, cible, label in ARETES_METIER:
    dessiner_fleche(ax, source, cible, label, COULEUR_FLECHE_METIER, style="metier")

for source, cible in ARETES_TYPE:
    dessiner_fleche(ax, source, cible, "EST DE TYPE", COULEUR_FLECHE_TYPE,
                     style="type", rad=0.15)

for cle in NODES:
    dessiner_noeud(ax, cle)

# =================================================================
# 8. LÉGENDE (boîte mesurée elle aussi sur l'image d'origine :
#    de x=1053 à x=1515, de y=690 à y=982 en coordonnées image)
# =================================================================
leg_x0, leg_y1 = 1053, conv_y(690)     # coin haut-gauche (en coord. matplotlib)
leg_x1, leg_y0 = 1515, conv_y(982)     # coin bas-droit
leg_w, leg_h = leg_x1 - leg_x0, leg_y1 - leg_y0

boite = mpatches.FancyBboxPatch(
    (leg_x0, leg_y0), leg_w, leg_h,
    boxstyle="round,pad=4",
    linewidth=2, edgecolor="#0E217B", facecolor="white", zorder=6,
)
ax.add_patch(boite)
ax.text(leg_x0 + leg_w / 2, leg_y0 + leg_h - 25, "LÉGENDE",
        ha="center", va="center", fontsize=15, fontweight="bold", zorder=7)

legende_categories = [
    ("Membre / Personne", COLOR_BY_TYPE["Membre"]),
    ("Auteur",            COLOR_BY_TYPE["Auteur"]),
    ("Livre",              COLOR_BY_TYPE["Livre"]),
    ("Éditeur",            COLOR_BY_TYPE["Éditeur"]),
    ("Catégorie",          COLOR_BY_TYPE["Catégorie"]),
    ("Rayon",              COLOR_BY_TYPE["Rayon"]),
    ("Bibliothèque",       COLOR_BY_TYPE["Bibliothèque"]),
    ("Ville",              COLOR_BY_TYPE["Ville"]),
    ("Bibliothécaire",     COLOR_BY_TYPE["Bibliothécaire"]),
]

y_cursor = leg_y0 + leg_h - 60
for texte, couleur in legende_categories:
    ax.add_patch(Circle((leg_x0 + 30, y_cursor), 12,
                         facecolor="white", edgecolor=couleur, linewidth=2.5, zorder=7))
    ax.text(leg_x0 + 55, y_cursor, texte, ha="left", va="center",
            fontsize=10.5, zorder=7)
    y_cursor -= 27

ax.annotate("", xy=(leg_x0 + 300, leg_y0 + leg_h - 60),
            xytext=(leg_x0 + 250, leg_y0 + leg_h - 60),
            arrowprops=dict(arrowstyle="-|>", color=COULEUR_FLECHE_METIER, linewidth=2.2),
            zorder=7)
ax.text(leg_x0 + 245, leg_y0 + leg_h - 78, "Relation fonctionnelle\n(orientée)",
        ha="left", va="top", fontsize=9, zorder=7)

ax.annotate("", xy=(leg_x0 + 300, leg_y0 + leg_h - 130),
            xytext=(leg_x0 + 250, leg_y0 + leg_h - 130),
            arrowprops=dict(arrowstyle="-|>", color=COULEUR_FLECHE_TYPE,
                             linewidth=1.5, linestyle=(0, (5, 4))),
            zorder=7)
ax.text(leg_x0 + 245, leg_y0 + leg_h - 148, "Relation de type\n(EST DE TYPE)",
        ha="left", va="top", fontsize=9, zorder=7)

# =================================================================
# 9. EXPORT (même résolution que l'image d'origine : 1536x1024)
# =================================================================
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("graphe_bibliotheque_final.png", dpi=DPI, facecolor="white")
print("Graphe généré : graphe_bibliotheque_final.png")

# Ouvre automatiquement l'image générée avec la visionneuse par défaut de
# Windows (équivalent d'un double-clic sur le fichier). Sans ces lignes,
# le script se contente d'ENREGISTRER le PNG sur le disque : matplotlib
# ne montre une fenêtre que si on appelle plt.show(), ce qu'on ne fait
# pas ici puisqu'on veut un fichier exporté, pas une fenêtre interactive.
import os
os.startfile("graphe_bibliotheque_final.png")
