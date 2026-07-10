"""
Génération du graphe de connaissances multi-relationnel
Domaine : Bibliothèque 
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

import matplotlib
matplotlib.use("Agg")

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
    ("dr_arthur", "biblio_centrale",   "TRAVAILLE DANS", -0.12),
    ("dr_arthur", "biblio_univ",       "TRAVAILLE DANS", 0.12),

    ("biblio_centrale", "zongo",       "INSCRIT", -0.06),
    ("biblio_centrale", "gnambre",     "INSCRIT", 0.10),

    ("biblio_centrale", "ouaga",       "APPARTIENT À", 0.10),
    ("biblio_univ", "bobo",            "EST SITUÉE À", -0.12),
    ("biblio_univ", "koudougou",       "EST SITUÉE À", 0.18),

    ("biblio_centrale", "rayon_b",     "POSSÈDE", -0.05),

    ("zongo", "parachutage",           "EMPRUNTE", -0.10),
    ("gnambre", "sentiers",            "EMPRUNTE", -0.10),

    ("nazi_boni", "crepuscule",        "ÉCRIT", -0.08),
    ("norbert_zongo", "parachutage",   "ÉCRIT", 0.18),

    # --- CORRIGÉ : Le Parachutage va avec Roman/Les Éditions du Faso,
    #     Les Sentiers d'Abraham va avec Histoire/Presses Univ. de Ouaga
    #     (dans la v5, ces 4 arêtes étaient mal réparties : les deux
    #     partaient de "parachutage" et les deux autres de "sentiers")
    ("parachutage", "roman",           "APPARTIENT À", 0.12),
    ("sentiers", "histoire",           "APPARTIENT À", 0.12),
    ("parachutage", "editions_faso",   "EST PUBLIÉ PAR", -0.12),
    ("sentiers", "presses",            "EST PUBLIÉ PAR", -0.12),

    ("parachutage", "rayon_b",         "EST RANGÉ DANS", 0.08),
    ("sentiers", "rayon_a",            "EST RANGÉ DANS", -0.08),
]

# =================================================================
# 5. ARÊTES D'ONTOLOGIE (flèches grises pointillées "EST DE TYPE")
# =================================================================
ARETES_TYPE = [
    ("dr_arthur", "type_biblio_caire", 0.15),
    ("zongo", "type_personne", -0.10),
    ("zongo", "type_membre", 0.20),
    ("gnambre", "type_personne", -0.10),
    ("gnambre", "type_membre", 0.20),
    ("norbert_zongo", "type_auteur", 0.15),
    ("crepuscule", "type_auteur", -0.15),
    ("ouaga", "type_ville", 0.15),
    ("koudougou", "type_ville", -0.10),
    ("bobo", "type_ville", 0.20),
    ("biblio_centrale", "type_biblio", 0.10),
    ("biblio_univ", "type_biblio", -0.15),
    ("roman", "type_categorie", -0.15),
    ("histoire", "type_categorie", 0.15),
    ("editions_faso", "type_editeur", -0.15),
    ("presses", "type_editeur", 0.15),
    ("rayon_a", "type_rayon", -0.15),
    ("rayon_b", "type_rayon", 0.15),
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
        # Le VRAI milieu d'une courbe "arc3,rad=r" n'est pas le milieu
        # du segment droit : on le calcule à partir de la formule de
        # la courbe de Bézier quadratique utilisée par matplotlib pour
        # ce connectionstyle. Cela permet aux libellés de deux arêtes
        # courbées en sens opposé de ne plus se superposer.
        mx_droit, my_droit = (xa + xb) / 2, (ya + yb) / 2
        dx, dy = xb - xa, yb - ya
        # vecteur perpendiculaire normalisé
        norme = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        perp_x, perp_y = -dy / norme, dx / norme
        decalage = rad * norme * 0.5
        mx = mx_droit + perp_x * decalage
        my = my_droit + perp_y * decalage
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

for source, cible, label, rad in ARETES_METIER:
    dessiner_fleche(ax, source, cible, label, COULEUR_FLECHE_METIER, style="metier", rad=rad)

for source, cible, rad in ARETES_TYPE:
    dessiner_fleche(ax, source, cible, "EST DE TYPE", COULEUR_FLECHE_TYPE,
                     style="type", rad=rad)

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
plt.close(fig)   # libère complètement les ressources matplotlib avant d'ouvrir Tkinter
print("Graphe généré : graphe_bibliotheque_final.png")

# Affiche l'image générée dans une fenêtre TKINTER (et non la
# visionneuse Windows par défaut). Sans ce bloc, le script se
# contente d'ENREGISTRER le PNG sur le disque : matplotlib
# n'affiche une fenêtre que si on appelle plt.show(), ce qu'on ne
# fait pas ici puisqu'on veut d'abord exporter un fichier propre,
# puis l'afficher nous-mêmes avec Tkinter (bibliothèque graphique
# standard de Python, incluse par défaut).
# --------------------------------------------------------------
# Affichage dans une fenêtre TKINTER (au lieu d'ouvrir la
# visionneuse Photos de Windows).
#
# NOTE IMPORTANTE : l'erreur "_tkinter.TclError: image "pyimageXX"
# doesn't exist" que tu as rencontrée venait d'un conflit entre DEUX
# interpréteurs Tk différents : celui créé en coulisses par le
# backend "TkAgg" de matplotlib, et celui qu'on crée nous-mêmes
# juste ici (fenetre = tk.Tk()). On a réglé ça de deux façons :
#   1) matplotlib.use("Agg") tout en haut du fichier, pour empêcher
#      matplotlib de créer le moindre interpréteur Tk caché ;
#   2) master=fenetre passé explicitement à tk.PhotoImage, pour
#      garantir que l'image est bien créée dans NOTRE interpréteur.
# --------------------------------------------------------------
import tkinter as tk
from PIL import Image

fenetre = tk.Tk()
fenetre.title("Graphe de connaissances — Domaine Bibliothèque")

# On redimensionne l'image AVANT affichage si elle est plus grande
# que l'écran (avec Pillow, juste pour le redimensionnement), puis
# on l'enregistre dans un fichier temporaire que Tkinter rechargera
# lui-même avec son propre PhotoImage (donc plus d'objet Pillow
# transmis directement à Tkinter -> plus de conflit possible).
image_pil = Image.open("graphe_bibliotheque_final.png")

ecran_w = fenetre.winfo_screenwidth()
ecran_h = fenetre.winfo_screenheight()
marge = 100
chemin_affiche = "graphe_bibliotheque_final.png"
if image_pil.width > ecran_w - marge or image_pil.height > ecran_h - marge:
    ratio = min((ecran_w - marge) / image_pil.width, (ecran_h - marge) / image_pil.height)
    nouvelle_taille = (int(image_pil.width * ratio), int(image_pil.height * ratio))
    image_pil = image_pil.resize(nouvelle_taille, Image.LANCZOS)
    chemin_affiche = "graphe_bibliotheque_final_redimensionne.png"
    image_pil.save(chemin_affiche)

photo = tk.PhotoImage(file=chemin_affiche, master=fenetre)

label_image = tk.Label(fenetre, image=photo)
label_image.image = photo   # garde une référence pour éviter le garbage collector
label_image.pack()

fenetre.mainloop()
