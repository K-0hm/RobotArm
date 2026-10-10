import re

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import matplotlib.pyplot as plt

from math import sin, cos, atan, atan2, sqrt, pi, degrees, radians
import numpy as np

from robotABB import (
    mci_rapport,
    mcd_rapport,
    MGD_matriciel,
    MGD_vectoriel,
    MGI,
    MGI_meilleure,
    MCD,
    MCI,
    mgd_frames,
    dans_butees,
    bornes_xyz,
    balayage,
    nuage_grille,
    ecart_angle,
    controleur,
    deriver_trajectoire,
    singularites,
    validation_mgi,
    R_to_rpy,
    pose_vers_T,
    R_OUTIL_BAS,
    D1,
    PORTEE,
    Q_MIN,
    Q_MAX,
    TOLERANCE,
    QDOT_MAX,
    Trajectoire,
)

# =============================================================================
#  DESSIN 3D REALISTE  (volumes pleins avec ombrage)
# =============================================================================

# Drawing constants/
LUMIERE = np.array([0.35, -0.45, 0.82])
LUMIERE = LUMIERE / np.linalg.norm(LUMIERE)

ZSOL = -0.55          # hauteur du sol (m)

def _ajout(scene, pts, normale, couleur):
    """Ajoute un polygone a la scene, colore selon l'eclairage."""
    k = 0.62 + 0.55 * max(0.0, float(np.dot(normale, LUMIERE)))
    r, g, bl = to_rgb(couleur)
    scene[0].append(pts)
    scene[1].append((min(r * k, 1.0), min(g * k, 1.0), min(bl * k, 1.0), 1.0))


def _repere(d):
    """Deux vecteurs unitaires perpendiculaires a d."""
    t = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, t)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2


def cylindre(scene, p0, p1, rayon, couleur, n=24, nseg=1):
    """Cylindre plein entre les points p0 et p1."""
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    axe = p1 - p0
    long = np.linalg.norm(axe)
    if long < 1e-9:
        return
    d = axe / long
    e1, e2 = _repere(d)
    ang = np.linspace(0.0, 2.0 * np.pi, n + 1)
    for s in range(nseg):
        c0 = p0 + d * long * s / nseg
        c1 = p0 + d * long * (s + 1) / nseg
        for i in range(n):
            t0, t1 = ang[i], ang[i + 1]
            r0 = rayon * (np.cos(t0) * e1 + np.sin(t0) * e2)
            r1 = rayon * (np.cos(t1) * e1 + np.sin(t1) * e2)
            tm = 0.5 * (t0 + t1)
            nrm = np.cos(tm) * e1 + np.sin(tm) * e2
            _ajout(scene, [c0 + r0, c0 + r1, c1 + r1, c1 + r0], nrm, couleur)
    for c, nrm in ((p0, -d), (p1, d)):          # deux disques aux extremites
        pts = [c + rayon * (np.cos(t) * e1 + np.sin(t) * e2) for t in ang[:-1]]
        _ajout(scene, pts, nrm, couleur)


def boite(scene, centre, u, v, w, dims, couleur):
    """Parallelepipede centre en 'centre', axes u, v, w (unitaires)."""
    c = np.array(centre, float)
    axes = [(np.array(u, float), dims[0] / 2),
            (np.array(v, float), dims[1] / 2),
            (np.array(w, float), dims[2] / 2)]
    for i in range(3):
        for sgn in (-1, 1):
            ni, hi = axes[i]
            j, k = [x for x in range(3) if x != i]
            nj, hj = axes[j]
            nk, hk = axes[k]
            cf = c + sgn * hi * ni
            pts = [cf + hj * nj + hk * nk, cf - hj * nj + hk * nk,
                   cf - hj * nj - hk * nk, cf + hj * nj - hk * nk]
            _ajout(scene, pts, sgn * ni, couleur)


def sphere(scene, centre, rayon, couleur, nlat=8, nlon=14):
    c = np.array(centre, float)

    def pt(ph, th):
        return np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])

    for i in range(nlat):
        ph0, ph1 = np.pi * i / nlat, np.pi * (i + 1) / nlat
        for j in range(nlon):
            th0, th1 = 2 * np.pi * j / nlon, 2 * np.pi * (j + 1) / nlon
            ps = [pt(ph0, th0), pt(ph0, th1), pt(ph1, th1), pt(ph1, th0)]
            nrm = sum(ps)
            nrm = nrm / np.linalg.norm(nrm)
            _ajout(scene, [c + rayon * q for q in ps], nrm, couleur)


def cone(scene, p0, p1, r0, r1, couleur, n=28, nseg=1, caps=True):
    """Tronc de cone (bras effiles) entre p0 (rayon r0) et p1 (rayon r1)."""
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    axe = p1 - p0
    long = np.linalg.norm(axe)
    if long < 1e-9:
        return
    d = axe / long
    e1, e2 = _repere(d)
    ang = np.linspace(0.0, 2.0 * np.pi, n + 1)
    pente = (r0 - r1) / long
    for s in range(nseg):
        f0, f1 = s / nseg, (s + 1) / nseg
        c0, c1 = p0 + axe * f0, p0 + axe * f1
        ra, rb = r0 + (r1 - r0) * f0, r0 + (r1 - r0) * f1
        for i in range(n):
            t0, t1 = ang[i], ang[i + 1]
            tm = 0.5 * (t0 + t1)
            u0 = np.cos(t0) * e1 + np.sin(t0) * e2
            u1 = np.cos(t1) * e1 + np.sin(t1) * e2
            nrm = np.cos(tm) * e1 + np.sin(tm) * e2 + pente * d
            nrm = nrm / np.linalg.norm(nrm)
            _ajout(scene, [c0 + ra * u0, c0 + ra * u1, c1 + rb * u1, c1 + rb * u0], nrm, couleur)
    if caps:
        for c, r, nrm in ((p0, r0, -d), (p1, r1, d)):
            pts = [c + r * (np.cos(t) * e1 + np.sin(t) * e2) for t in ang[:-1]]
            _ajout(scene, pts, nrm, couleur)


# =============================================================================
#  IHM  (selon le schema :  [q1..q6] [robot 3D] [X Y Z + orientation] [MGD / MGI / Validation])
# =============================================================================

ZSOL = -0.01          # hauteur du sol (m) : le socle du robot est pose a z = 0


# points de position du carre (X  Y  Z), un par ligne : modifiables dans l'interface
# (carre dans le plan vertical XZ, devant le robot, outil vertical)
def _carre(n_cote=20, cx=0.45, cy=0.0, cz=0.40, cote=0.16):
    """Carre discretise : n_cote points par cote."""
    h = cote/2
    coins = [(cx-h, cz-h), (cx+h, cz-h), (cx+h, cz+h), (cx-h, cz+h)]
    lignes = []
    for i in range(4):
        x0, z0 = coins[i]
        x1, z1 = coins[(i+1) % 4]
        for k in range(n_cote):
            t = k/n_cote
            lignes.append("%6.3f  %6.3f  %6.3f"
                          % (x0 + t*(x1-x0), cy, z0 + t*(z1-z0)))
    lignes.append("%6.3f  %6.3f  %6.3f" % (coins[0][0], cy, coins[0][1]))
    return "\n".join(lignes) + "\n"


# Trajectory GUI presets
POINTS_DEFAUT = _carre(n_cote=5)

# points de position du cercle (X  Y  Z), un par ligne : modifiables dans l'interface
POINTS_CERCLE = "\n".join(
    "%6.3f  %6.3f  %.2f" % (0.45 + 0.08 * np.cos(2 * np.pi * k / 16),
                            0.0,
                            0.40 + 0.08 * np.sin(2 * np.pi * k / 16))
    for k in range(17)
) + "\n"

BG = "#f4f4f4"
ROUGE = "#c62828"
BLEU = "#1f4fbf"
VERT = "#2e7d32"
ORANGE_ABB = "#ff6a00"
# commande en vitesse : trajectoire pd(t) (cercle ou carre dans le plan vertical XZ)
CENTRE_TRAJ = (0.45, 0.0, 0.40)                 # centre de la figure (m)
TAILLE_TRAJ = {"Cercle": 0.08, "Carre": 0.16}   # rayon du cercle / cote du carre (m)
DUREE_TRAJ = 8.0                                # duree d'un tour (s)

NOMS_Q = ["q1  (deg)", "q2  (deg)", "q3  (deg)", "q4  (deg)", "q5  (deg)", "q6  (deg)"]
Q_INIT = [radians(0), radians(20), radians(10), radians(0), radians(55), radians(0)]


class Application(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("IHM  -  Robot ABB IRB 140  (6 ddl)")
        larg = min(1400, self.winfo_screenwidth())
        haut = min(1000, self.winfo_screenheight() - 70)
        self.geometry("%dx%d+0+0" % (larg, haut))
        self.minsize(900, 560)
        self.configure(bg=BG)
        # etat courant du robot
        self.q = list(Q_INIT)
        self.ax_vit = None
        self.ax_traj = None
        # variables liees aux champs de saisie
        self.var_q = [tk.StringVar() for _ in range(6)]
        self.var_p = [tk.StringVar() for _ in range(3)]
        self.var_o = [tk.StringVar() for _ in range(3)]

        # cases a cocher
        self.var_mgd = tk.BooleanVar(value=True)
        self.var_mgi = tk.BooleanVar(value=False)
        self.var_val = tk.BooleanVar(value=False)
        self.var_val_mci = tk.BooleanVar(value=False)
        self.var_espace = tk.BooleanVar(value=False)
        self.points_travail = None    # cache, calcule a la demande
        # solutions du MGI (jusqu'a 8)
        self.solutions = []
        # trajectoire : liste de points de position (X, Y, Z) suivis par la MGI
        self.apercu = None           # points lus dans la zone de texte
        self.en_cours = False        # True pendant le deplacement en petits pas
        self.apercu_ok = False       # True si tous les points sont atteignables
        self.num_lignes = []         # numero de ligne (zone de texte) de chaque point
        self.i_traj = 0              # prochain point a atteindre
        self.trace = []              # points deja parcourus par l'effecteur (trait vert)
        self.auto = False            # True pendant l'enchainement automatique
        self.compte_auto = 0
        self.total_auto = 0
        self.after_id_auto = None
        self.hist_temps  = []
        self.hist_erreur = []
        self.hist_vit_d  = []
        self.hist_vit_r  = []
        self.ax_err = None
        self.ax_vit = None
        self.canvas_graph = None
        # commande en vitesse
        self.qdot = [0.0] * 6
        self.k_cmd = 0
        self.after_id_cmd = None
        self.facteur_vitesse = tk.DoubleVar(value=1.0)   # x1.0 = vitesse nominale

        self._styles()
        self._construire()
        self.maj_etats()
        self.remplir_q()
        self.maj_apercu(dessiner=False)
        self.calculer()


    # ---------------- styles ----------------
    def _styles(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG)
        st.configure("TCheckbutton", background=BG, font=("TkDefaultFont", 11))
        for nom, couleur in [("Rouge", ROUGE), ("Bleu", BLEU)]:
            st.configure(nom + ".TLabelframe", background=BG,
                         bordercolor=couleur, borderwidth=2)
            st.configure(nom + ".TLabelframe.Label", background=BG,
                         foreground=couleur, font=("TkDefaultFont", 11, "bold"))
        st.configure("Champ.TEntry", padding=4)
        st.configure("Hors.TEntry", padding=4, foreground=ROUGE)
        st.map("Hors.TEntry", foreground=[("readonly", ROUGE), ("disabled", ROUGE)])

    # ---------------- construction ----------------
    def _construire(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1, minsize=300)

        # --- colonne 0 : q1 ... q6
        cadre_q = ttk.Labelframe(self, text=" Articulations ", style="Rouge.TLabelframe")
        cadre_q.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=(12, 6))
        self.lim_q = [(degrees(Q_MIN[i]), degrees(Q_MAX[i])) for i in range(6)]
        self.ent_q = []
        for i, nom in enumerate(NOMS_Q):
            ttk.Label(cadre_q, text=nom, foreground=ROUGE,
                      font=("TkDefaultFont", 9, "bold")).grid(
                row=i, column=0, sticky="w", padx=(8, 2), pady=1)
            e = ttk.Entry(cadre_q, textvariable=self.var_q[i], width=7,
                          justify="right", style="Champ.TEntry")
            e.grid(row=i, column=1, padx=2, pady=1)
            ttk.Label(cadre_q, text="[%.0f ; %.0f]" % self.lim_q[i], foreground="#777",
                      font=("TkDefaultFont", 8)).grid(row=i, column=2, sticky="w", padx=(2, 8))
            e.bind("<KeyRelease>", lambda ev: self.calculer())
            self.ent_q.append(e)

        ttk.Checkbutton(cadre_q, text="Espace de travail", variable=self.var_espace,
                        command=self.rafraichir).grid(
            row=6, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 4))

        # --- colonne 1 : vue 3D du robot
        cadre_3d = ttk.Frame(self)
        cadre_3d.grid(row=0, column=1, sticky="nsew", padx=6, pady=(12, 6))
        self.fig = Figure(figsize=(6, 5), facecolor="white")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=22, azim=-55)      # vue initiale (modifiable a la souris)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cadre_3d)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

                # --- ligne 1 : pose de l'effecteur et modes, cote a cote (horizontal)
        barre_h = ttk.Frame(self)
        barre_h.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 4))
        cadre_p = ttk.Labelframe(barre_h, text=" Pose de l'effecteur ", style="Bleu.TLabelframe")
        cadre_p.grid(row=0, column=0, sticky="ns")
        police = ("TkDefaultFont", 8, "bold")
        xmin, xmax, ymin, ymax, zmin, zmax = bornes_xyz()
        self.lim_p = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        self.ent_p, self.ent_o = [], []
        noms_p = ["X (m)", "Y (m)", "Z (m)"]
        noms_o = ["roulis (deg)", "tangage (deg)", "lacet (deg)"]
        for i in range(6):                          # 6 champs sur une seule ligne
            ttk.Label(cadre_p, text=(noms_p + noms_o)[i], foreground=BLEU, font=police).grid(
                row=0, column=2 * i, sticky="e", padx=(8, 2), pady=(2, 0))
            var = self.var_p[i] if i < 3 else self.var_o[i - 3]
            e = ttk.Entry(cadre_p, textvariable=var, width=7, justify="right", style="Champ.TEntry")
            e.grid(row=0, column=2 * i + 1, padx=(0, 4), pady=(2, 0))
            e.bind("<KeyRelease>", lambda ev: self.saisie_pose())
            (self.ent_p if i < 3 else self.ent_o).append(e)
            if i < 3:                               # bornes sous X, Y, Z
                ttk.Label(cadre_p, text="[%.2f ; %.2f]" % self.lim_p[i], foreground="#777",
                          font=("TkDefaultFont", 7)).grid(row=1, column=2 * i + 1, pady=(0, 2))

        # --- colonne 2 : les 3 graphes empiles a droite du robot
        cadre_g = ttk.Labelframe(self, text=" Erreur, vitesse et trajectoire ", style="Rouge.TLabelframe")
        cadre_g.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=(6, 12), pady=(12, 6))
        matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9})
        fig_g = Figure(figsize=(4.2, 4.5), facecolor="white")
        self.ax_err = fig_g.add_subplot(311)
        self.ax_vit = fig_g.add_subplot(312)
        self.ax_traj = fig_g.add_subplot(313)
        fig_g.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.09, hspace=0.8)
        self.canvas_graph = FigureCanvasTkAgg(fig_g, master=cadre_g)
        self.canvas_graph.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=2)
        self._dessiner_graphe()

        # --- trajectoire : points de position saisis en coordonnees X  Y  Z
        traj = ttk.Labelframe(self, text=" Trajectoire : points de position suivis par la MGI "
                                         "(outil maintenu vertical) ",
                              style="Bleu.TLabelframe")
        traj.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        cadre_txt = ttk.Frame(traj)
                # --- menu deroulant : modes (comme "Commande")
        ttk.Label(traj, text="Modes :", foreground=BLEU).grid(
            row=0, column=1, sticky="w", padx=(16, 4), pady=(6, 0))
        self.var_modes = tk.StringVar(value="MGD")
        menu_modes = ttk.Combobox(traj, textvariable=self.var_modes, state="readonly", width=15,
                                  values=["MGD", "MGI", "Validation MGI",
                                          "Validation MCI"])
        menu_modes.grid(row=0, column=2, padx=(0, 16), pady=(6, 0))
        menu_modes.bind("<<ComboboxSelected>>", self.changer_mode)
        self.txt_pts = tk.Text(cadre_txt, width=32, height=5, font="TkFixedFont",
                               relief="solid", bd=1, wrap="none")
        barre = ttk.Scrollbar(cadre_txt, orient="vertical", command=self.txt_pts.yview)
        self.txt_pts.configure(yscrollcommand=barre.set)
        self.txt_pts.pack(side="left")
        barre.pack(side="left", fill="y")
        self.txt_pts.insert("1.0", POINTS_DEFAUT)
        self.txt_pts.tag_configure("atteint", background="#c8e6c9")
        self.txt_pts.bind("<KeyRelease>", lambda ev: self.maj_apercu())
        # --- menu deroulant : forme de la trajectoire
        ttk.Label(traj, text="Trajectoire :", foreground=BLEU).grid(
            row=1, column=1, sticky="w", padx=(16, 4), pady=(2, 0))
        self.var_forme = tk.StringVar(value="Carre")
        menu_forme = ttk.Combobox(traj, textvariable=self.var_forme,
                                  values=["Carre", "Cercle"], state="readonly", width=9)
        menu_forme.grid(row=1, column=2, padx=(0, 16), pady=(2, 0))
        menu_forme.bind("<<ComboboxSelected>>", self.changer_forme)

        # --- menu deroulant : mode de commande
        ttk.Label(traj, text="Commande :", foreground=BLEU).grid(
            row=1, column=3, sticky="w", padx=(0, 4), pady=(2, 0))
        self.var_mode = tk.StringVar(value="Position")
        menu_mode = ttk.Combobox(traj, textvariable=self.var_mode,
                                 values=["Position", "Vitesse"], state="readonly", width=9)
        menu_mode.grid(row=1, column=4, padx=(0, 16), pady=(2, 0))

        # --- reglette : facteur de vitesse en direct (au-dessus de "Commande")
        ttk.Label(traj, text="Vitesse :", foreground=BLEU).grid(
            row=0, column=3, sticky="w", padx=(0, 4), pady=(6, 0))
        reglette = ttk.Scale(traj, from_=0.1, to=3.0, orient="horizontal",
                             variable=self.facteur_vitesse, length=110)
        reglette.grid(row=0, column=4, padx=(0, 4), pady=(6, 0))
        self.lbl_vitesse = ttk.Label(traj, text="x1.00", foreground="#777", width=6)
        self.lbl_vitesse.grid(row=0, column=5, sticky="w", pady=(6, 0))
        reglette.configure(command=lambda v: self.lbl_vitesse.config(text="x%.2f" % float(v)))


        # --- boutons d'execution (communs aux deux modes)
        ttk.Button(traj, text="Lancer", width=8, command=self.lancer).grid(
            row=1, column=5, sticky="w", padx=3, pady=(2, 0))
        ttk.Button(traj, text="Stop", width=8, command=self.stop_trajectoire).grid(
            row=1, column=6, sticky="w", padx=3, pady=(2, 0))
        ttk.Button(traj, text="Effacer trace", width=12, command=self.effacer_trace).grid(
            row=1, column=7, sticky="w", padx=(3, 6), pady=(2, 0))
        self.lbl_traj = ttk.Label(traj, text="", font=("TkDefaultFont", 10),
                                  wraplength=720, justify="left")
        

        # --- bas : messages
        bas = ttk.Labelframe(self, text=" Messages ")
        bas.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 8))
        gauche = ttk.Frame(bas)
        gauche.pack(side="left", anchor="nw", fill="x", expand=True)
        self.lbl_msg = ttk.Label(gauche, text="", font=("TkDefaultFont", 10))
        self.lbl_msg.pack(anchor="w", padx=10, pady=(4, 0))
        self.lbl_val = ttk.Label(gauche, text="", font=("TkDefaultFont", 10))
        self.lbl_val.pack(anchor="w", padx=10, pady=(1, 0))
        self.lbl_dh = ttk.Label(gauche, text="", font=("TkDefaultFont", 10))
        self.lbl_dh.pack(anchor="w", padx=10, pady=(1, 0))
        self.lbl_mci = ttk.Label(gauche, text="", font=("TkDefaultFont", 10))
        self.lbl_mci.pack(anchor="w", padx=10, pady=(1, 0))
        self.lbl_sing = ttk.Label(gauche, text="", font=("TkDefaultFont", 10))
        self.lbl_sing.pack(anchor="w", padx=10, pady=(1, 0))
        ttk.Label(gauche, foreground="#777",
                  text=("Butees (deg) :  q1 [-180 ; 180]   q2 [-90 ; 110]   q3 [-230 ; 50]   "
                        "q4 [-200 ; 200]   q5 [-115 ; 115]   q6 [-400 ; 400]")
                  ).pack(anchor="w", padx=10, pady=(1, 4))
        self.lbl_matrice = tk.Label(bas, text="", font=("TkFixedFont", 9),
                justify="left", bg=BG, anchor="nw")
        self.lbl_matrice.pack(side="right", anchor="ne", padx=10, pady=(4, 4))

    # ---------------- gestion des modes ----------------
    def changer_mode(self, event=None):
        """Menu Modes : MGD / MGI choisissent le sens du calcul, les autres choix
        ajoutent un affichage (validation ou nuage) en gardant le mode en cours."""
        choix = self.var_modes.get()
        if choix in ("MGD", "MGI"):
            self.var_mgd.set(choix == "MGD")
            self.var_mgi.set(choix == "MGI")
        self.var_val.set(choix == "Validation MGI")
        self.var_val_mci.set(choix == "Validation MCI")
        self.maj_etats()
        self.calculer()
    def clic_mgd(self):
        if self.var_mgd.get():
            self.var_mgi.set(False)
        self.maj_etats()
        self.calculer()

    def clic_mgi(self):
        if self.var_mgi.get():
            self.var_mgd.set(False)
        self.maj_etats()
        self.calculer()

    def clic_val(self):
        self.afficher_validation()

    def clic_val_mci(self):
        self.afficher_validation_mci()

    def maj_etats(self):
        """MGD coche : on saisit q.  MGI coche : on saisit X, Y, Z et l'orientation."""
        for e in self.ent_q:
            e.configure(state="normal" if self.var_mgd.get() else "readonly")
        for e in self.ent_p + self.ent_o:
            e.configure(state="normal" if self.var_mgi.get() else "readonly")

    def saisie_pose(self):
        """Une pose a ete modifiee a la main : on recalcule."""
        self.calculer()

    def message(self, texte, couleur="#333"):
        self.lbl_msg.config(text=texte, foreground=couleur)

    # ---------------- champs en rouge si hors [min ; max] ----------------
    def hors_plage(self, texte, lim):
        """True si le texte est un nombre en dehors de [min ; max]."""
        if lim is None:
            return False
        try:
            val = float(texte)
        except ValueError:
            return False
        return not (lim[0] - 1e-6 <= val <= lim[1] + 1e-6)

    def colorer_champs(self):
        for e, var, lim in zip(self.ent_q, self.var_q, self.lim_q):
            e.configure(style="Hors.TEntry" if self.hors_plage(var.get(), lim) else "Champ.TEntry")
        for e, var, lim in zip(self.ent_p, self.var_p, self.lim_p):
            e.configure(style="Hors.TEntry" if self.hors_plage(var.get(), lim) else "Champ.TEntry")

    # ---------------- remplissage des champs ----------------
    def remplir_q(self):
        for i in range(6):
            self.var_q[i].set("%.2f" % degrees(self.q[i]))

    def remplir_p(self, T):
        """T : matrice 4x4 de l'effecteur (position + orientation en roulis/tangage/lacet)."""
        for i in range(3):
            self.var_p[i].set("%.4f" % T[i, 3])
        r, p, y = R_to_rpy(T[:3, :3])
        for var, val in zip(self.var_o, (r, p, y)):
            var.set("%.2f" % degrees(val))

    def lire_pose(self):
        x, y, z = [float(v.get()) for v in self.var_p]
        r, p, w = [radians(float(v.get())) for v in self.var_o]
        return pose_vers_T(x, y, z, r, p, w)

    # ---------------- calcul ----------------
    def calculer(self):
        if self.en_cours:
            return
        self.colorer_champs()
        if self.var_mgd.get():
            try:
                q = [radians(float(v.get())) for v in self.var_q]
            except ValueError:
                self.message("MGD : saisie incomplete ou non numerique.", "#b26a00")
                return
            ok, msg = dans_butees(q)
            if not ok:
                self.message("MGD : " + msg, ROUGE)
                return
            self.q = q
            T = MGD_matriciel(q, retourner_T06=True)
            self.remplir_p(T)
            self.message("MGD :  q  ->  P6 = (%.4f ; %.4f ; %.4f)"
                         % (T[0, 3], T[1, 3], T[2, 3]), BLEU)

        elif self.var_mgi.get():
            try:
                T = self.lire_pose()
            except ValueError:
                self.message("MGI : saisie incomplete ou non numerique.", "#b26a00")
                return
            toutes = MGI(T)
            valides = [s for s in toutes if s['ok']]
            if not valides:
                raisons = sorted(set(s['raison'] for s in toutes))
                self.message("MGI : pose non atteignable  (%s)" % ", ".join(raisons), ROUGE)
                return
            # solution la plus proche de la configuration actuelle
            k = min(range(len(valides)),
                    key=lambda i: np.linalg.norm(np.array(valides[i]['q']) - np.array(self.q)))
            self.q = list(valides[k]['q'])
            self.remplir_q()
            self.message("MGI :  P6  ->  q = (%s) deg   [solution %d/%d : %s]"
                         % (" ; ".join("%.1f" % degrees(v) for v in self.q),
                            k + 1, len(valides), valides[k]['branche']), VERT)
        else:
            self.message("Cochez MGD ou MGI.", "#777")
            return

        self.colorer_champs()
        self.rafraichir()
        self.afficher_validation()
        self.afficher_comparaison_dh()
        self.afficher_matrice_dh()
        self.afficher_validation_mci()
        self.afficher_singularites()

    def afficher_validation(self):
        """Validation du MGI : q -> MGD -> MGI -> comparaison avec q."""
        if not self.var_val.get():
            self.lbl_val.config(text="")
            return
        ecarts, n = validation_mgi(self.q)
        if ecarts is None:
            self.lbl_val.config(text="Validation MGI : aucune solution trouvee", foreground=ROUGE)
            return
        ok = max(abs(e) for e in ecarts) < 1e-8
        self.lbl_val.config(
            text=("Validation MGI :  " + "   ".join("e%d = %.1e" % (i + 1, e) for i, e in enumerate(ecarts))
                  + "   ->   %s   (%d solutions valides pour cette pose)" % ("OK" if ok else "ECART", n)),
            foreground=VERT if ok else ROUGE)

    def afficher_comparaison_dh(self):
        """Compare MGD (vectoriel) et MGD_matriciel (Denavit-Hartenberg)."""
        p_vect = MGD_vectoriel(self.q)
        p_mat = MGD_matriciel(self.q)
        T_vect = MGD_vectoriel(self.q, retourner_T06=True)
        T_mat = MGD_matriciel(self.q, retourner_T06=True)
        ecart = max(abs(p_vect[i] - p_mat[i]) for i in range(3))
        ecart_T = float(np.max(np.abs(T_vect - T_mat)))
        self.lbl_dh.config(
            text=("MGD vectoriel  = (%.4f ; %.4f ; %.4f)     "
                  "MGD matriciel = (%.4f ; %.4f ; %.4f)     ecart position = %.2e     ecart T06 = %.2e"
                  % (p_vect[0], p_vect[1], p_vect[2],
                     p_mat[0], p_mat[1], p_mat[2], ecart, ecart_T)),
            foreground=VERT if ecart_T < 1e-9 else ROUGE)

    def afficher_matrice_dh(self):
        """Affiche la matrice T06 complete (methode matricielle)"""
        T06 = MGD_matriciel(self.q, retourner_T06=True)
        lignes = ["T06  =  (methode matricielle DH)"]
        for i in range(4):
            ligne = "  ".join("%8.4f" % T06[i, j] for j in range(4))
            lignes.append("   [ " + ligne + " ]")
        lignes.append("   -> 4e colonne = position P6 = (%.4f ; %.4f ; %.4f)"
                      % (T06[0, 3], T06[1, 3], T06[2, 3]))
        self.lbl_matrice.config(text="\n".join(lignes))

    def afficher_validation_mci(self):
        """Validation MCI (modele du rapport, poignet bloque) : V -> MCI -> qdot -> MCD -> V recalculee."""
        if not self.var_val_mci.get():
            self.lbl_mci.pack_forget()
            return
        self.lbl_mci.pack(anchor="w", padx=10, pady=(1, 0), after=self.lbl_msg)   # juste sous le message
        V = [0.10, 0.05, -0.05]                        # vitesse test de O6 (m/s) dans R0
        qdot, k, msg = mci_rapport(self.q, V)
        if qdot is None:
            self.lbl_mci.config(text="Validation MCI : MCI non definie (%s)" % msg, foreground=ROUGE)
            return
        Vr = mcd_rapport(self.q, qdot)
        ecart = max(abs(Vr[i] - k * V[i]) for i in range(3))
        ok = ecart < 1e-9
        self.lbl_mci.config(
            text=("Validation MCI :  V = (%.2f ; %.2f ; %.2f) m/s  ->  qdot = (%.1f ; %.1f ; %.1f) deg/s   (k = %.2f)\n"
                  "                  MCD(qdot) = (%.3f ; %.3f ; %.3f) m/s    ecart = %.1e   ->   %s"
                  % (V[0], V[1], V[2], degrees(qdot[0]), degrees(qdot[1]), degrees(qdot[2]), k,
                     Vr[0], Vr[1], Vr[2], ecart, "OK" if ok else "ECART")),
            foreground=VERT if ok else ROUGE)

    def afficher_singularites(self):
        liste, det, smin = singularites(self.q)
        if liste:
            self.lbl_sing.config(text="Singularite proche :  " + "  ;  ".join(liste)
                                 + "   (sigma_min = %.3f)" % smin, foreground=ROUGE)
        else:
            self.lbl_sing.config(text="Singularites : aucune   (sigma_min = %.3f   det J = %.2e)"
                                 % (smin, det), foreground=VERT)

    # ---------------- trajectoire : lecture et verification ----------------
    def lire_points(self):
        """Lit la zone de texte. Renvoie (points, numeros_de_ligne, message_erreur)."""
        pts, nums = [], []
        for n, ligne in enumerate(self.txt_pts.get("1.0", "end").split("\n"), 1):
            texte = ligne.strip()
            if texte == "" or texte.startswith("#"):
                continue
            texte = re.sub(r"\s*,\s+", " ", texte)        # "0.2, -1.65, 0.25" -> "0.2 -1.65 0.25"
            if ";" in texte:
                morceaux = [m.strip() for m in texte.split(";") if m.strip()]
            else:
                morceaux = texte.split()
                if len(morceaux) == 1 and texte.count(",") == 2:
                    morceaux = texte.split(",")
            morceaux = [m.replace(",", ".") for m in morceaux]   # virgule decimale acceptee
            if len(morceaux) != 3:
                return None, None, "Ligne %d : il faut 3 nombres  X  Y  Z." % n
            try:
                pts.append(tuple(float(m) for m in morceaux))
            except ValueError:
                return None, None, "Ligne %d : nombre non valide." % n
            nums.append(n)
        if not pts:
            return None, None, "Aucun point : ecrivez un point par ligne (X  Y  Z)."
        return pts, nums, None

    def q_pour_point(self, x, y, z, q_ref):
        """Meilleure solution MGI (outil vertical) la plus proche de q_ref, ou None."""
        T = np.eye(4)
        T[:3, :3] = R_OUTIL_BAS
        T[:3, 3] = [x, y, z]
        s = MGI_meilleure(T, q_ref)
        return None if s is None else list(s['q'])

    def verifier_points(self, pts, nums):
        """(True, "") si tous les points sont atteignables, sinon (False, message)."""
        q_ref = list(self.q)
        for (x, y, z), n in zip(pts, nums):
            q = self.q_pour_point(x, y, z, q_ref)
            if q is None:
                return False, ("Ligne %d (%.3f ; %.3f ; %.3f) hors espace de travail "
                               "(ou butees depassees) avec l'outil vertical" % (n, x, y, z))
            q_ref = q
        return True, ""

    def maj_apercu(self, dessiner=True):
        if self.en_cours:
            return
        """Relit les points quand on modifie le texte, et redessine le trajet prevu."""
        self.i_traj = 0
        self.txt_pts.tag_remove("atteint", "1.0", "end")
        pts, nums, err = self.lire_points()
        if err:
            self.apercu, self.apercu_ok, self.num_lignes = None, False, []
            self.lbl_traj.config(text=err, foreground="#b26a00")
        else:
            self.apercu, self.num_lignes = pts, nums
            self.apercu_ok, msg = self.verifier_points(pts, nums)
            if self.apercu_ok:
                self.lbl_traj.config(text="%d points, tous atteignables. "
                 "Cliquez sur \"Lancer la trajectoire\"." % len(pts), foreground=VERT)
            else:
                self.lbl_traj.config(text=msg, foreground=ROUGE)
        if dessiner:
            self.rafraichir()

    def point_suivant(self):
        """Calcule les q cibles avec la MGI, puis lance le deplacement petit a petit."""
        if self.en_cours:
            return
        if not self.apercu_ok or not self.apercu:
            self.message("Trajectoire : corrigez d'abord la liste de points (voir le cadre).", ROUGE)
            return
        pts = self.apercu
        if self.i_traj >= len(pts):
            self.i_traj = 0
        i = self.i_traj
        x, y, z = pts[i]
        q_cible = self.q_pour_point(x, y, z, self.q)
        if q_cible is None:
            self.message("Trajectoire : point %d hors espace de travail" % (i + 1), ROUGE)
            return
        self.en_cours = True
        self.deplacer_pas_a_pas(list(self.q), q_cible, 1, 10, i, x, y, z)

    def deplacer_pas_a_pas(self, q_depart, q_arrivee, k, n, i, x, y, z):
        """Un pas du deplacement : q avance de 1/n vers q_arrivee, puis on redessine."""
        t = k / n
        self.q = [q_depart[j] + t * (q_arrivee[j] - q_depart[j]) for j in range(6)]
        self.remplir_q()
        self.remplir_p(MGD_matriciel(self.q, retourner_T06=True))
        self.colorer_champs()
        self.rafraichir()
        if k < n:
            delai = max(1, int(30 / self.facteur_vitesse.get()))
            self.after(delai, lambda: self.deplacer_pas_a_pas(q_depart, q_arrivee, k + 1, n, i, x, y, z))
        else:
            self.terminer_deplacement(i, x, y, z)

    def terminer_deplacement(self, i, x, y, z):
        """Une fois les n pas effectues : mise a jour du texte et de la validation."""
        self.trace.append((x, y, z))
        pts = self.apercu
        self.txt_pts.tag_remove("atteint", "1.0", "end")
        ligne = self.num_lignes[i]
        self.txt_pts.tag_add("atteint", "%d.0" % ligne, "%d.end" % ligne)
        self.i_traj = i + 1
        fin = ""
        if self.i_traj >= len(pts):
            ferme = len(pts) > 1 and max(abs(pts[0][k] - pts[-1][k]) for k in range(3)) < 1e-9
            self.i_traj = 1 if ferme else 0
            fin = "   (fin du tour)"
        self.message("Trajectoire : point %d/%d   P6 = (%.3f ; %.3f ; %.3f)   q = (%s) deg%s"
                     % (i + 1, len(pts), x, y, z,
                        " ; ".join("%.0f" % degrees(v) for v in self.q), fin), BLEU)
        self.afficher_validation()
        self.afficher_comparaison_dh()
        self.afficher_matrice_dh()
        self.afficher_singularites()
        self.en_cours = False

        if self.auto:
            self.compte_auto += 1
            if self.compte_auto < self.total_auto:
                delai = max(1, int(150 / self.facteur_vitesse.get()))
                self.after_id_auto = self.after(delai, self.point_suivant)
            else:
                self.auto = False
                self.message("Trajectoire complete : %d points parcourus." % self.total_auto, VERT)

    def changer_forme(self, event=None):
        texte = POINTS_DEFAUT if self.var_forme.get() == "Carre" else POINTS_CERCLE
        self.charger_points(texte)

    def lancer(self):
        """Aiguille vers le mode Position (MGI point a point) ou Vitesse (commande cinematique)."""
        if self.var_mode.get() == "Position":
            self.lancer_trajectoire()
        else:
            self.lancer_commande()

    def lancer_trajectoire(self):
        """Lance tous les points de la liste, a la suite, en un seul clic."""
        if self.en_cours or self.auto:
            return
        if not self.apercu_ok or not self.apercu:
            self.message("Trajectoire : corrigez d'abord la liste de points (voir le cadre).", ROUGE)
            return
        self.auto = True
        self.compte_auto = 0
        self.total_auto = len(self.apercu)
        self.i_traj = 0
        self.point_suivant()

    def stop_trajectoire(self):
        """Arrete l'enchainement automatique. Le bras finit son pas en cours puis s'arrete."""
        if self.after_id_cmd is not None:
            self.after_cancel(self.after_id_cmd)
            self.after_id_cmd = None
        if self.after_id_auto is not None:
            self.after_cancel(self.after_id_auto)
            self.after_id_auto = None
        if self.auto:
            self.auto = False
            self.message("Trajectoire arretee. Le bras reste ou il est.", "#b26a00")

    def effacer_trace(self):
        """Efface le trait vert. Le bras ne bouge pas."""
        self.trace = []
        self.message("Trace effacee. Le bras reste a sa position actuelle.", "#777")
        self.rafraichir()

    def charger_points(self, texte):
        """Remplace la liste de points par 'texte' (carre ou cercle), sans bouger le bras."""
        if self.en_cours:
            return
        self.txt_pts.delete("1.0", "end")
        self.txt_pts.insert("1.0", texte)
        self.maj_apercu()

    def ouvrir_graphe(self):
        """Les courbes sont dans la fenetre principale : on les remet simplement a jour."""
        self._dessiner_graphe()

    def _dessiner_graphe(self):
        self.ax_err.clear()
        self.ax_err.plot(self.hist_temps, self.hist_erreur, color=ROUGE, lw=1.5)
        self.ax_err.set_ylabel("erreur |Pd - Pr|  (m)")
        self.ax_err.set_title("Erreur de position")
        self.ax_err.grid(True, alpha=0.3)

        self.ax_vit.clear()
        self.ax_vit.plot(self.hist_temps, self.hist_vit_d, "--", color=BLEU,
                         lw=1.3, label="vitesse desiree")
        self.ax_vit.plot(self.hist_temps, self.hist_vit_r, "-", color=VERT,
                         lw=1.5, label="vitesse reelle")
        self.ax_vit.set_xlabel("temps (s)")
        self.ax_vit.set_ylabel("|Pdot|  (m/s)")
        self.ax_vit.set_title("Suivi de vitesse")
        self.ax_vit.legend(loc="upper right", fontsize=8)
        self.ax_vit.grid(True, alpha=0.3)
        # trajectoire dans le plan XZ : desiree (trait epais) et reelle (se trace en direct)
        self.ax_traj.clear()
        tr = getattr(self, "traj_t", None)
        if tr is not None:
            pd = tr.echantillon(200) * 1000
            self.ax_traj.plot(pd[:, 0], pd[:, 2], color=BLEU, lw=4, alpha=0.3, label="desiree")
        if self.trace:
            pr = np.array(self.trace) * 1000
            self.ax_traj.plot(pr[:, 0], pr[:, 2], color=ROUGE, lw=1.5, label="reelle")
            self.ax_traj.plot(pr[-1, 0], pr[-1, 2], "o", color=ROUGE, ms=5)
        self.ax_traj.set_aspect("equal", adjustable="datalim")
        self.ax_traj.set_xlabel("x (mm)")
        self.ax_traj.set_ylabel("z (mm)")
        self.ax_traj.set_title("Trajectoire (plan XZ)")
        self.ax_traj.legend(loc="upper right", fontsize=8)
        self.ax_traj.grid(True, alpha=0.3)

        self.canvas_graph.draw_idle()

    def lancer_commande(self):
        """Mode Vitesse : suit pd(t) (loi quintique) avec qdot = J#(pd_dot + Kp (pd - p))."""
        if self.en_cours or self.auto:
            return
        forme = self.var_forme.get()
        self.traj_t = Trajectoire(forme.lower(), CENTRE_TRAJ, TAILLE_TRAJ[forme], DUREE_TRAJ,
                                  "XZ (vertical)", "quintique")
        q0 = self.q_pour_point(*self.traj_t.eval(0.0)[0], self.q)
        if q0 is None:
            self.message("Commande : point de depart hors espace de travail.", ROUGE)
            return
        self.q = q0
        self.t_sim = 0.0
        self.trace = []
        self.hist_temps, self.hist_erreur, self.hist_vit_d, self.hist_vit_r = [], [], [], []
        self.ouvrir_graphe()
        self.pas_commande(0.02, 15.0, n_sub=5)

    def pas_commande(self, dt, Kp, n_sub=5):
        """Un pas affiche = n_sub sous-pas d'integration ; le temps simule avance de dt.
        La reglette Vitesse ne change que la vitesse d'affichage, pas la trajectoire."""
        tr = self.traj_t
        if self.t_sim > tr.T + 1e-9:
            self.message("Commande terminee : tour fait en %.1f s   erreur max = %.2f mm"
                         % (tr.T, 1000 * max(self.hist_erreur)), VERT)
            return
        dts = dt / n_sub
        for j in range(n_sub):
            Pd, Pdot_d = tr.eval(min(self.t_sim + j * dts, tr.T))
            qdot_c, _ = controleur(self.q, Pd, Pdot_d, Kp)
            self.q = list(np.clip(np.array(self.q) + qdot_c * dts, Q_MIN, Q_MAX))
        self.t_sim += dt
        self.qdot = qdot_c

        T = MGD_matriciel(self.q, retourner_T06=True)
        p = T[:3, 3]
        self.trace.append(tuple(p))
        self.remplir_q()
        self.remplir_p(T)
        self.rafraichir()

        Pd, Pdot_d = tr.eval(min(self.t_sim, tr.T))
        err = np.array(Pd) - p
        Pdot = MCD(self.q, qdot_c)[:3]
        trop = [str(i + 1) for i in range(6) if abs(qdot_c[i]) > QDOT_MAX[i]]
        self.message("Commande  t = %.2f / %.1f s   |Pdot| = %.3f m/s   erreur = %.2f mm   "
                     "|qdot|max = %.0f deg/s%s"
                     % (self.t_sim, tr.T, np.linalg.norm(Pdot), 1000 * np.linalg.norm(err),
                        np.degrees(np.max(np.abs(qdot_c))),
                        ("   VITESSE MAX DEPASSEE (axe %s)" % ",".join(trop)) if trop else ""),
                     ROUGE if trop else BLEU)
        self.afficher_singularites()

        self.hist_temps.append(self.t_sim)
        self.hist_erreur.append(np.linalg.norm(err))
        self.hist_vit_d.append(np.linalg.norm(Pdot_d))
        self.hist_vit_r.append(np.linalg.norm(Pdot))
        self._dessiner_graphe()
        delai = max(1, int(dt * 1000 / self.facteur_vitesse.get()))
        self.after_id_cmd = self.after(delai, lambda: self.pas_commande(dt, Kp, n_sub))

    # ---------------- dessin du robot ----------------
    def rafraichir(self):
        fr = mgd_frames(self.q)
        O = [f[:3, 3] for f in fr]                 # O0 ... O6
        z1, z2, z4, z5, z6 = (fr[k][:3, 2] for k in (1, 2, 4, 5, 6))
        O1, O2, O3, O4, O5, O6 = O[1], O[2], O[3], O[4], O[5], O[6]
        p = O6

        # ---------- corps du robot (ABB IRB 140 : orange, carters gris fonce) ----------
        sc = ([], [])
        GRIS, GRIS2, ALU = "#2b3238", "#4a545c", "#b0bec5"
        ex = fr[1][:3, 0]                          # direction du decalage d'epaule (tourne avec q1)
        ez = np.array([0.0, 0.0, 1.0])
        ey0 = np.array([0.0, 1.0, 0.0])
        ex0 = np.array([1.0, 0.0, 0.0])
        # embase fixe : plaque + carter
        boite(sc, (0, 0, 0.012), ex0, ey0, ez, (0.30, 0.30, 0.024), GRIS)
        cone(sc, (0, 0, 0.024), (0, 0, 0.13), 0.125, 0.105, ORANGE_ABB, n=36)
        cylindre(sc, (0, 0, 0.13), (0, 0, 0.145), 0.108, GRIS2, n=36)
        # colonne tournante (axe 1) qui porte le decalage a1 vers l'epaule
        cone(sc, (0, 0, 0.145), (0.03 * ex[0], 0.03 * ex[1], D1 - 0.03), 0.100, 0.085, ORANGE_ABB, n=32)
        cone(sc, (0.03 * ex[0], 0.03 * ex[1], D1 - 0.03), O1 - ez * 0.0, 0.085, 0.07, ORANGE_ABB, n=28)
        # moteur d'epaule (axe 2) : carter + capots
        cylindre(sc, O1 - z1 * 0.065, O1 + z1 * 0.065, 0.078, ORANGE_ABB, n=32)
        cylindre(sc, O1 + z1 * 0.065, O1 + z1 * 0.085, 0.062, GRIS, n=28)
        cylindre(sc, O1 - z1 * 0.085, O1 - z1 * 0.065, 0.062, GRIS, n=28)
        # bras superieur effile (a2)
        cone(sc, O1, O2, 0.074, 0.056, ORANGE_ABB, n=28, nseg=3)
        # coude (axe 3)
        cylindre(sc, O2 - z2 * 0.058, O2 + z2 * 0.058, 0.060, ORANGE_ABB, n=28)
        cylindre(sc, O2 + z2 * 0.058, O2 + z2 * 0.075, 0.046, GRIS, n=24)
        cylindre(sc, O2 - z2 * 0.075, O2 - z2 * 0.058, 0.046, GRIS, n=24)
        # avant-bras (axe 4) : carter arriere + corps effile
        d4v = (O4 - O3) / max(np.linalg.norm(O4 - O3), 1e-9)
        cylindre(sc, O3 - d4v * 0.06, O3 + d4v * 0.04, 0.056, ORANGE_ABB, n=28)
        cone(sc, O3 + d4v * 0.04, O4 - d4v * 0.03, 0.054, 0.040, ORANGE_ABB, n=28, nseg=2)
        # poignet spherique (axes 5 et 6 concourants en O4 = O5)
        sphere(sc, O4, 0.046, GRIS2, nlat=8, nlon=16)
        cylindre(sc, O4 - z4 * 0.048, O4 + z4 * 0.048, 0.040, ORANGE_ABB, n=24)
        cone(sc, O5, O6 - z6 * 0.010, 0.036, 0.030, ORANGE_ABB, n=24)
        # bride (axe 6) + outil : pince a deux doigts
        cylindre(sc, O6 - z6 * 0.012, O6, 0.034, ALU, n=24)
        x6, y6 = fr[6][:3, 0], fr[6][:3, 1]
        boite(sc, O6 + z6 * 0.030, x6, y6, z6, (0.070, 0.036, 0.036), GRIS)
        for sg in (-1, 1):
            boite(sc, O6 + z6 * 0.075 + sg * x6 * 0.026, x6, y6, z6, (0.010, 0.020, 0.060), ALU)
        sphere(sc, O6 + z6 * 0.11, 0.008, "#ff9800", nlat=5, nlon=8)     # bout de l'outil

        # ombre portee sur le sol (projection le long de la lumiere)
        ombre = []
        for poly in sc[0]:
            q_ = np.array(poly, float)
            q_ = q_ - np.outer((q_[:, 2] - ZSOL) / LUMIERE[2], LUMIERE)
            q_[:, 2] = ZSOL + 0.001
            ombre.append(q_)

        # ---------- scene ----------
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_axis_off()
        self.ax.computed_zorder = False           # ordre de dessin impose par zorder

        S = 0.95
        # sol (disque avec cercles et rayons)
        th = np.linspace(0, 2 * np.pi, 90)
        disque = [(S * np.cos(t), S * np.sin(t), ZSOL) for t in th]
        sol = Poly3DCollection([disque], facecolors="#eceff1",
                               edgecolors="#b0bec5", linewidths=1.0)
        sol.set_zorder(0)
        sol.set_clip_on(False)
        self.ax.add_collection3d(sol)
        traits = []
        for r in (0.25, 0.50, 0.75):
            traits.append([(r * np.cos(t), r * np.sin(t), ZSOL) for t in th])
        for k in range(12):
            t = k * np.pi / 6
            traits.append([(rr * np.cos(t), rr * np.sin(t), ZSOL)
                           for rr in np.linspace(0, S, len(th))])
        grille = Line3DCollection(traits, colors="#cfd8dc", linewidths=0.6)
        grille.set_zorder(1)
        grille.set_clip_on(False)
        self.ax.add_collection3d(grille)

        # cercle de portee maximale
        self.ax.plot(PORTEE * np.cos(th), PORTEE * np.sin(th), [ZSOL] * len(th),
                     "--", color="#90a4ae", lw=1.0, zorder=1, clip_on=False)

        # projection de P6 sur le sol
        self.ax.plot([p[0], p[0]], [p[1], p[1]], [ZSOL, p[2]], ":",
                     color="#f57c00", lw=1.3, zorder=1, clip_on=False)
        self.ax.plot([p[0]], [p[1]], [ZSOL], "o", color="#f57c00", ms=5, zorder=1, clip_on=False)

        # petit repere X Y Z dans un coin
        o = np.array([-0.95, -0.95, ZSOL])
        for vec, col, nom in ((np.array([1, 0, 0]), "#c62828", "X"),
                              (np.array([0, 1, 0]), "#2e7d32", "Y"),
                              (np.array([0, 0, 1]), "#1f4fbf", "Z")):
            fin_ = o + 0.25 * vec
            self.ax.plot([o[0], fin_[0]], [o[1], fin_[1]], [o[2], fin_[2]],
                         color=col, lw=2, zorder=1, clip_on=False)
            self.ax.text(fin_[0], fin_[1], fin_[2], nom, color=col,
                         fontsize=9, fontweight="bold", zorder=1)

        # ombre au sol
        om = Poly3DCollection(ombre, facecolors=(0.2, 0.25, 0.3, 0.16), edgecolors="none")
        om.set_zorder(2)
        om.set_clip_on(False)
        self.ax.add_collection3d(om)

        # robot
        robot = Poly3DCollection(sc[0], facecolors=sc[1], edgecolors=sc[1], linewidths=0.15)
        robot.set_zorder(5)
        robot.set_clip_on(False)
        self.ax.add_collection3d(robot)

        # repere de l'outil (X rouge, Y vert, Z bleu)
        for k, col in enumerate(("#c62828", "#2e7d32", "#1f4fbf")):
            v = fr[6][:3, k] * 0.12
            self.ax.plot([O6[0], O6[0] + v[0]], [O6[1], O6[1] + v[1]], [O6[2], O6[2] + v[2]],
                         color=col, lw=2, zorder=8, clip_on=False)

        # nuage de points : espace de travail atteignable (grille de 10 deg, poignet bloque)
        if self.var_espace.get():
            if self.points_travail is None:
                self.points_travail = nuage_grille(10.0)      # 21 924 points, calcule une seule fois
            pts = self.points_travail
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=pts[:, 2], cmap="viridis",
                    alpha=0.8, linewidths=0, zorder=2, clip_on=False)

        # trace de la trajectoire prevue (carre, cercle...) sur l'espace de travail
        if self.apercu:
            pts = np.array(self.apercu)
            couleur = "#6a1b9a" if self.apercu_ok else ROUGE
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "--", color=couleur, lw=1.6,
                         zorder=6, clip_on=False)
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o", color=couleur, mfc="white",
                         mew=1.3, ms=5, zorder=6, clip_on=False)

        # trace deja parcourue par le bras (trait vert plein)
        if self.trace:
            tr = np.array(self.trace)
            self.ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], "-o", color="#00897b", lw=2.6,
                         ms=5, zorder=7, clip_on=False)
        haut = 1.2 if self.var_espace.get() else 0.95    # le nuage monte jusqu'a z = 1.157 m
        self.ax.set_xlim(-S, S)
        self.ax.set_ylim(-S, S)
        self.ax.set_zlim(ZSOL, haut)
        self.ax.set_box_aspect((2 * S, 2 * S, haut - ZSOL), zoom=1.75)
        self.canvas.draw()


if __name__ == "__main__":
    Application().mainloop()