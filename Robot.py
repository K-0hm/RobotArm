import re
from math import sin, cos, atan, atan2, sqrt, pi, degrees, radians

import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


# setup
PI = 3.14159265358979323846
b = 0.250  # m
a = 1.150  # m


def MGD(q1, q2, q3, q4):
    x =  (a + sin(q2)*(q4+b))*sin(q1)
    y = -(a + sin(q2)*(q4+b))*cos(q1)
    z =  (q4+b)*cos(q2)
    return [x, y, z]


def MGI(x, y, z):
    R = sqrt(x*x + y*y)
    u = R - a

    q = [0, 0, 0, 0]
    q[0] = atan2(x, -y)
    q[1] = atan2(u, z)
    q[2] = 0
    q[3] = sqrt(u*u + z*z) - b
    return q

def deriver_trajectoire(pts, dt):
    """Vitesse desiree Pdot le long de la trajectoire (differences finies)."""
    P = np.array(pts, float)
    V = np.zeros_like(P)
    V[1:-1] = (P[2:] - P[:-2]) / (2*dt)      # differences centrees
    V[0]    = (P[1] - P[0]) / dt
    V[-1]   = (P[-1] - P[-2]) / dt
    return V

def jacobienne(q1, q2, q3, q4):
    """Matrice jacobienne 3x4 : Pdot = J . qdot"""
    R = a + sin(q2)*(q4+b)
    L = q4 + b
    s1, c1 = sin(q1), cos(q1)
    s2, c2 = sin(q2), cos(q2)

    J = np.zeros((3, 4))

    # colonne 1 : d/dq1
    J[0, 0] =  R*c1
    J[1, 0] =  R*s1
    J[2, 0] =  0.0

    # colonne 2 : d/dq2
    J[0, 1] =  L*c2*s1
    J[1, 1] = -L*c2*c1
    J[2, 1] = -L*s2

    # colonne 3 : d/dq3  ->  nulle (q3 n'agit pas sur la position)
    J[0, 2] = 0.0
    J[1, 2] = 0.0
    J[2, 2] = 0.0

    # colonne 4 : d/dq4
    J[0, 3] =  s2*s1
    J[1, 3] = -s2*c1
    J[2, 3] =  c2

    return J

def MCD(q1, q2, q3, q4, qdot):
    """Modele Cinematique Direct : Pdot = J . qdot"""
    J = jacobienne(q1, q2, q3, q4)
    return J @ np.array(qdot)


def MCI(q1, q2, q3, q4, Pdot):
    """Modele Cinematique Inverse : qdot = J+ . Pdot"""
    J = jacobienne(q1, q2, q3, q4)
    return np.linalg.pinv(J) @ np.array(Pdot)

def controleur(q, Pd, Pdot_d, Kp):
    """qdot_c = J+ . ( Pdot_d + Kp*(Pd - Pr) )"""
    Pr = np.array(MGD(q[0], q[1], q[2], q[3]))    # position reconstruite
    erreur = np.array(Pd) - Pr                     # Pd - Pr
    Pdot_c = np.array(Pdot_d) + Kp * erreur        # vitesse corrigee
    qdot_c = MCI(q[0], q[1], q[2], q[3], Pdot_c)   # -> vitesses moteur
    return qdot_c, erreur 

def verification(q1, q2, q3, q4):
    p = MGD(q1, q2, q3, q4)
    qnew = MGI(p[0], p[1], p[2])

    e1 = qnew[0] - q1
    e2 = qnew[1] - q2
    e3 = qnew[2] - q3
    e4 = qnew[3] - q4

    print("e1=%f  e2=%f  e3=%f  e4=%f" % (e1, e2, e3, e4))


def balayage(inc):
    # determination de l'espace de travail du robot
    q1 = PI
    while q1 > -PI:
        q2 = 0
        while q2 <= PI*0.75:      # 0.75*PI = 3PI/4
            q4 = 0
            while q4 < 0.350:
                pointy = MGD(q1, q2, 0, q4)
                print("x=%f,y=%f,z=%f" % (pointy[0], pointy[1], pointy[2]))
                q4 += 0.005
            q2 += inc
        q1 -= inc


# =============================================================================
#  OUTILS POUR L'IHM
# =============================================================================

# butees articulaires
Q1_MIN, Q1_MAX = -pi, pi
Q2_MIN, Q2_MAX = 0.0, 0.75 * pi
Q4_MIN, Q4_MAX = 0.0, 0.350

TOLERANCE = 1e-9


def dans_butees(q1, q2, q4):
    """Verifie que la configuration respecte les butees mecaniques."""
    if not (Q1_MIN - 1e-6 <= q1 <= Q1_MAX + 1e-6):
        return False, "q1 = %.1f deg hors de [%.0f ; %.0f] deg" % (
            degrees(q1), degrees(Q1_MIN), degrees(Q1_MAX))
    if not (Q2_MIN - 1e-6 <= q2 <= Q2_MAX + 1e-6):
        return False, "q2 = %.1f deg hors de [%.0f ; %.0f] deg" % (
            degrees(q2), degrees(Q2_MIN), degrees(Q2_MAX))
    if not (Q4_MIN - 1e-6 <= q4 <= Q4_MAX + 1e-6):
        return False, "q4 = %.3f hors de [%.3f ; %.3f]" % (q4, Q4_MIN, Q4_MAX)
    return True, ""


def ecart_angle(e):
    """Ramene un ecart angulaire dans [-pi ; pi]."""
    return atan2(sin(e), cos(e))


def bornes_xyz():
    """Min et max de X, Y, Z sur tout l'espace de travail (balayage des butees)."""
    q1 = np.linspace(Q1_MIN, Q1_MAX, 361)[:, None, None]
    q2 = np.linspace(Q2_MIN, Q2_MAX, 136)[None, :, None]
    q4 = np.linspace(Q4_MIN, Q4_MAX, 36)[None, None, :]
    R = a + np.sin(q2) * (q4 + b)
    x = R * np.sin(q1)
    y = -R * np.cos(q1)
    z = (q4 + b) * np.cos(q2)
    return x.min(), x.max(), y.min(), y.max(), z.min(), z.max()


def positions_corps(q1, q2, q4):
    """Points O0, O2, O3, P3 dans le repere de base, pour le dessin."""
    y1 = (-sin(q1), cos(q1), 0.0)
    z1 = (0.0, 0.0, 1.0)
    y2 = (-sin(q2) * y1[0] + cos(q2) * z1[0],
          -sin(q2) * y1[1] + cos(q2) * z1[1],
          -sin(q2) * y1[2] + cos(q2) * z1[2])
    O0 = (0.0, 0.0, 0.0)
    O2 = (-a * y1[0], -a * y1[1], -a * y1[2])
    O3 = (O2[0] + q4 * y2[0], O2[1] + q4 * y2[1], O2[2] + q4 * y2[2])
    P3 = (O3[0] + b * y2[0], O3[1] + b * y2[1], O3[2] + b * y2[2])
    return [O0, O2, O3, P3]


# =============================================================================
#  DESSIN 3D REALISTE  (volumes pleins avec ombrage)
# =============================================================================
LUMIERE = np.array([0.35, -0.45, 0.82])
LUMIERE = LUMIERE / np.linalg.norm(LUMIERE)

ZSOL = -0.55          # hauteur du sol (m)


def _ajout(scene, pts, normale, couleur):
    """Ajoute un polygone a la scene, colore selon l'eclairage."""
    k = 0.40 + 0.60 * max(0.0, float(np.dot(normale, LUMIERE)))
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


# =============================================================================
#  IHM  (selon le schema :  [q1..q4] [robot 3D] [X Y Z] [MGD / MGI / Validation])
# =============================================================================
# points de position du carre (X  Y  Z), un par ligne : modifiables dans l'interface
def _carre(n_cote=20, cx=0.0, cy=-1.45, cz=0.25, cote=0.40):
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

POINTS_DEFAUT = _carre(n_cote=5)

# points de position du cercle (X  Y  Z), un par ligne : modifiables dans l'interface
POINTS_CERCLE = "\n".join(
    "%6.3f  %6.3f  %.2f" % (0.20 * np.cos(2 * np.pi * k / 16),
                            -1.45 + 0.20 * np.sin(2 * np.pi * k / 16),
                            0.25)
    for k in range(17)
) + "\n"

BG = "#f4f4f4"
ROUGE = "#c62828"
BLEU = "#1f4fbf"
VERT = "#2e7d32"


class Application(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("IHM  -  Robot porte-outil 4 ddl")
        self.geometry("1320x830")
        self.minsize(1000, 700)
        self.configure(bg=BG)

        # etat courant du robot
        self.q = [radians(30), radians(60), 0.0, 0.350]

        # variables liees aux champs de saisie
        self.var_q = [tk.StringVar() for _ in range(4)]
        self.var_p = [tk.StringVar() for _ in range(3)]

        # cases a cocher
        self.var_mgd = tk.BooleanVar(value=True)
        self.var_mgi = tk.BooleanVar(value=False)
        self.var_val = tk.BooleanVar(value=False)

        # trajectoire : liste de points de position (X, Y, Z) suivis par la MGI
        self.apercu = None           # points lus dans la zone de texte
        self.en_cours = False        # True pendant le deplacement en petits pas
        self.apercu_ok = False       # True si tous les points sont atteignables
        self.num_lignes = []         # numero de ligne (zone de texte) de chaque point
        self.i_traj = 0              # prochain point a atteindre
        self.trace = []              # points deja parcourus par P3 (trait vert)
        self.auto = False            # True pendant l'enchainement automatique
        self.compte_auto = 0
        self.total_auto = 0
        self.after_id_auto = None
        # commande en vitesse
        self.qdot = [0.0, 0.0, 0.0, 0.0]
        self.k_cmd = 0
        self.after_id_cmd = None

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
        self.rowconfigure(0, weight=1)

        # --- colonne 0 : q1 q2 q3 q4
        cadre_q = ttk.Labelframe(self, text=" Articulations ", style="Rouge.TLabelframe")
        cadre_q.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=(12, 6))
        plages_q = ["min : %.0f\nmax : %.0f" % (degrees(Q1_MIN), degrees(Q1_MAX)),
                    "min : %.0f\nmax : %.0f" % (degrees(Q2_MIN), degrees(Q2_MAX)),
                    "libre\n(sans butee)",
                    "min : %.3f\nmax : %.3f" % (Q4_MIN, Q4_MAX)]
        self.lim_q = [(degrees(Q1_MIN), degrees(Q1_MAX)), (degrees(Q2_MIN), degrees(Q2_MAX)),
                      None, (Q4_MIN, Q4_MAX)]
        self.ent_q = []
        for i, nom in enumerate(["q1  (deg)", "q2  (deg)", "q3  (deg)", "q4  (m)"]):
            ttk.Label(cadre_q, text=nom, foreground=ROUGE,
                      font=("TkDefaultFont", 11, "bold")).grid(
                row=2 * i, column=0, sticky="w", padx=12, pady=(14, 0))
            e = ttk.Entry(cadre_q, textvariable=self.var_q[i], width=11,
                          justify="right", style="Champ.TEntry")
            e.grid(row=2 * i + 1, column=0, padx=12, pady=(2, 4))
            ttk.Label(cadre_q, text=plages_q[i], foreground="#777", justify="left",
                      font=("TkDefaultFont", 9)).grid(
                row=2 * i + 1, column=1, sticky="w", padx=(0, 10))
            e.bind("<KeyRelease>", lambda ev: self.calculer())
            self.ent_q.append(e)

        # --- colonne 1 : vue 3D du robot
        cadre_3d = ttk.Frame(self)
        cadre_3d.grid(row=0, column=1, sticky="nsew", padx=6, pady=(12, 6))
        self.fig = Figure(figsize=(6, 5), facecolor="white")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=35, azim=-35)      # vue initiale (modifiable a la souris)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cadre_3d)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- colonne 2 : X Y Z
        cadre_p = ttk.Labelframe(self, text=" Position P3 ", style="Bleu.TLabelframe")
        cadre_p.grid(row=0, column=2, sticky="ns", padx=6, pady=(12, 6))
        xmin, xmax, ymin, ymax, zmin, zmax = bornes_xyz()
        plages_p = ["min : %.3f\nmax : %.3f" % (xmin, xmax),
                    "min : %.3f\nmax : %.3f" % (ymin, ymax),
                    "min : %.3f\nmax : %.3f" % (zmin, zmax)]
        self.lim_p = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        self.ent_p = []
        for i, nom in enumerate(["X  (m)", "Y  (m)", "Z  (m)"]):
            ttk.Label(cadre_p, text=nom, foreground=BLEU,
                      font=("TkDefaultFont", 11, "bold")).grid(
                row=2 * i, column=0, sticky="w", padx=12, pady=(14, 0))
            e = ttk.Entry(cadre_p, textvariable=self.var_p[i], width=11,
                          justify="right", style="Champ.TEntry")
            e.grid(row=2 * i + 1, column=0, padx=12, pady=(2, 4))
            ttk.Label(cadre_p, text=plages_p[i], foreground="#777", justify="left",
                      font=("TkDefaultFont", 9)).grid(
                row=2 * i + 1, column=1, sticky="w", padx=(0, 10))
            e.bind("<KeyRelease>", lambda ev: self.calculer())
            self.ent_p.append(e)
        ttk.Label(cadre_p, text="bornes du volume atteignable\n(toutes les combinaisons\nne sont pas possibles)",
                  foreground="#777", justify="left", font=("TkDefaultFont", 8)).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 8))

        # --- colonne 3 : cases a cocher
        cadre_c = ttk.Frame(self)
        cadre_c.grid(row=0, column=3, sticky="n", padx=(6, 12), pady=(24, 6))
        ttk.Checkbutton(cadre_c, text="MGD", variable=self.var_mgd,
                        command=self.clic_mgd).pack(anchor="w", pady=8)
        ttk.Checkbutton(cadre_c, text="MGI", variable=self.var_mgi,
                        command=self.clic_mgi).pack(anchor="w", pady=8)
        ttk.Checkbutton(cadre_c, text="Validation MGI", variable=self.var_val,
                        command=self.clic_val).pack(anchor="w", pady=8)

        # --- trajectoire : points de position saisis en coordonnees X  Y  Z
        traj = ttk.Labelframe(self, text=" Trajectoire : points de position suivis par la MGI ",
                              style="Bleu.TLabelframe")
        traj.grid(row=1, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 6))
        ttk.Label(traj, text="Un point par ligne :   X    Y    Z    (en m)",
                  foreground=BLEU).grid(row=0, column=0, sticky="w", padx=12, pady=(6, 0))
        cadre_txt = ttk.Frame(traj)
        cadre_txt.grid(row=1, column=0, rowspan=2, sticky="w", padx=12, pady=(2, 8))
        self.txt_pts = tk.Text(cadre_txt, width=32, height=5, font="TkFixedFont",
                               relief="solid", bd=1, wrap="none")
        barre = ttk.Scrollbar(cadre_txt, orient="vertical", command=self.txt_pts.yview)
        self.txt_pts.configure(yscrollcommand=barre.set)
        self.txt_pts.pack(side="left")
        barre.pack(side="left", fill="y")
        self.txt_pts.insert("1.0", POINTS_DEFAUT)
        self.txt_pts.tag_configure("atteint", background="#c8e6c9")
        self.txt_pts.bind("<KeyRelease>", lambda ev: self.maj_apercu())
        ttk.Button(traj, text="Point suivant", command=self.point_suivant).grid(
            row=1, column=1, sticky="n", padx=(16, 6), pady=(2, 0))
        ttk.Button(traj, text="Repartir du 1er point", command=self.repartir).grid(
            row=1, column=2, sticky="n", padx=6, pady=(2, 0))
        ttk.Button(traj, text="Carre", command=lambda: self.charger_points(POINTS_DEFAUT)).grid(
            row=1, column=3, sticky="n", padx=6, pady=(2, 0))
        ttk.Button(traj, text="Cercle", command=lambda: self.charger_points(POINTS_CERCLE)).grid(
            row=1, column=4, sticky="n", padx=6, pady=(2, 0))
        ttk.Button(traj, text="Lancer la trajectoire", command=self.lancer_trajectoire).grid(
            row=1, column=5, sticky="n", padx=(16, 6), pady=(2, 0))
        ttk.Button(traj, text="Stop", command=self.stop_trajectoire).grid(
            row=1, column=6, sticky="n", padx=6, pady=(2, 0))
        ttk.Button(traj, text="Commande en vitesse",command=self.lancer_commande).grid(
            row=1, column=8, sticky="n", padx=6, pady=(2, 0))
        ttk.Button(traj, text="Effacer la trace", command=self.effacer_trace).grid(
            row=1, column=7, sticky="n", padx=(16, 6), pady=(2, 0))
        self.lbl_traj = ttk.Label(traj, text="", font=("TkDefaultFont", 10),
                                  wraplength=720, justify="left")
        self.lbl_traj.grid(row=2, column=1, columnspan=4, sticky="nw", padx=(16, 12), pady=(6, 6))

        # --- bas : messages
        bas = ttk.Labelframe(self, text=" Messages ")
        bas.grid(row=2, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 12))
        self.lbl_msg = ttk.Label(bas, text="", font=("TkDefaultFont", 10))
        self.lbl_msg.pack(anchor="w", padx=10, pady=(6, 0))
        self.lbl_val = ttk.Label(bas, text="", font=("TkDefaultFont", 10))
        self.lbl_val.pack(anchor="w", padx=10, pady=(2, 0))
        ttk.Label(bas, foreground="#777",
                  text=("Butees :  q1 [-180 ; 180] deg    q2 [0 ; 135] deg    "
                        "q3 libre    q4 [0 ; 0.350] m        a = %.3f   b = %.3f"
                        % (a, b))).pack(anchor="w", padx=10, pady=(2, 6))

    # ---------------- gestion des modes ----------------
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

    def maj_etats(self):
        """MGD coche : on saisit q.  MGI coche : on saisit X, Y, Z."""
        for e in self.ent_q:
            e.configure(state="normal" if self.var_mgd.get() else "readonly")
        for e in self.ent_p:
            e.configure(state="normal" if self.var_mgi.get() else "readonly")

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
        for i in range(3):
            self.var_q[i].set("%.2f" % degrees(self.q[i]))
        self.var_q[3].set("%.4f" % self.q[3])

    def remplir_p(self, p):
        for i in range(3):
            self.var_p[i].set("%.4f" % p[i])

    # ---------------- calcul ----------------
    def calculer(self):
        if self.en_cours:
            return
        self.colorer_champs()
        if self.var_mgd.get():
            try:
                q1 = radians(float(self.var_q[0].get()))
                q2 = radians(float(self.var_q[1].get()))
                q3 = radians(float(self.var_q[2].get()))
                q4 = float(self.var_q[3].get())
            except ValueError:
                self.message("MGD : saisie incomplete ou non numerique.", "#b26a00")
                return
            ok, msg = dans_butees(q1, q2, q4)
            if not ok:
                self.message("MGD : " + msg, ROUGE)
                return
            self.q = [q1, q2, q3, q4]
            p = MGD(q1, q2, q3, q4)
            self.remplir_p(p)
            self.message("MGD :  q  ->  P3 = (%.4f ; %.4f ; %.4f)"
                         % (p[0], p[1], p[2]), BLEU)

        elif self.var_mgi.get():
            try:
                x = float(self.var_p[0].get())
                y = float(self.var_p[1].get())
                z = float(self.var_p[2].get())
            except ValueError:
                self.message("MGI : saisie incomplete ou non numerique.", "#b26a00")
                return
            try:
                q = MGI(x, y, z)
            except ZeroDivisionError:
                self.message("MGI : division par zero.", ROUGE)
                return
            ok, msg = dans_butees(q[0], q[1], q[3])
            if not ok:
                self.message("MGI : point hors espace de travail  (%s)" % msg, ROUGE)
                return
            self.q = q
            self.remplir_q()
            self.message("MGI :  P3  ->  q1 = %.2f deg   q2 = %.2f deg   "
                         "q4 = %.4f   (q3 libre)"
                         % (degrees(q[0]), degrees(q[1]), q[3]), VERT)
        else:
            self.message("Cochez MGD ou MGI.", "#777")
            return

        self.colorer_champs()
        self.rafraichir()
        self.afficher_validation()

    def afficher_validation(self):
        """Validation du MGI : q -> MGD -> MGI -> comparaison avec q."""
        if not self.var_val.get():
            self.lbl_val.config(text="")
            return
        q1, q2, q3, q4 = self.q
        p = MGD(q1, q2, q3, q4)
        qn = MGI(p[0], p[1], p[2])
        e1 = ecart_angle(qn[0] - q1)
        e2 = ecart_angle(qn[1] - q2)
        e3 = ecart_angle(qn[2] - q3)
        e4 = qn[3] - q4
        ok = max(abs(e1), abs(e2), abs(e4)) < TOLERANCE
        self.lbl_val.config(
            text=("Validation MGI :  e1 = %.2e   e2 = %.2e   e3 = %.2e   e4 = %.2e"
                  "   ->   %s%s"
                  % (e1, e2, e3, e4, "OK" if ok else "ECART",
                     "" if abs(e3) < TOLERANCE else "   (q3 n'agit pas sur P3)")),
            foreground=VERT if ok else ROUGE)

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

    def verifier_points(self, pts, nums):
        """(True, "") si tous les points sont atteignables, sinon (False, message)."""
        for (x, y, z), n in zip(pts, nums):
            q = MGI(x, y, z)
            ok, msg = dans_butees(q[0], q[1], q[3])
            if not ok:
                return False, "Ligne %d (%.3f ; %.3f ; %.3f) hors espace de travail : %s" % (
                    n, x, y, z, msg)
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
                                     "Cliquez sur \"Point suivant\"." % len(pts), foreground=VERT)
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
        q_cible = MGI(x, y, z)
        ok, msg = dans_butees(q_cible[0], q_cible[1], q_cible[3])
        if not ok:
            self.message("Trajectoire : point %d hors espace de travail (%s)" % (i + 1, msg), ROUGE)
            return
        self.en_cours = True
        self.deplacer_pas_a_pas(list(self.q), q_cible, 1, 10, i, x, y, z)

    def deplacer_pas_a_pas(self, q_depart, q_arrivee, k, n, i, x, y, z):
        """Un pas du deplacement : q avance de 1/n vers q_arrivee, puis on redessine."""
        t = k / n
        self.q = [q_depart[j] + t * (q_arrivee[j] - q_depart[j]) for j in range(4)]
        self.remplir_q()
        p = MGD(self.q[0], self.q[1], self.q[2], self.q[3])
        self.remplir_p(p)
        self.colorer_champs()
        self.rafraichir()
        if k < n:
            self.after(30, lambda: self.deplacer_pas_a_pas(q_depart, q_arrivee, k + 1, n, i, x, y, z))
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
            fin = "   (fin du tour : le prochain clic repart au debut)"
        self.message("Trajectoire : point %d/%d   P3 = (%.3f ; %.3f ; %.3f)   "
                     "q1 = %.1f   q2 = %.1f   q4 = %.3f%s"
                     % (i + 1, len(pts), x, y, z,
                        degrees(self.q[0]), degrees(self.q[1]), self.q[3], fin), BLEU)
        self.afficher_validation()
        self.en_cours = False

        if self.auto:
            self.compte_auto += 1
            if self.compte_auto < self.total_auto:
                self.after_id_auto = self.after(150, self.point_suivant)
            else:
                self.auto = False
                self.message("Trajectoire complete : %d points parcourus." % self.total_auto, VERT)

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
        self.point_suivant()

    def stop_trajectoire(self):
        """Arrete l'enchainement automatique. Le bras finit son pas en cours puis s'arrete."""
        if self.after_id_auto is not None:
            self.after_cancel(self.after_id_auto)
            self.after_id_auto = None
        if self.auto:
            self.auto = False
            self.message("Trajectoire arretee. Le bras reste ou il est.", "#b26a00")

    def repartir(self):
        """Le prochain clic ira au 1er point. Le bras ne bouge pas."""
        self.i_traj = 0
        self.txt_pts.tag_remove("atteint", "1.0", "end")
        self.message("Le prochain clic ira au 1er point. Le bras reste ou il est.", "#777")

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

    def lancer_commande(self):
        if self.en_cours or self.auto:
            return
        if not self.apercu_ok or not self.apercu:
            self.message("Corrigez d'abord la liste de points.", ROUGE)
            return
        dt = 0.05
        self.traj_pos = self.apercu
        self.traj_vit = deriver_trajectoire(self.apercu, dt)
        self.k_cmd = 0
        self.trace = []
        self.q = MGI(*self.traj_pos[0])
        self.pas_commande(dt, 5.0)

    def pas_commande(self, dt, Kp):
        """Un pas : controleur -> qdot -> integration q = q + qdot*dt."""
        if self.k_cmd >= len(self.traj_pos):
            self.message("Commande terminee. Vitesses non nulles aux points "
                         "de passage : mouvement continu.", VERT)
            return

        Pd     = self.traj_pos[self.k_cmd]
        Pdot_d = self.traj_vit[self.k_cmd]

        qdot_c, err = controleur(self.q, Pd, Pdot_d, Kp)

        self.q = [self.q[j] + qdot_c[j]*dt for j in range(4)]
        self.qdot = qdot_c

        p = MGD(*self.q)
        self.trace.append(tuple(p))
        self.remplir_q()
        self.remplir_p(p)
        self.rafraichir()

        Pdot = MCD(self.q[0], self.q[1], self.q[2], self.q[3], qdot_c)
        self.message("Commande %d/%d   qdot = [%.3f  %.3f  %.3f  %.3f]   "
                     "|Pdot| = %.3f m/s   erreur = %.4f m"
                     % (self.k_cmd+1, len(self.traj_pos),
                        qdot_c[0], qdot_c[1], qdot_c[2], qdot_c[3],
                        np.linalg.norm(Pdot), np.linalg.norm(err)), BLEU)

        self.k_cmd += 1
        self.after_id_cmd = self.after(int(dt*1000),lambda: self.pas_commande(dt, Kp))

    # ---------------- dessin du robot ----------------
    def rafraichir(self):
        q1, q2, q3, q4 = self.q
        O0, O2, O3, P3 = [np.array(v) for v in positions_corps(q1, q2, q4)]
        p = MGD(q1, q2, q3, q4)

        u = np.array([sin(q1), -cos(q1), 0.0])    # direction du bras 1
        v = np.array([cos(q1), sin(q1), 0.0])     # axe de l'epaule (q2)
        z = np.array([0.0, 0.0, 1.0])
        d = (P3 - O3) / b                         # axe de l'outil

        # ---------- corps du robot ----------
        sc = ([], [])
        # socle fixe
        cylindre(sc, (0, 0, ZSOL), (0, 0, ZSOL + 0.05), 0.45, "#263238", n=32)
        cylindre(sc, (0, 0, ZSOL + 0.05), (0, 0, -0.10), 0.30, "#37474f", n=32)
        # tourelle (tourne avec q1)
        cylindre(sc, (0, 0, -0.10), (0, 0, 0.13), 0.22, "#78909c", n=32)
        # bras 1, longueur a
        boite(sc, O0 + u * a / 2, u, v, z, (a, 0.16, 0.20), "#c62828")
        # articulation d'epaule (q2)
        cylindre(sc, O2 - v * 0.13, O2 + v * 0.13, 0.13, "#263238")
        # manchon (tourne avec q2)
        cylindre(sc, O2 - d * 0.10, O2 + d * 0.15, 0.085, "#1f4fbf", nseg=2)
        # tige coulissante (q4)
        cylindre(sc, O2 - d * 0.05, O3, 0.055, "#cfd8dc", nseg=2)
        # outil, longueur b
        cylindre(sc, O3, O3 + d * 0.03, 0.075, "#455a64")                 # bride
        cylindre(sc, O3 + d * 0.03, P3 - d * 0.07, 0.05, "#2e7d32", nseg=2)
        cylindre(sc, P3 - d * 0.07, P3, 0.025, "#a5d6a7")                 # pointe
        # repere de rotation de l'outil (q3)
        e = cos(q3) * v + sin(q3) * np.cross(d, v)
        boite(sc, O3 + d * 0.03 + e * 0.10, e, d, np.cross(e, d),
              (0.07, 0.03, 0.03), "#fdd835")
        sphere(sc, P3, 0.04, "#ff9800")                                    # centre outil P3

        # ---------- scene ----------
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_axis_off()
        self.ax.computed_zorder = False           # ordre de dessin impose par zorder

        S = 1.9
        # sol (disque avec cercles et rayons)
        th = np.linspace(0, 2 * np.pi, 90)
        disque = [(S * np.cos(t), S * np.sin(t), ZSOL) for t in th]
        sol = Poly3DCollection([disque], facecolors="#eceff1",
                               edgecolors="#b0bec5", linewidths=1.0)
        sol.set_zorder(0)
        sol.set_clip_on(False)
        self.ax.add_collection3d(sol)
        traits = []
        for r in (0.5, 1.0, 1.5):
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
        R = a + Q4_MAX + b
        self.ax.plot(R * np.cos(th), R * np.sin(th), [ZSOL] * len(th),
                     "--", color="#90a4ae", lw=1.0, zorder=1, clip_on=False)

        # projection de P3 sur le sol
        self.ax.plot([p[0], p[0]], [p[1], p[1]], [ZSOL, p[2]], ":",
                     color="#f57c00", lw=1.3, zorder=1, clip_on=False)
        self.ax.plot([p[0]], [p[1]], [ZSOL], "o", color="#f57c00", ms=5, zorder=1, clip_on=False)

        # petit repere X Y Z dans un coin
        o = np.array([-1.35, -1.35, ZSOL])
        for vec, col, nom in ((np.array([1, 0, 0]), "#c62828", "X"),
                              (np.array([0, 1, 0]), "#2e7d32", "Y"),
                              (np.array([0, 0, 1]), "#1f4fbf", "Z")):
            fin_ = o + 0.4 * vec
            self.ax.plot([o[0], fin_[0]], [o[1], fin_[1]], [o[2], fin_[2]],
                         color=col, lw=2, zorder=1, clip_on=False)
            self.ax.text(fin_[0], fin_[1], fin_[2], nom, color=col,
                         fontsize=9, fontweight="bold", zorder=1)

        # robot
        robot = Poly3DCollection(sc[0], facecolors=sc[1], edgecolors=sc[1], linewidths=0.3)
        robot.set_zorder(5)
        robot.set_clip_on(False)
        self.ax.add_collection3d(robot)

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
        haut = 0.55
        self.ax.set_xlim(-S, S)
        self.ax.set_ylim(-S, S)
        self.ax.set_zlim(ZSOL, haut)
        self.ax.set_box_aspect((2 * S, 2 * S, haut - ZSOL), zoom=1.8)
        self.canvas.draw()


# =============================================================================
#  PROGRAMME PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    # --- tests en console (decommenter au besoin)
    # verification(0.5, 0.7, 0, 0.2)
    # balayage(0.3)
    q = [0.5, 0.7, 0, 0.2]
    print("Pdot =", MCD(*q, [0.1, 0.0, 0.0, 0.0]))
    print("qdot =", MCI(*q, [0.05, 0.0, 0.0]))
    # --- lancement de l'interface
    Application().mainloop()