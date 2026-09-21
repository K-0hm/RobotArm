
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
BG = "#f4f4f4"
ROUGE = "#c62828"
BLEU = "#1f4fbf"
VERT = "#2e7d32"


class Application(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("IHM  -  Robot porte-outil 4 ddl")
        self.geometry("1200x660")
        self.minsize(1000, 560)
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

        self._styles()
        self._construire()
        self.maj_etats()
        self.remplir_q()
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

    # ---------------- construction ----------------
    def _construire(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # --- colonne 0 : q1 q2 q3 q4
        cadre_q = ttk.Labelframe(self, text=" Articulations ", style="Rouge.TLabelframe")
        cadre_q.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=(12, 6))
        self.ent_q = []
        for i, nom in enumerate(["q1  (deg)", "q2  (deg)", "q3  (deg)", "q4  (m)"]):
            ttk.Label(cadre_q, text=nom, foreground=ROUGE,
                      font=("TkDefaultFont", 11, "bold")).grid(
                row=2 * i, column=0, sticky="w", padx=12, pady=(14, 0))
            e = ttk.Entry(cadre_q, textvariable=self.var_q[i], width=11,
                          justify="right", style="Champ.TEntry")
            e.grid(row=2 * i + 1, column=0, padx=12, pady=(2, 4))
            e.bind("<KeyRelease>", lambda ev: self.calculer())
            self.ent_q.append(e)

        # --- colonne 1 : vue 3D du robot
        cadre_3d = ttk.Frame(self)
        cadre_3d.grid(row=0, column=1, sticky="nsew", padx=6, pady=(12, 6))
        self.fig = Figure(figsize=(6, 5), facecolor="white")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=22, azim=30)      # vue initiale (modifiable a la souris)
        self.canvas = FigureCanvasTkAgg(self.fig, master=cadre_3d)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- colonne 2 : X Y Z
        cadre_p = ttk.Labelframe(self, text=" Position P3 ", style="Bleu.TLabelframe")
        cadre_p.grid(row=0, column=2, sticky="ns", padx=6, pady=(12, 6))
        self.ent_p = []
        for i, nom in enumerate(["X  (m)", "Y  (m)", "Z  (m)"]):
            ttk.Label(cadre_p, text=nom, foreground=BLEU,
                      font=("TkDefaultFont", 11, "bold")).grid(
                row=2 * i, column=0, sticky="w", padx=12, pady=(14, 0))
            e = ttk.Entry(cadre_p, textvariable=self.var_p[i], width=11,
                          justify="right", style="Champ.TEntry")
            e.grid(row=2 * i + 1, column=0, padx=12, pady=(2, 4))
            e.bind("<KeyRelease>", lambda ev: self.calculer())
            self.ent_p.append(e)

        # --- colonne 3 : cases a cocher
        cadre_c = ttk.Frame(self)
        cadre_c.grid(row=0, column=3, sticky="n", padx=(6, 12), pady=(24, 6))
        ttk.Checkbutton(cadre_c, text="MGD", variable=self.var_mgd,
                        command=self.clic_mgd).pack(anchor="w", pady=8)
        ttk.Checkbutton(cadre_c, text="MGI", variable=self.var_mgi,
                        command=self.clic_mgi).pack(anchor="w", pady=8)
        ttk.Checkbutton(cadre_c, text="Validation MGI", variable=self.var_val,
                        command=self.clic_val).pack(anchor="w", pady=8)

        # --- bas : messages
        bas = ttk.Labelframe(self, text=" Messages ")
        bas.grid(row=1, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 12))
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

        haut = 0.55
        self.ax.set_xlim(-S, S)
        self.ax.set_ylim(-S, S)
        self.ax.set_zlim(ZSOL, haut)
        self.ax.set_box_aspect((2 * S, 2 * S, haut - ZSOL), zoom=2.0)
        self.canvas.draw()


# =============================================================================
#  PROGRAMME PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    # --- tests en console (decommenter au besoin)
    # verification(0.5, 0.7, 0, 0.2)
    # balayage(0.3)

    # --- lancement de l'interface
    Application().mainloop()