import math
import numpy as np

# ----------------------------------------------------------------------------
# 1. PARAMETRES
# ----------------------------------------------------------------------------
D = np.array([0.352, 0.0, 0.0, 0.380, 0.0, 0.065])         # d_i  (m)
A = np.array([0.070, 0.360, 0.0, 0.0, 0.0, 0.0])           # a_i  (m)
ALPHA = np.radians([-90.0, 0.0, -90.0, 90.0, -90.0, 0.0])  # alpha_i
OFFSET = np.radians([0.0, -90.0, 0.0, 0.0, 0.0, 0.0])      # theta_i = q_i + offset_i
 
# Limites articulaires (a verifier avec la documentation ABB si disponible)
Q_MIN = np.radians([-180.0, -90.0, -230.0, -200.0, -115.0, -400.0])
Q_MAX = np.radians([180.0, 110.0, 50.0, 200.0, 115.0, 400.0])
 
D1, A1, A2, D4, D6 = 0.352, 0.070, 0.360, 0.380, 0.065   # m
TOLERANCE = 1e-9
L3 = D4 + D6      # 0.445 m : O6 au bout de l'avant-bras (modele du rapport, poignet bloque)
# vitesses articulaires maximales (tableau 4 du rapport), en rad/s
QDOT_MAX = np.radians([200.0, 200.0, 260.0, 360.0, 360.0, 450.0])

PORTEE = 0.810      # portee maximale constructeur (m)
EPS_SING = math.radians(3.0)  # seuil de detection des singularites
 
 
def wrap(a):
    """Ramene un angle dans ]-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
 
 
# ----------------------------------------------------------------------------
# 2. MGD
# ----------------------------------------------------------------------------
def dh(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])
 
 
def mgd_frames(q):
    """Retourne [T00, T01, ..., T06] (matrices 4x4 dans la base)."""
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        T = T @ dh(q[i] + OFFSET[i], D[i], A[i], ALPHA[i])
        frames.append(T.copy())
    return frames
 
 
def mgd(q):
    """MGD matriciel : T06."""
    return mgd_frames(q)[-1]
 
 
def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
 
 
def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
 
 
def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
 
 
def mgd_vectoriel(q):
    """MGD vectoriel : position = somme de vecteurs, orientation = rotations elementaires.
 
    R03 = Rz(q1) Ry(q2+q3-90deg) Rx(180deg)      R36 = Rz(q4) Ry(-q5) Rz(q6)
    P   = d1 z0 + Rz(q1)[a1 + a2 sin q2 + d4 cos q23 , 0 , a2 cos q2 - d4 sin q23] + d6 z6
    """
    q = np.asarray(q, float)
    q1, q2, q3, q4, q5, q6 = q
    q23 = q2 + q3
    r = A1 + A2 * math.sin(q2) + D4 * math.cos(q23)
    z = D1 + A2 * math.cos(q2) - D4 * math.sin(q23)
    wc = np.array([r * math.cos(q1), r * math.sin(q1), z])           # centre du poignet
    R03 = rotz(q1) @ roty(q23 - math.pi / 2) @ rotx(math.pi)
    R36 = rotz(q4) @ roty(-q5) @ rotz(q6)
    R = R03 @ R36
    p = wc + D6 * R[:, 2]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T
 
 
def pos_vectorielle(Q):
    """Version vectorisee (N x 6) -> (N x 3) de la position de l'effecteur."""
    Q = np.atleast_2d(Q)
    q1, q2, q3, q4, q5 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 4]
    q23 = q2 + q3
    r = A1 + A2 * np.sin(q2) + D4 * np.cos(q23)
    z = D1 + A2 * np.cos(q2) - D4 * np.sin(q23)
    # axe de l'outil dans le repere R03 : R36 * ez
    wx = -np.sin(q5) * np.cos(q4)
    wy = -np.sin(q5) * np.sin(q4)
    wz = np.cos(q5)
    # Rx(180) puis Ry(q23-90) puis Rz(q1)
    ux, uy, uz = wx, -wy, -wz
    psi = q23 - math.pi / 2
    vx = np.cos(psi) * ux + np.sin(psi) * uz
    vy = uy
    vz = -np.sin(psi) * ux + np.cos(psi) * uz
    ax = np.cos(q1) * vx - np.sin(q1) * vy
    ay = np.sin(q1) * vx + np.cos(q1) * vy
    az = vz
    x = r * np.cos(q1) + D6 * ax
    y = r * np.sin(q1) + D6 * ay
    zz = z + D6 * az
    return np.column_stack([x, y, zz])
 
 
def rpy_to_R(roll, pitch, yaw):
    return rotz(yaw) @ roty(pitch) @ rotx(roll)
 
 
def R_to_rpy(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw
 
 
# ----------------------------------------------------------------------------
# 3. MGI
# ----------------------------------------------------------------------------
def mgi_all(T):
    """MGI analytique. Retourne une liste de dictionnaires
    {'q': array(6), 'ok': bool, 'raison': str, 'branche': str} (8 solutions max).
    Decouplage : centre du poignet (axes 4,5,6 concourants) -> q1,q2,q3 ; puis q4,q5,q6.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    wc = p - D6 * R[:, 2]
    sols = []
    rho = math.hypot(wc[0], wc[1])
    zc = wc[2]
    v = zc - D1
 
    # deux branches pour q1 : epaule "devant" (r>0) ou "derriere" (r<0)
    branches_q1 = []
    if rho < 1e-6:
        branches_q1 = [(0.0, 1.0, "singularite epaule")]
    else:
        branches_q1 = [(math.atan2(wc[1], wc[0]), 1.0, "epaule devant"),
                       (math.atan2(-wc[1], -wc[0]), -1.0, "epaule derriere")]
 
    for q1, sgn, nom1 in branches_q1:
        u = sgn * rho - A1
        c = (u * u + v * v - A2 * A2 - D4 * D4) / (2 * A2 * D4)
        if abs(c) > 1.0 + 1e-9:
            sols.append({'q': None, 'ok': False, 'raison': 'hors de portee', 'branche': nom1})
            continue
        c = max(-1.0, min(1.0, c))
        for sd, nom2 in ((1.0, "coude 1"), (-1.0, "coude 2")):
            delta = sd * math.acos(c)
            beta = math.atan2(v, u) - math.atan2(D4 * math.sin(delta), A2 + D4 * math.cos(delta))
            q2 = wrap(math.pi / 2 - beta)
            phi = -(beta + delta)
            q3 = wrap(phi - q2)
            R03 = rotz(q1) @ roty(q2 + q3 - math.pi / 2) @ rotx(math.pi)
            R36 = R03.T @ R
            # R36 = Rz(q4) Ry(b) Rz(q6) avec b = -q5
            s_b = math.hypot(R36[0, 2], R36[1, 2])
            if s_b > 1e-7:
                b = math.atan2(s_b, R36[2, 2])
                a1_ = math.atan2(R36[1, 2], R36[0, 2])
                c1_ = math.atan2(R36[2, 1], -R36[2, 0])
                poignets = [(a1_, -b, c1_, "poignet 1"),
                            (wrap(a1_ + math.pi), b, wrap(c1_ + math.pi), "poignet 2")]
            else:  # singularite du poignet (q5 = 0 ou pi) : q4 choisi nul, q6 = somme
                b = 0.0 if R36[2, 2] > 0 else math.pi
                s = math.atan2(R36[1, 0], R36[0, 0]) if b == 0.0 else math.atan2(-R36[1, 0], -R36[0, 0])
                poignets = [(0.0, -b, s, "poignet singulier")]
            for q4, q5, q6, nom3 in poignets:
                q = np.array([q1, q2, q3, wrap(q4), wrap(q5), wrap(q6)])
                ok, raison = True, "valide"
                for i in range(6):
                    if q[i] < Q_MIN[i] - 1e-9 or q[i] > Q_MAX[i] + 1e-9:
                        # q4 et q6 peuvent aussi etre atteints a 2*pi pres
                        alt = q[i] + (2 * math.pi if q[i] < 0 else -2 * math.pi)
                        if Q_MIN[i] - 1e-9 <= alt <= Q_MAX[i] + 1e-9:
                            q[i] = alt
                        else:
                            ok, raison = False, "limite q%d" % (i + 1)
                            break
                sols.append({'q': q, 'ok': ok, 'raison': raison,
                             'branche': "%s / %s / %s" % (nom1, nom2, nom3)})
    return sols
 
 
def mgi_best(T, q_ref):
    """Meilleure solution valide (la plus proche de q_ref) ou None."""
    valides = [s for s in mgi_all(T) if s['ok']]
    if not valides:
        return None
    return min(valides, key=lambda s: np.linalg.norm(s['q'] - q_ref))
 
 
# ----------------------------------------------------------------------------
# 4. JACOBIENNE, MCD, MCI, SINGULARITES
# ----------------------------------------------------------------------------
def jacobienne(q, frames=None):
    """J (6x6) : [v ; w] = J qdot  (liaisons rotoides : Jv = z x (pe - o), Jw = z)."""
    if frames is None:
        frames = mgd_frames(q)
    pe = frames[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        J[:3, i] = np.cross(z, pe - o)
        J[3:, i] = z
    return J
 
 
def mcd(q, qdot):
    """Modele cinematique direct : xdot = J(q) qdot."""
    return jacobienne(q) @ np.asarray(qdot, float)
 
 
L_NORM = 0.4        # m : longueur caracteristique pour homogeneiser lignes lineaires/angulaires
SEUIL_SING = 0.05   # sigma_min (Jacobienne normalisee) sous lequel on amortit
LAMBDA_MAX = 0.05
 
 
def jacobienne_normalisee(J):
    """Lignes de vitesse lineaire divisees par L_NORM : toutes les lignes deviennent sans unite."""
    Js = np.array(J, float).copy()
    Js[:3] /= L_NORM
    return Js
 
 
def sigma_min(J):
    """Plus petite valeur singuliere de la Jacobienne normalisee (0 = singularite)."""
    return float(np.linalg.svd(jacobienne_normalisee(J), compute_uv=False)[-1])
 
 
def mci(J, xdot):
    """Modele cinematique inverse : qdot = J# xdot (pseudo-inverse amortie, DLS).
    L'amortissement n'est actif qu'au voisinage d'une singularite (sigma_min < SEUIL_SING)."""
    Js = jacobienne_normalisee(J)
    xs = np.array(xdot, float).copy()
    xs[:3] /= L_NORM
    smin = np.linalg.svd(Js, compute_uv=False)[-1]
    lam2 = 0.0 if smin >= SEUIL_SING else (1.0 - (smin / SEUIL_SING) ** 2) * LAMBDA_MAX ** 2
    return Js.T @ np.linalg.solve(Js @ Js.T + lam2 * np.eye(6), xs)
 
 
def singularites(q):
    """Liste des singularites proches de q + indicateurs numeriques."""
    q = np.asarray(q, float)
    res = []
    r = A1 + A2 * math.sin(q[1]) + D4 * math.cos(q[1] + q[2])
    if abs(r) < 0.025:
        res.append("epaule (centre du poignet sur l'axe 1, r=%.3f m)" % r)
    if abs(math.cos(q[2])) < math.sin(EPS_SING):
        res.append("coude (bras tendu/replie, q3=%.0f deg)" % math.degrees(q[2]))
    if abs(math.sin(q[4])) < math.sin(EPS_SING):
        res.append("poignet (axes 4 et 6 alignes, q5=%.1f deg)" % math.degrees(q[4]))
    J = jacobienne(q)
    return res, float(np.linalg.det(J)), sigma_min(J)
 
 # ---------------- modele du rapport (sections 7 et 8) : poignet bloque, 3 axes ----------------
def _abcd(q):
    """r, a, b, c, d du rapport (7.2) : rdot = a q2p + b q3p ,  hdot = c q2p + d q3p."""
    q2, q3 = q[1], q[2]
    s23, c23 = math.sin(q2 + q3), math.cos(q2 + q3)
    r = A1 + A2 * math.sin(q2) + L3 * c23
    a, b = A2 * math.cos(q2) - L3 * s23, -L3 * s23
    c, d = -A2 * math.sin(q2) - L3 * c23, -L3 * c23
    return r, a, b, c, d


def mcd_rapport(q, qdot):
    """MCD du rapport (7.2) : vitesse (xdot, ydot, zdot) de O6 dans R0 (q4 = q5 = q6 = 0)."""
    r, a, b, c, d = _abcd(q)
    rd = a * qdot[1] + b * qdot[2]
    hd = c * qdot[1] + d * qdot[2]
    return np.array([rd * math.cos(q[0]) - r * qdot[0] * math.sin(q[0]),
                     rd * math.sin(q[0]) + r * qdot[0] * math.cos(q[0]), hd])


def mci_rapport(q, V):
    """MCI du rapport (8.2 a 8.4). Retourne (qdot, k, msg).
    qdot = [q1p, q2p, q3p] en rad/s, deja multiplie par k (8.4) ; k = 1 si aucune limite depassee ;
    si la MCI n'est pas definie (8.3) : qdot = None et msg donne la raison."""
    r, a, b, c, d = _abcd(q)
    delta = a * d - b * c                          # = -A2 * L3 * cos q3
    if abs(r) < 1e-6:
        return None, 1.0, "r = 0 : outil sur l'axe 1"
    if abs(delta) < 1e-6:
        return None, 1.0, "cos q3 = 0 : bras tendu"
    rd = V[0] * math.cos(q[0]) + V[1] * math.sin(q[0])
    vt = -V[0] * math.sin(q[0]) + V[1] * math.cos(q[0])
    qdot = np.array([vt / r, (d * rd - b * V[2]) / delta, (a * V[2] - c * rd) / delta])
    k = min(1.0, float(np.min(QDOT_MAX[:3] / np.maximum(np.abs(qdot), 1e-12))))
    return qdot * k, k, ""
    
# ----------------------------------------------------------------------------
# 5. ESPACE DE TRAVAIL
# ----------------------------------------------------------------------------
def espace_de_travail(n=60000, seed=1, poignet_bloque=True):
    """Nuage de points de l'espace de travail (m) : tirage uniforme de q dans [Q_MIN, Q_MAX].
    poignet_bloque=True : q4 = q5 = q6 = 0, comme dans le rapport (3 axes : q1, q2, q3)."""
    rng = np.random.default_rng(seed)
    Q = rng.uniform(Q_MIN, Q_MAX, size=(n, 6))
    if poignet_bloque:
        Q[:, 3:] = 0.0
    return pos_vectorielle(Q)


def nuage_grille(pas_deg=10.0):
    """Nuage sur une grille reguliere en (q1, q2, q3), poignet bloque (3 boucles imbriquees).
    pas 10 deg -> 36 x 21 x 29 = 21 924 points."""
    q1 = np.radians(np.arange(-180.0, 180.0, pas_deg))                  # sans +180 (= -180)
    q2 = np.radians(np.arange(-90.0, 110.0 + 1e-9, pas_deg))
    q3 = np.radians(np.arange(-230.0, 50.0 + 1e-9, pas_deg))
    Q1, Q2, Q3 = np.meshgrid(q1, q2, q3, indexing="ij")
    Q = np.zeros((Q1.size, 6))
    Q[:, 0], Q[:, 1], Q[:, 2] = Q1.ravel(), Q2.ravel(), Q3.ravel()
    return pos_vectorielle(Q)


def coupe_meridienne(n_q2=1201, n_q3=2001):
    """Points (r, h) de la coupe (poignet bloque) : r = distance a l'axe 1, h = hauteur (m)."""
    Q2, Q3 = np.meshgrid(np.linspace(Q_MIN[1], Q_MAX[1], n_q2), np.linspace(Q_MIN[2], Q_MAX[2], n_q3))
    q23 = Q2 + Q3
    r = A1 + A2 * np.sin(Q2) + (D4 + D6) * np.cos(q23)
    h = D1 + A2 * np.cos(Q2) - (D4 + D6) * np.sin(q23)
    return r.ravel(), h.ravel()


def volume_revolution(cote=0.004):
    """Volume (m^3) du solide de revolution : somme de 2*pi*rho*dA sur les cases (rho, h) atteintes."""
    r, h = coupe_meridienne()
    cases = np.unique(np.column_stack([np.floor(np.abs(r) / cote), np.floor(h / cote)]).astype(int), axis=0)
    rho = (cases[:, 0] + 0.5) * cote
    return float(np.sum(2 * np.pi * rho) * cote ** 2)


def volume_estime(P, pas=0.025):
    """Volume ~ nombre de voxels occupes (depend fortement du nombre de points de P)."""
    vox = np.unique(np.floor(P / pas).astype(int), axis=0)
    return len(vox) * pas ** 3
 
 
# ----------------------------------------------------------------------------
# 6. TRAJECTOIRES + ASSERVISSEMENT
# ----------------------------------------------------------------------------
def loi_quintique(tau):
    """s(tau), ds/dtau : polynome de degre 5 (vitesse et acceleration nulles aux extremites)."""
    tau = min(max(tau, 0.0), 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4
    return s, ds
 
 
def loi_cubique(tau):
    tau = min(max(tau, 0.0), 1.0)
    return 3 * tau ** 2 - 2 * tau ** 3, 6 * tau - 6 * tau ** 2
 
 
def loi_lineaire(tau):
    tau = min(max(tau, 0.0), 1.0)
    return tau, 1.0
 
 
LOIS = {"quintique": loi_quintique, "cubique": loi_cubique, "lineaire": loi_lineaire}
 
 
def base_plan(plan):
    """Deux vecteurs unitaires (u,v) du plan de la trajectoire."""
    if plan == "XY (horizontal)":
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    if plan == "XZ (vertical)":
        return np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    return np.array([0, 1.0, 0]), np.array([0, 0, 1.0])  # YZ
 
 
class Trajectoire:
    def __init__(self, type_, centre, taille, duree, plan, loi="quintique"):
        self.type = type_
        self.c = np.asarray(centre, float)
        self.L = float(taille)
        self.T = float(duree)
        self.u, self.v = base_plan(plan)
        self.loi = LOIS[loi]
 
    def eval(self, t):
        """Retourne (pd, pd_dot) a l'instant t."""
        if self.type == "cercle":
            s, ds = self.loi(t / self.T)
            th, dth = 2 * math.pi * s, 2 * math.pi * ds / self.T
            # le cercle demarre au point (c + L u)
            p = self.c + self.L * (math.cos(th) * self.u + math.sin(th) * self.v)
            dp = self.L * dth * (-math.sin(th) * self.u + math.cos(th) * self.v)
            return p, dp
        # carre : 4 sommets, 4 segments parcourus en T/4 chacun, arret a chaque sommet
        h = self.L / 2.0
        S = [self.c + (-h) * self.u + (-h) * self.v, self.c + h * self.u + (-h) * self.v,
             self.c + h * self.u + h * self.v, self.c + (-h) * self.u + h * self.v]
        Ts = self.T / 4.0
        k = min(int(t // Ts), 3)
        s, ds = self.loi((t - k * Ts) / Ts)
        P0, P1 = S[k], S[(k + 1) % 4]
        return P0 + s * (P1 - P0), ds * (P1 - P0) / Ts
 
    def echantillon(self, n=200):
        ts = np.linspace(0, self.T, n)
        return np.array([self.eval(t)[0] for t in ts])
 
 
R_OUTIL_BAS = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # axe z de l'outil vers le bas
 
 
def erreur_orientation(R, Rd):
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))
 
 
def simuler(traj, q0, kp=8.0, ko=8.0, dt=0.01, Rd=R_OUTIL_BAS):
    """Boucle de commande : qdot_c = J# ( pd_dot + Kp (pd - p) ),  q <- q + qdot dt.
    L'orientation est maintenue constante (erreur d'orientation corrigee avec le gain ko)."""
    n = int(traj.T / dt)
    q = np.array(q0, float)
    log = {k: [] for k in ("t", "p", "pd", "e", "q", "smin", "qdot")}
    for k in range(n + 1):
        t = k * dt
        pd, vd = traj.eval(min(t, traj.T))
        fr = mgd_frames(q)
        p, R = fr[6][:3, 3], fr[6][:3, :3]
        J = jacobienne(q, fr)
        ep = pd - p
        eo = erreur_orientation(R, Rd)
        xdot = np.concatenate([vd + kp * ep, ko * eo])
        qdot = mci(J, xdot)
        smin = sigma_min(J)
        for key, val in (("t", t), ("p", p), ("pd", pd), ("e", ep), ("q", q.copy()),
                         ("smin", smin), ("qdot", qdot)):
            log[key].append(val)
        q = np.clip(q + qdot * dt, Q_MIN, Q_MAX)
    return {k: np.array(v) for k, v in log.items()}
 
 
# =============================================================================
#  ADAPTATION A InterfaceABB.py
#  (unites : metres ; noms et signatures attendus par l'interface)
# =============================================================================
def ecart_angle(e):
    """Ramene un ecart angulaire dans [-pi ; pi]."""
    return math.atan2(math.sin(e), math.cos(e))


def MGD_matriciel(q, retourner_T06=False):
    """MGD matriciel : renvoie [x, y, z] (ou T06 si retourner_T06=True)."""
    T = mgd(q)
    return T if retourner_T06 else [T[0, 3], T[1, 3], T[2, 3]]


def MGD_vectoriel(q, retourner_T06=False):
    """MGD vectoriel : renvoie [x, y, z] (ou T06 si retourner_T06=True)."""
    T = mgd_vectoriel(q)
    return T if retourner_T06 else [T[0, 3], T[1, 3], T[2, 3]]


def MGD(q):
    return MGD_vectoriel(q)


def pose_vers_T(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


MGI = mgi_all
MGI_meilleure = mgi_best
MCD = mcd


def MCI(q, xdot):
    """MCI(q, xdot) : qdot = J#(q) xdot  (appelle votre mci(J, xdot))."""
    return mci(jacobienne(q), xdot)


def dans_butees(q):
    for i in range(6):
        if not (Q_MIN[i] - 1e-6 <= q[i] <= Q_MAX[i] + 1e-6):
            return False, "q%d = %.1f deg hors de [%.0f ; %.0f] deg" % (
                i + 1, math.degrees(q[i]), math.degrees(Q_MIN[i]), math.degrees(Q_MAX[i]))
    return True, ""


def deriver_trajectoire(pts, dt):
    """Vitesse desiree Pdot le long de la liste de points (differences finies)."""
    P = np.array(pts, float)
    n = len(P)
    ferme = n > 2 and np.allclose(P[0], P[-1], atol=1e-9)
    V = np.zeros_like(P)
    V[1:-1] = (P[2:] - P[:-2]) / (2 * dt)
    if ferme:
        V[0] = (P[1] - P[-2]) / (2 * dt)
        V[-1] = V[0]
    else:
        V[0] = (P[1] - P[0]) / dt
        V[-1] = (P[-1] - P[-2]) / dt
    return V


def controleur(q, Pd, Pdot_d, Kp, Rd=None):
    """qdot_c = J# ( [Pdot_d + Kp (Pd - Pr)] ; Kp * erreur_orientation ) ; retourne (qdot_c, erreur)."""
    Rd = R_OUTIL_BAS if Rd is None else Rd
    fr = mgd_frames(q)
    Pr = fr[6][:3, 3]
    erreur = np.array(Pd) - Pr
    eo = erreur_orientation(fr[6][:3, :3], Rd)
    xdot_c = np.concatenate([np.array(Pdot_d) + Kp * erreur, Kp * eo])
    return MCI(q, xdot_c), erreur


def validation_mgi(q):
    """q -> MGD -> MGI -> comparaison : (ecarts q1..q6, nb de solutions valides)."""
    valides = [s for s in mgi_all(mgd(q)) if s['ok']]
    if not valides:
        return None, 0
    s = min(valides, key=lambda s: sum(abs(ecart_angle(s['q'][i] - q[i])) for i in range(6)))
    return [ecart_angle(s['q'][i] - q[i]) for i in range(6)], len(valides)


def balayage(n=40000, seed=1):
    """Nuage de points de l'espace de travail (m)."""
    return espace_de_travail(n, seed)


_CACHE_BORNES = {}


def bornes_xyz():
    if 'b' not in _CACHE_BORNES:
        P = espace_de_travail(60000, 7)
        _CACHE_BORNES['b'] = (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(),
                              P[:, 2].min(), P[:, 2].max())
    return _CACHE_BORNES['b']



# =============================================================================
#  TESTS EN CONSOLE
# =============================================================================
if __name__ == "__main__":
    q = np.radians([20, 30, -40, 25, 50, 10])
    T = mgd(q)
    print("MGD matriciel :\n", np.round(T, 3))
    print("ecart MGD vectoriel :", np.max(np.abs(T - mgd_vectoriel(q))))
    sols = [s for s in mgi_all(T) if s['ok']]
    print("MGI : %d solution(s) valide(s)" % len(sols))
    print("singularites :", singularites(q)[0])