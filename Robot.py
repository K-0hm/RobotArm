from math import sin, cos, atan, atan2, sqrt, pi, degrees, radians

import numpy as np

import tkinter as tk





# setup
PI = 3.14159265358979323846
b = 0.250  # m
a = 1.150  # m


def MGD(q1, q2, q3, q4):
    x =  (a + sin(q2)*(q4+b))*sin(q1)
    y = -(a + sin(q2)*(q4+b))*cos(q1)
    z =  (q4+b)*cos(q2)
    return [x, y, z]

def matrice_dh(theta, d, a_dh, alpha):
    """Matrice de transformation homogene 4x4, convention DH standard."""
    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a_dh*ct],
        [st,  ct*ca, -ct*sa, a_dh*st],
        [0,      sa,     ca,       d],
        [0,       0,      0,       1]
    ])


def MGD_matriciel(q1, q2, q3, q4, retourner_T04=False):
    """Modele geometrique direct par la methode matricielle (Denavit-Hartenberg)"""
    T01 = matrice_dh(q1 - pi/2, 0, a, pi/2)
    T12 = matrice_dh(-q2,       0, 0, -pi/2)
    T23 = matrice_dh(0,        q4, 0, 0)
    T34 = matrice_dh(q3,        b, 0, 0)

    T04 = T01 @ T12 @ T23 @ T34
    if retourner_T04:
        return T04
    x, y, z = T04[0, 3], T04[1, 3], T04[2, 3]
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
    n = len(P)
    ferme = n > 2 and np.allclose(P[0], P[-1], atol=1e-9)
    V = np.zeros_like(P)
    V[1:-1] = (P[2:] - P[:-2]) / (2*dt)      # differences centrees
    if ferme:
        V[0]  = (P[1] - P[-2]) / (2*dt)
        V[-1] = V[0]
    else:
        V[0]  = (P[1] - P[0]) / dt
        V[-1] = (P[-1] - P[-2]) / dt
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


def balayage(inc, afficher=True):
    # determination de l'espace de travail du robot
    points=[]
    q1 = PI
    while q1 > -PI:
        q2 = 0
        while q2 <= PI*0.75:      # 0.75*PI = 3PI/4
            q4 = 0
            while q4 < 0.350:
                #pointy = MGD(q1, q2, 0, q4) #old relics?
                points.append(MGD(q1, q2, 0, q4))
                q4 += 0.005
            q2 += inc
        q1 -= inc
    points = np.array(points)

    return points


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
#  PROGRAMME PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    # --- tests en console (decommenter au besoin)
    # verification(0.5, 0.7, 0, 0.2)
    balayage(0.3, True)
    #q = [0.5, 0.7, 0, 0.2]
    #print("Pdot =", MCD(*q, [0.1, 0.0, 0.0, 0.0]))
    #print("qdot =", MCI(*q, [0.05, 0.0, 0.0]))
    # --- lancement de l'interface
    # Application().mainloop() #mmoved to gui.py