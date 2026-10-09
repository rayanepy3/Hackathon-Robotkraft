import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "COMMUN"))
import commun  # noqa: E402

CALIB = commun.ROOT / "TRI" / "tri_calib.json"
COULEURS = "BJR"
NOMS = {"B": "bleu", "J": "jaune", "R": "rouge"}
DESSIN = {"B": (255, 140, 0), "J": (0, 215, 255), "R": (200, 0, 255), None: (170, 170, 170)}
NB_PLACES = 6
STABILISATION_S = 1.5  # attente après un essai, le temps que le bras s'immobilise
ECHELLE = 1.5          # agrandissement des fenêtres (clics plus précis)

DEFAUTS = {
    "places": [],
    "table": [],
    "fenetre_px": 15,
    "part_min": 0.25,
    "surface_min_px": 150,
    "fusion_px": 35,
    "tube_px": [12, 45, 15],
    # Mesuré sur l'OAK : liquide bleu H 95-107 (bouchons bleus 108-118, exclus), jaune 19-23 sur la table
    # et 25-28 dans le rack (rack vert H 25-35 mais V < 110), magenta 175-3, bouchons orange 10-14.
    "couleurs": {
        "B": {"teinte": [[88, 107]], "s_min": 120, "v_min": 45},
        "J": {"teinte": [[17, 29]], "s_min": 160, "v_min": 120},
        "R": {"teinte": [[0, 7], [160, 179]], "s_min": 120, "v_min": 70},
    },
}
AIDE = ("places : 6 points [x, y] (image 640x480), place 1 = la plus à gauche vue du robot, sur le liquide "
        "d'un tube rangé. table : coins de la zone où sont posés les tubes. fenetre_px : côté du carré mesuré "
        "sur chaque place. part_min : part du carré qui doit être de la couleur pour dire la place occupée. "
        "surface_min_px : surface minimale d'une tache de couleur sur la table. fusion_px : deux taches plus "
        "proches comptent pour un seul tube (étiquette). tube_px : [demi-largeur, hauteur au-dessus, hauteur "
        "au-dessous] du tube debout dans une place occupée, retiré de la zone de la table pour ne pas le "
        "compter deux fois. couleurs : teinte = plages de H OpenCV (0-179), "
        "s_min et v_min = saturation et luminosité minimales (0-255). Réglage : make tri-voir.")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def charger_calib(path=CALIB):
    """Calibration du fichier, complétée par les valeurs par défaut."""
    calib = json.loads(json.dumps(DEFAUTS))  # copie profonde
    if path.exists():
        lue = json.loads(path.read_text())
        lue.pop("_aide", None)
        couleurs = lue.pop("couleurs", {})
        calib.update(lue)
        for c in COULEURS:
            calib["couleurs"][c].update(couleurs.get(c, {}))
    return calib


def calib_ok(calib):
    return len(calib["places"]) == NB_PLACES and len(calib["table"]) >= 3


def sauver_calib(calib, path=CALIB):
    texte = json.dumps({"_aide": AIDE, **calib}, indent=2, ensure_ascii=False)
    # listes de nombres sur une ligne ([x, y]), listes d'une ou deux plages aussi ([[0, 7], [160, 179]])
    texte = re.sub(r"\[\s*([-\d.]+(?:,\s*[-\d.]+)*)\s*\]",
                   lambda m: "[" + ", ".join(m[1].replace(",", " ").split()) + "]", texte)
    texte = re.sub(r"\[\s*(\[[^\[\]]*\](?:,\s*\[[^\[\]]*\])?)\s*\]",
                   lambda m: "[" + re.sub(r",\s+", ", ", m[1]) + "]", texte)
    path.write_text(texte + "\n")


def lire_points(texte, option):
    """ "x,y;x,y;..." -> [[x, y], ...]"""
    try:
        pts = [[int(round(float(v))) for v in p.split(",")] for p in texte.replace(" ", "").strip(";").split(";")]
    except ValueError:
        pts = None
    if not pts or any(len(p) != 2 for p in pts):
        sys.exit(f'ERREUR : {option} "{texte}" : attendu "x,y;x,y;..."')
    return pts


# ---------------------------------------------------------------------------
# Vision : fonctions pures sur une image BGR et une calibration
# ---------------------------------------------------------------------------
class Vue:
    """Ce que voit la caméra : couleur de chaque place (None = vide) et taches de couleur sur la table."""

    def __init__(self, img, places, parts, taches):
        self.img = img
        self.places = places  # B, J, R ou None, place 1 en premier
        self.parts = parts    # {couleur: part du carré} pour chaque place
        self.taches = taches  # {couleur: [(x, y, w, h, surface), ...]} dans la zone de la table
        self.compte = {c: len(taches[c]) for c in COULEURS}

    def libre(self):
        """Index (0-5) de la place libre la plus à gauche, None si le porte-tubes est plein."""
        return next((i for i, p in enumerate(self.places) if p is None), None)

    def resume(self):
        return f"porte-tubes {texte_places(self.places)}, table {texte_compte(self.compte)}"


def texte_places(places):
    return " ".join(p or "-" for p in places)


def texte_compte(compte):
    return " ".join(f"{c}{compte[c]}" for c in COULEURS)


def masques(img, calib):
    """Masque (0/255) de chaque couleur sur toute l'image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    res = {}
    for c in COULEURS:
        p = calib["couleurs"][c]
        m = np.zeros(img.shape[:2], np.uint8)
        for lo, hi in p["teinte"]:
            m |= cv2.inRange(hsv, (int(lo), int(p["s_min"]), int(p["v_min"])), (int(hi), 255, 255))
        res[c] = m
    return res


def etat_places(img, calib, m=None):
    """Couleur vue sur chaque place (None = vide) et parts de chaque couleur dans son carré.

    On ne mesure que dans le carré de chaque place : les flancs et le ruban rouges du porte-tubes
    seraient pris pour des tubes rouges.
    """
    m = masques(img, calib) if m is None else m
    hauteur, largeur = img.shape[:2]
    r = int(calib["fenetre_px"]) // 2
    places, parts = [], []
    for x, y in calib["places"]:
        x0, x1, y0, y1 = max(0, x - r), min(largeur, x + r + 1), max(0, y - r), min(hauteur, y + r + 1)
        n = max(1, (x1 - x0) * (y1 - y0))
        p = {c: np.count_nonzero(m[c][y0:y1, x0:x1]) / n for c in COULEURS}
        meilleure = max(p, key=p.get)
        places.append(meilleure if p[meilleure] >= calib["part_min"] else None)
        parts.append(p)
    return places, parts


def rect_tube(calib, x, y):
    """Rectangle (coins) occupé dans l'image par un tube debout dans la place (x, y)."""
    demi, haut, bas = calib["tube_px"]
    return (x - demi, y - haut), (x + demi, y + bas)


def zone_table(shape, calib, places=None):
    """Masque du polygone de la table (toute l'image si la zone n'est pas calibrée),
    sans les tubes debout dans les places occupées (places = résultat de etat_places)."""
    if len(calib["table"]) < 3:
        zone = np.full(shape[:2], 255, np.uint8)
    else:
        zone = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(zone, [np.array(calib["table"], np.int32)], 255)
    for (x, y), p in zip(calib["places"], places or []):
        if p is not None:
            cv2.rectangle(zone, *rect_tube(calib, x, y), 0, -1)
    return zone


def taches_table(img, calib, m=None, places=None):
    """Taches de chaque couleur dans la zone de la table : {couleur: [(x, y, w, h, surface), ...]}.

    Deux tubes collés font une seule tache : on s'en sert surtout pour savoir s'il en reste.
    places (couleurs des places) : retire de la zone les tubes debout dans le porte-tubes.
    """
    m = masques(img, calib) if m is None else m
    zone = zone_table(img.shape, calib, places)
    ouverture = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k = max(1, int(calib["fusion_px"]))
    fusion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    res = {}
    for c in COULEURS:
        mc = cv2.morphologyEx(cv2.bitwise_and(m[c], zone), cv2.MORPH_OPEN, ouverture)
        groupes = cv2.dilate(mc, fusion) if k > 1 else mc  # regroupe les morceaux d'un tube coupé par l'étiquette
        n, lab, stats, _ = cv2.connectedComponentsWithStats(groupes)
        surface = np.bincount(lab[mc > 0], minlength=n)  # pixels de couleur, sans la dilatation
        res[c] = [(*map(int, stats[i, :4]), int(surface[i])) for i in range(1, n)
                  if surface[i] >= calib["surface_min_px"]]
    return res


def analyser(img, calib):
    m = masques(img, calib)
    places, parts = etat_places(img, calib, m)
    return Vue(img, places, parts, taches_table(img, calib, m, places))


def observer(source, calib, n=3):
    """Vision sur n images successives : vote par place et médiane des comptes (bruit de la caméra)."""
    vues = [analyser(source.frame(), calib) for _ in range(n)]
    vue = vues[-1]
    if n > 1:
        vue.places = [Counter(p).most_common(1)[0][0] for p in zip(*(v.places for v in vues))]
        vue.compte = {c: int(np.median([v.compte[c] for v in vues])) for c in COULEURS}
    return vue


# ---------------------------------------------------------------------------
# Dessin (texte ASCII : les polices d'OpenCV n'ont pas d'accents)
# ---------------------------------------------------------------------------
def ecrire(img, texte, org, coul=(255, 255, 255), taille=0.45, fond=True):
    """Texte sur fond noir (pas de contour : avec OpenCV 5, la largeur du texte change avec l'épaisseur)."""
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    if fond:
        (w, h), b = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, taille, 1)
        cv2.rectangle(img, (org[0] - 1, org[1] - h - 2), (org[0] + w + 1, org[1] + b), (0, 0, 0), -1)
    cv2.putText(img, texte, org, cv2.FONT_HERSHEY_SIMPLEX, taille, coul, 1, cv2.LINE_AA)


def annoter(vue, calib, titre=(), cible=None):
    """Image annotée : zone de la table, taches, places (cible encadrée en blanc), bandeau de texte."""
    out = vue.img.copy()
    if len(calib["table"]) >= 3:
        cv2.polylines(out, [np.array(calib["table"], np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    for c in COULEURS:
        for x, y, w, h, _ in vue.taches[c]:
            cv2.rectangle(out, (x, y), (x + w, y + h), DESSIN[c], 2)
            ecrire(out, c, (x, max(12, y - 4)), DESSIN[c])
    r = int(calib["fenetre_px"]) // 2
    for i, ((x, y), c) in enumerate(zip(calib["places"], vue.places)):
        if c is not None:  # tube debout, hors zone de la table
            cv2.rectangle(out, *rect_tube(calib, x, y), DESSIN[c], 1)
        if i == cible:
            cv2.rectangle(out, (x - r - 4, y - r - 4), (x + r + 4, y + r + 4), (255, 255, 255), 2)
        cv2.rectangle(out, (x - r, y - r), (x + r, y + r), DESSIN[c], 2)
        ecrire(out, str(i + 1), (x + r + 3, y + 5), DESSIN[c])
    lignes = [titre] if isinstance(titre, str) else list(titre)
    if calib["places"]:
        lignes.append("Places  " + "  ".join(f"{i + 1}:{c or '-'} {100 * max(p.values()):.0f}%"
                                             for i, (c, p) in enumerate(zip(vue.places, vue.parts))))
    lignes.append("Table   " + "  ".join(f"{NOMS[c]} {vue.compte[c]}" for c in COULEURS))
    h = 8 + 18 * len(lignes)
    out[:h] = (out[:h] * 0.35).astype(np.uint8)
    for i, ligne in enumerate(lignes):
        ecrire(out, ligne, (8, 20 + 18 * i), fond=False)
    return out


def fenetre(nom, souris=None):
    cv2.namedWindow(nom, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    if souris:
        cv2.setMouseCallback(nom, souris)


def afficher(nom, img):
    cv2.imshow(nom, cv2.resize(img, None, fx=ECHELLE, fy=ECHELLE, interpolation=cv2.INTER_LINEAR))


def fenetre_fermee(nom):
    try:
        return cv2.getWindowProperty(nom, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


class ImageFixe:
    """Remplace l'OAK par une image enregistrée (tests sans caméra)."""

    def __init__(self, path):
        self.img = cv2.imread(path)
        if self.img is None:
            sys.exit(f"ERREUR : image illisible : {path}")

    def frame(self, *_, **__):
        return self.img.copy()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# calib : 6 places puis la zone de la table, à la souris (ou --slots/--table)
# ---------------------------------------------------------------------------
class Saisie:
    """Points cliqués : d'abord les 6 places, puis les coins de la table (Entrée ferme le polygone)."""

    def __init__(self):
        self.places, self.table, self.ferme = [], [], False

    def clic(self, x, y):
        if len(self.places) < NB_PLACES:
            self.places.append([x, y])
        elif not self.ferme:
            self.table.append([x, y])

    def annuler(self):
        if self.ferme:
            self.ferme = False
        elif self.table:
            self.table.pop()
        elif self.places:
            self.places.pop()

    def fermer(self):
        self.ferme = len(self.places) == NB_PLACES and len(self.table) >= 3

    def consigne(self):
        if len(self.places) < NB_PLACES:
            return [f"Clic gauche : place {len(self.places) + 1}/6 (1 = la plus a gauche vue du robot),",
                    "sur l'endroit ou le liquide d'un tube range est visible"]
        if not self.ferme:
            return [f"Clic gauche : coins de la zone table, sans le porte-tubes ({len(self.table)} points)",
                    "Entree : fermer le polygone (3 points minimum)"]
        return ["Calibration complete : s pour sauver", ""]


def calib(args):
    calib_ = charger_calib()
    if args.slots or args.table:
        if args.slots:
            calib_["places"] = lire_points(args.slots, "--slots")
        if args.table:
            calib_["table"] = lire_points(args.table, "--table")
        if len(calib_["places"]) != NB_PLACES:
            sys.exit(f"ERREUR : {len(calib_['places'])} places, il en faut {NB_PLACES} (--slots)")
        if len(calib_["table"]) < 3:
            sys.exit("ERREUR : la zone de la table demande au moins 3 points (--table)")
        sauver_calib(calib_)
        print(f"Calibration enregistrée : {CALIB}")
        return
    calib_souris(calib_)


def calib_souris(calib_):
    ancienne = {k: calib_[k] for k in ("places", "table")}
    oak = commun.Oak()
    nom = "TRI - calibration"
    s = Saisie()

    def souris(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            s.clic(int(round(x / ECHELLE)), int(round(y / ECHELLE)))
        elif ev == cv2.EVENT_RBUTTONDOWN:
            s.annuler()

    print("Clic gauche : ajouter un point. Clic droit ou Retour arrière : annuler le dernier.\n"
          "Entrée : fermer la zone de la table. s : sauver. r : tout recommencer. q ou Échap : quitter sans sauver.")
    fenetre(nom, souris)
    try:
        while True:
            img = oak.frame()
            essai = {**calib_, "places": s.places, "table": s.table if s.ferme else []}
            vue = analyser(img, essai)
            if not s.ferme:
                vue.taches = {c: [] for c in COULEURS}
                vue.compte = {c: 0 for c in COULEURS}
            out = annoter(vue, essai, s.consigne() + ["Clic droit/Retour : annuler  r : recommencer  q : quitter"])
            for x, y in ancienne["places"]:  # ancienne calibration en gris, pour repère
                cv2.circle(out, (x, y), 3, (130, 130, 130), -1)
            if not s.ferme and s.table:
                cv2.polylines(out, [np.array(s.table, np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
                for x, y in s.table:
                    cv2.circle(out, (x, y), 3, (255, 255, 255), -1)
            afficher(nom, out)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27) or fenetre_fermee(nom):
                print("Calibration abandonnée, rien n'a été enregistré.")
                return
            if k in (8, ord("u")):
                s.annuler()
            elif k in (10, 13):
                s.fermer()
            elif k == ord("r"):
                s.__init__()
            elif k == ord("s") and s.ferme:
                calib_["places"], calib_["table"] = s.places, s.table
                sauver_calib(calib_)
                print(f"Calibration enregistrée : {CALIB}\nVue actuelle : {analyser(oak.frame(), calib_).resume()}")
                return
    finally:
        cv2.destroyAllWindows()
        oak.close()


# ---------------------------------------------------------------------------
# voir : vision en direct
# ---------------------------------------------------------------------------
def voir(args):
    calib_ = charger_calib()
    if not calib_ok(calib_):
        print("Calibration incomplète (make tri-calib) : taches cherchées sur toute l'image.")
    source = ImageFixe(args.image) if args.image else commun.Oak()
    try:
        if args.once:
            vue = observer(source, calib_, n=1 if args.image else 3)
            cv2.imwrite(args.once, annoter(vue, calib_, "TRI - vision"))
            print(f"{vue.resume()}\nImage annotée : {args.once}")
            return
        nom = "TRI - vision (q pour quitter)"
        fenetre(nom)
        dernier = None
        while True:
            vue = analyser(source.frame(), calib_)
            if vue.resume() != dernier:  # n'affiche que les changements
                dernier = vue.resume()
                print(dernier, flush=True)
            afficher(nom, annoter(vue, calib_, "q : quitter"))
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27) or fenetre_fermee(nom):
                break
    finally:
        cv2.destroyAllWindows()
        source.close()


# ---------------------------------------------------------------------------
# run : l'épreuve
# ---------------------------------------------------------------------------
def cible(places, compte, ordre):
    """Configuration attendue : les tubes vus (porte-tubes + table) groupés dans l'ordre, de gauche à droite."""
    seq = [c for c in ordre for _ in range(places.count(c) + compte[c])][:NB_PLACES]
    return seq + [None] * (NB_PLACES - len(seq))


def score(places, attendu):
    return sum(1 for p, a in zip(places, attendu) if p is not None and p == a)


def pluriel(n, mot):
    return f"{n} {mot}{'s' if n > 1 else ''}"


class Tri:
    def __init__(self, args, calib_, source, journal):
        self.args, self.calib, self.source, self.journal = args, calib_, source, journal
        self.textes = {"B": args.text_b, "J": args.text_j, "R": args.text_r}
        self.limite = NB_PLACES * args.max_tries  # essais pour toute l'épreuve
        self.essais = 0
        self.ranges = []     # (couleur, place 1-6)
        self.incidents = []
        self.vue = None

    def noter(self, msg):
        self.incidents.append(msg)
        self.journal.log(msg)

    def regarder(self):
        precedente, self.vue = self.vue, observer(self.source, self.calib)
        if precedente is None or self.vue.resume() != precedente.resume():
            self.journal.log(f"vu : {self.vue.resume()}")
        return self.vue

    def image(self, vue, label, titre, cible_=None):
        self.journal.image(annoter(vue, self.calib, titre, cible_), label)

    def executer(self, ordre):
        debut, statut = time.time(), "interrompu"
        try:
            vue = self.regarder()
            self.image(vue, "depart", f"Depart, ordre {ordre}")
            commun.say("Tri, ordre " + " ".join(NOMS[c] for c in ordre))
            for c in ordre:
                if not self.couleur(c):
                    break
            statut = "terminé"
        except KeyboardInterrupt:
            self.journal.log("Ctrl+C : arrêt demandé")
        finally:
            self.bilan(ordre, statut, time.time() - debut)
        return statut == "terminé"

    def couleur(self, c):
        """Range les tubes de couleur c tant qu'il en reste sur la table. False : inutile de continuer le tri."""
        nom, essai = NOMS[c], 0  # essai : numéro de l'essai pour le tube en cours
        while True:
            avant = self.regarder()
            if avant.compte[c] == 0:
                self.journal.log(f"{nom} : plus de tube sur la table")
                return True
            place = avant.libre()
            if place is None:
                self.journal.log("porte-tubes plein")
                return False
            if self.essais >= self.limite:
                self.noter(f"limite de {self.limite} essais atteinte, arrêt du tri")
                commun.say("Limite d'essais atteinte")
                return False
            essai += 1
            self.essais += 1
            n = place + 1
            self.journal.log(f"essai {self.essais} : tube {nom} vers la place {n} "
                             f"(essai {essai}/{self.args.max_tries} pour ce tube)")
            self.image(avant, f"avant_essai{self.essais}", f"Essai {self.essais} : {nom} -> place {n} (avant)", place)
            commun.say(f"Tube {nom}, place {n}, essai {essai}")
            a = self.args
            if not commun.run_policy(a.policy, self.textes[c], a.seconds, a.fps, a.async_server, a.dry_run,
                                     self.journal):
                self.noter(f"essai {self.essais} : le modèle s'est arrêté en erreur (voir modele.log)")
            time.sleep(STABILISATION_S)
            apres = self.regarder()
            vu = apres.places[place]
            self.image(apres, f"apres_essai{self.essais}",
                       f"Essai {self.essais} : {nom} -> place {n} : {NOMS.get(vu, 'vide')}", place)
            if vu == c:
                self.ranges.append((c, n))
                self.journal.log(f"rangé : tube {nom} en place {n}")
                commun.say("Rangé")
                essai = 0
                continue
            if vu is not None:  # on ne revient pas dessus : la place suivante devient la cible
                self.noter(f"mauvaise couleur : place {n} = {NOMS[vu]} au lieu de {nom}")
                commun.say("Mauvaise couleur")
                essai = 0
                continue
            ailleurs = [i + 1 for i in range(NB_PLACES) if avant.places[i] is None and apres.places[i] is not None]
            if ailleurs:
                self.noter(f"place {n} vide, mais tube posé en place {', '.join(map(str, ailleurs))}")
            elif apres.compte[c] < avant.compte[c]:
                self.noter(f"place {n} vide mais un tube {nom} de moins sur la table : tombé ou posé ailleurs")
            else:
                self.journal.log(f"raté : place {n} toujours vide")
            if essai >= self.args.max_tries:
                self.noter(f"échec du tube {nom} (place {n}) après {essai} essais, couleur suivante")
                commun.say(f"Échec du tube {nom}, couleur suivante")
                return True
            commun.say("Tube perdu, je recommence" if apres.compte[c] < avant.compte[c] else "Raté, je recommence")

    def bilan(self, ordre, statut, duree):
        try:
            fin = observer(self.source, self.calib)
            self.image(fin, "fin", f"Fin ({statut})")
        except (Exception, KeyboardInterrupt) as e:  # caméra perdue ou second Ctrl+C : dernière vue connue
            self.journal.log(f"pas d'image finale ({e or type(e).__name__})")
            fin = self.vue
        n = len(self.ranges)
        ranges = f"{pluriel(n, 'tube')} rangé{'s' if n > 1 else ''}" if n else "aucun tube rangé"
        self.journal.log(f"===== Bilan ({statut}) : {ranges} en {pluriel(self.essais, 'essai')}, {duree / 60:.1f} min")
        if fin is not None:
            attendu = cible(fin.places, fin.compte, ordre)
            self.journal.log(f"porte-tubes  : {texte_places(fin.places)}")
            self.journal.log(f"cible ({ordre})  : {texte_places(attendu)}")
            self.journal.log(f"score estimé : {score(fin.places, attendu)}/{sum(p is not None for p in attendu)}"
                             f" (places conformes à la cible)")
            self.journal.log(f"reste sur la table : {texte_compte(fin.compte)}")
        for msg in self.incidents:
            self.journal.log(f"incident : {msg}")
        self.journal.log(f"images et journal : {self.journal.dir}")
        try:
            commun.say(f"{'Tri terminé' if statut == 'terminé' else 'Tri interrompu'}, {ranges}")
        except KeyboardInterrupt:
            pass


def run(args):
    ordre = args.order.upper()
    if len(ordre) != 3 or set(ordre) != set(COULEURS):
        sys.exit(f"ERREUR : --order {args.order} : les trois lettres B, J, R dans l'ordre voulu (ex. BJR, RBJ)")
    if args.max_tries < 1:
        sys.exit("ERREUR : --max-tries doit valoir au moins 1")
    if args.image and not args.dry_run:
        sys.exit("ERREUR : --image (image fixe) seulement avec --dry-run")
    calib_ = charger_calib()
    if not calib_ok(calib_):
        commun.die(f"calibration TRI absente ou incomplète ({CALIB}) : lance make tri-calib")
    journal = commun.Journal("tri")
    journal.log(f"Épreuve TRI, ordre {ordre} ({', '.join(NOMS[c] for c in ordre)}) ; modèle {args.policy} "
                f"à {args.fps} fps, {args.seconds:g} s par essai, {args.max_tries} essais max par tube"
                + (f", serveur {args.async_server}" if args.async_server else ", CPU du PC")
                + (" [TEST À BLANC : le bras ne bouge pas]" if args.dry_run else ""))
    source = ImageFixe(args.image) if args.image else commun.Oak()
    try:
        ok = Tri(args, calib_, source, journal).executer(ordre)
    finally:
        source.close()
    if not ok:
        sys.exit(130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="épreuve complète")
    p.add_argument("--order", required=True, help="couleurs de gauche à droite vu du robot, ex. BJR")
    commun.parse_common_args(p)
    p.add_argument("--text-b", required=True, help="phrase du modèle pour un tube bleu")
    p.add_argument("--text-j", required=True, help="phrase du modèle pour un tube jaune")
    p.add_argument("--text-r", required=True, help="phrase du modèle pour un tube rouge")
    p.add_argument("--image", help="test : image fixe au lieu de l'OAK (avec --dry-run)")
    p.set_defaults(func=run)

    p = sub.add_parser("calib", help="calibration à la souris (ou --slots/--table)")
    p.add_argument("--slots", help='sans souris : les 6 places "x,y;x,y;...", de gauche à droite vu du robot')
    p.add_argument("--table", help='sans souris : coins de la zone de la table "x,y;x,y;..."')
    p.set_defaults(func=calib)

    p = sub.add_parser("voir", help="vision en direct")
    p.add_argument("--once", metavar="IMAGE.jpg", help="enregistre une image annotée sans fenêtre, puis quitte")
    p.add_argument("--image", help="image fixe au lieu de l'OAK")
    p.set_defaults(func=voir)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()



















    """Épreuve TRI : ranger les tubes de couleur dans le porte-tubes, groupés selon un ordre donné.

  make tri ORDER=BJR [DRY=1]   épreuve complète (DRY=1 : vision réelle, aucun mouvement du bras)
  make tri-calib               une fois : cliquer les 6 places du porte-tubes puis la zone de la table
  make tri-voir                vision en direct, pour vérifier la calibration et régler les seuils

Stratégie : couleur par couleur dans l'ordre demandé (B bleu/cyan, J jaune, R rouge/magenta), toujours
dans la place libre la plus à gauche vue du robot. Le modèle connaît une phrase par couleur et choisit
lui-même la place ; la vision (OAK, seuillage HSV) compte les tubes restant sur la table, désigne la
place attendue et vérifie le résultat après chaque essai.

Calibration et seuils : TRI/tri_calib.json (éditable à la main, voir la clé "_aide").
Sans souris : tri.py calib --slots "x,y;x,y;..." --table "x,y;x,y;..." ; tri.py voir --once vue.jpg
"""