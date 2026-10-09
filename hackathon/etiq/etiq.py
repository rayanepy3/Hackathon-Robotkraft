import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import unicodedata

import cv2
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "COMMUN"))
import commun  # noqa: E402
from commun import say

CALIB = commun.ROOT / "ETIQ" / "etiq_calib.json"
NB_PLACES = 6
SEUIL = float(os.environ.get("ETIQ_SEUIL", 0.75))  # ressemblance minimale mot lu / mot demandé
POIGNET_INVERSE = os.environ.get("ETIQ_POIGNET_INVERSE", "") not in ("", "0")
CONF_MIN = 0.4       # confiance OCR minimale d'une variante prise en compte
CONF_MOT = 0.6       # confiance pour compter un texte comme « mot lu » (bilans, étiquette lisible)
CHANGEMENT = 0.20    # part des pixels changés dans une région pour dire « l'objet est parti / revenu »
OCCUPE = 0.5         # part des pixels du trou différents du rack vide pour dire la place occupée
                     # (mesuré : tube 1,0 ; place vide 0 à 0,23, bord d'un bouchon voisin)
RACK_BOUGE = 0.55    # au-delà, la zone du rack ne ressemble plus au rack vide mémorisé : il a bougé
ATTENTE_S = 1.5      # après un geste, le temps que le bras et l'image se posent
ZOOM_CALIB = 2       # la fenêtre de calibration affiche l'image agrandie (clics plus précis)

VERT, ORANGE, BLEU = (60, 200, 60), (0, 165, 255), (255, 160, 60)


# ---------------------------------------------------------------------------
# Mots : normalisation et ressemblance tolérante aux erreurs d'OCR
# ---------------------------------------------------------------------------
_CHIFFRES = str.maketrans("0125678", "OIZSGTB")
_PROCHES = {frozenset(p) for p in ("OD", "OQ", "OC", "OG", "DQ", "CG", "IT", "IL", "IJ", "LJ", "TY",
                                   "UV", "VY", "MN", "NH", "HM", "BE", "EF", "FP", "PR", "RK", "BR", "SZ")}


def normalise(texte):
    """Majuscules, sans accents ni espaces ; chiffres lus à la place de lettres (0->O, 1->I...)."""
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z]", "", t.translate(_CHIFFRES))


def ressemblance(lu, mot):
    """1 = mot exact, 0 = rien à voir. Distance d'édition où les confusions classiques de l'OCR
    coûtent 0,5 ; des lettres en trop avant ou après le mot (étiquettes voisines) sont gratuites."""
    if not lu or not mot:
        return 0.0
    prev = [0.0] * (len(lu) + 1)  # début libre dans le texte lu
    for i, a in enumerate(mot, 1):
        cur = [float(i)] + [0.0] * len(lu)
        for j, b in enumerate(lu, 1):
            sub = 0.0 if a == b else (0.5 if frozenset((a, b)) in _PROCHES else 1.0)
            cur[j] = min(prev[j - 1] + sub, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return max(0.0, 1 - min(prev) / len(mot))  # fin libre aussi


def score(lecture, mot):
    """(ressemblance, texte) de la meilleure variante d'une lecture."""
    best = (0.0, "")
    for texte, conf in lecture["variantes"]:
        if conf >= CONF_MIN:
            best = max(best, (ressemblance(texte, mot), texte))
    return best


def est_mot(L):
    return L["conf"] >= CONF_MOT and len(L["texte"]) >= 3


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def _ordonne(q):
    """Coins dans l'ordre horaire, en partant du coin haut-gauche."""
    q = np.asarray(q, np.float32)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]  # horaire (y vers le bas)
    return np.roll(q, -int(np.argmin(q.sum(1))), 0)


def _agrandit(q, marge):
    """Élargit le quadrilatère de marge x (petit côté) dans ses propres axes."""
    u, v = q[1] - q[0], q[3] - q[0]
    lu, lv = np.linalg.norm(u) or 1, np.linalg.norm(v) or 1
    m = marge * min(lu, lv)
    u, v = u / lu * m, v / lv * m
    return np.array([q[0] - u - v, q[1] + u - v, q[2] + u + v, q[3] - u + v], np.float32)


def _redresse(img, q, k, h=48):
    """Recadre le quadrilatère q à plat, lu en partant du coin k (0 = normal, 2 = à l'envers,
    1 et 3 = texte vertical de haut en bas / de bas en haut). Hauteur h (entrée du modèle)."""
    q = np.roll(q, -k, 0)
    w0 = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h0 = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]), 1)
    w = int(np.clip(round(w0 * h / h0), 8, 20 * h))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = *a.min(0), *a.max(0)
    bx1, by1, bx2, by2 = *b.min(0), *b.max(0)
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6)


def rect_rack(slots, w, h, marge=0.3):
    """Rectangle englobant toutes les places, agrandi."""
    a = np.array(slots, float)
    x1, y1, x2, y2 = a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()
    mx, my = (x2 - x1) * marge, (y2 - y1) * marge
    return int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))


class Lecteur:
    """RapidOCR : détection des boîtes de texte, puis reconnaissance de chaque boîte redressée dans
    plusieurs sens, avec deux marges et deux contrastes. Une lecture = une boîte :
    {quad, centre, texte, conf (meilleure variante), variantes [(texte, conf)], place}
    place = k pour la lecture directe du rectangle de la place k du rack, None pour une boîte détectée."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.eng = RapidOCR()
        self.eng.text_rec.rec_batch_num = 16
        self.clahe = cv2.createCLAHE(2.0, (4, 4))

    def _detecte(self, img, echelle=1.0):
        if echelle != 1:
            img = cv2.resize(img, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_CUBIC)
        boites, _ = self.eng.text_det(img)
        if boites is None or len(boites) == 0:
            return []
        return [_ordonne(np.asarray(b, np.float32) / echelle) for b in boites]

    tourne = False  # détection aussi sur l'image tournée de 90° (+0,3 s ; rien gagné sur nos images)

    def detecte(self, img, slots=()):
        """Boîtes de texte : image entière x2 (le détecteur trouve aussi le texte vertical ; le sens de
        lecture est essayé ensuite boîte par boîte), et rack agrandi."""
        h, w = img.shape[:2]
        quads = self._detecte(img, 2.0)
        if self.tourne:
            for q in self._detecte(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 2.0):
                quads.append(_ordonne(np.stack([q[:, 1], h - 1 - q[:, 0]], 1)))  # retour au repère d'origine
        if len(slots):
            x1, y1, x2, y2 = rect_rack(slots, w, h)
            quads += [q + np.float32([x1, y1]) for q in self._detecte(img[y1:y2, x1:x2])]  # agrandi par RapidOCR
        gardes = []
        for q in quads:
            if min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])) >= 4 \
                    and all(_iou(q, g) < 0.7 for g in gardes):
                gardes.append(q)
        return gardes

    @staticmethod
    def _sens(q):
        cotes = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
        if max(cotes) < 1.6 * min(cotes):
            return 0, 1, 2, 3  # boîte presque carrée : sens de lecture inconnu
        return (0, 2) if cotes[0] >= cotes[1] else (1, 3)  # le long du grand côté, dans les deux sens

    def _variantes(self, img, q, sens, marges, contraste):
        crops = []
        for marge in marges:
            qq = _agrandit(q, marge)
            for k in sens:
                c = _redresse(img, qq, k)
                crops.append(c)
                if contraste:
                    g = self.clahe.apply(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
                    crops.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        return crops

    def _reconnait(self, img, lectures, idx, marges, contraste):
        crops, qui = [], []
        for i in idx:
            L = lectures[i]
            v = self._variantes(img, L["quad"], L["sens"], marges, contraste)
            crops += v
            qui += [i] * len(v)
        for i, (texte, conf) in zip(qui, self.eng.text_rec(crops)[0] if crops else []):
            t, c = normalise(texte), float(conf)
            if t:
                L = lectures[i]
                L["variantes"].append((t, c))
                if c > L["conf"]:
                    L["texte"], L["conf"] = t, c

    def lit(self, img, slots=()):
        """Toutes les lectures d'une image BGR. slots : places du rack, lues en plus une par une
        telles quelles (0°, 90° et 270° : texte en travers ou le long du tube).
        Passe 1 : une variante par sens de lecture ; passe 2, pour les boîtes lues avec 3 lettres ou
        plus et une confiance de 0,4 à 0,9 : deux marges et deux contrastes (étiquettes petites ou floues)."""
        lectures = [{"quad": q, "place": None, "sens": self._sens(q)} for q in self.detecte(img, slots)]
        for k, (x1, y1, x2, y2) in enumerate(slots, 1):
            lectures.append({"quad": np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), "place": k,
                             "sens": (0, 1, 3)})
        for L in lectures:
            L.update(centre=L["quad"].mean(0), variantes=[], texte="", conf=0.0)
        self._reconnait(img, lectures, range(len(lectures)), (0.25,), False)
        faibles = [i for i, L in enumerate(lectures)  # texte plausible mais peu sûr (pas les boîtes parasites)
                   if L["place"] is None and 0.4 <= L["conf"] < 0.9 and len(L["texte"]) >= 3]
        self._reconnait(img, lectures, faibles, (0.12, 0.4), True)
        return [L for L in lectures if L["variantes"]]


# ---------------------------------------------------------------------------
# Calibration : places du rack (rectangles des étiquettes), zones de la table (centres),
# et, facultatif, une image du rack vide (pour savoir quelles places sont occupées)
# ---------------------------------------------------------------------------
def charge_calib(obligatoire=True):
    if CALIB.exists():
        return json.loads(CALIB.read_text())
    if obligatoire:
        commun.die(f"calibration absente ({CALIB}) : make etiquette-calib")
    return {"slots": [], "zones": []}


def capture_vide(img, slots):
    """Rack vide : recadrage JPEG (base64) du rectangle englobant les places."""
    x1, y1, x2, y2 = rect_rack(slots, img.shape[1], img.shape[0])
    ok, buf = cv2.imencode(".jpg", img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"rect": [x1, y1, x2, y2], "jpg": base64.b64encode(buf.tobytes()).decode()}


def sauve_calib(slots, zones, taille, provisoire=False, vide=None):
    data = {"slots": [[int(v) for v in s] for s in slots], "zones": [[int(v) for v in z] for z in zones],
            "taille": [int(v) for v in taille], "date": time.strftime("%Y-%m-%d %H:%M"),
            "aide": "slots : 6 rectangles [x1,y1,x2,y2] de l'étiquette d'un tube rangé, du haut de l'étiquette "
                    "au trou du rack, place 1 à 6 de gauche à droite vu du robot ; zones : centres [x,y] "
                    "numérotés 1..n ; image OAK 640x480. "
                    "vide (facultatif) : image du rack sans tube."}
    if provisoire:
        data["provisoire"] = "calibration provisoire, faite sans tubes rangés : refaire make etiquette-calib"
    if vide:
        data["vide"] = vide
    CALIB.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"Calibration sauvée : {CALIB} ({len(slots)} places, {len(zones)} zones, "
          f"rack vide {'mémorisé' if vide else 'non mémorisé'})")


def image_vide(calib, taille):
    """Image (h, w de l'OAK) contenant le rack vide mémorisé, ou None (absent ou périmé)."""
    vide = calib.get("vide")
    if not vide or len(calib["slots"]) != NB_PLACES:
        return None
    x1, y1, x2, y2 = vide["rect"]
    ref = cv2.imdecode(np.frombuffer(base64.b64decode(vide["jpg"]), np.uint8), cv2.IMREAD_COLOR)
    if ref is None or ref.shape[:2] != (y2 - y1, x2 - x1) or y2 > taille[0] or x2 > taille[1]:
        return None
    if not all(x1 <= s[0] and y1 <= s[1] and s[2] <= x2 and s[3] <= y2 for s in calib["slots"]):
        return None  # places recalibrées depuis
    img = np.zeros((taille[0], taille[1], 3), np.uint8)
    img[y1:y2, x1:x2] = ref
    return img


def rack_bouge(img, calib):
    """Part de la zone du rack différente du rack vide mémorisé (None sans image de rack vide).
    Mesuré : rack en place 0 à 0,37 (tubes rangés, objets derrière), rack déplacé 0,68."""
    ref = image_vide(calib, img.shape[:2])
    return None if ref is None else changement(ref, img, calib["vide"]["rect"])


def occupees(img, calib):
    """{place: occupée ?} par comparaison au rack vide mémorisé ; None sans image de rack vide, ou si
    le rack semble avoir bougé depuis la calibration."""
    ref = image_vide(calib, img.shape[:2])
    if ref is None or rack_bouge(img, calib) > RACK_BOUGE:
        return None
    return {k: changement(ref, img, rect_trou(s)) >= OCCUPE for k, s in enumerate(calib["slots"], 1)}


def rect_trou(s):
    """Bas du rectangle d'une place, au centre : là où le tube entre dans le rack. Peu de fond y est
    visible, et les places voisines ne le recouvrent pas : c'est là qu'on juge si la place est occupée."""
    x1, y1, x2, y2 = s
    cx, dx = (x1 + x2) / 2, (x2 - x1) * 0.3
    return int(cx - dx), int(y2 - (y2 - y1) * 0.3), int(cx + dx), int(y2)


def place_de(p, slots, marge=0.25):
    """Place (1..6) dont le rectangle (agrandi) contient le point p, ou None."""
    best = None
    for k, (x1, y1, x2, y2) in enumerate(slots, 1):
        mx, my = (x2 - x1) * marge, (y2 - y1) * marge
        if x1 - mx <= p[0] <= x2 + mx and y1 - my <= p[1] <= y2 + my:
            d = np.hypot(p[0] - (x1 + x2) / 2, p[1] - (y1 + y2) / 2)
            best = min(best or (d, k), (d, k))
    return best and best[1]


def zone_de(p, zones):
    """Zone (1..n) dont le centre est le plus proche du point p."""
    if not zones:
        return None
    return 1 + int(np.argmin([np.hypot(p[0] - x, p[1] - y) for x, y in zones]))


def region_de(L, calib, ou):
    """Région d'une lecture : numéro de place (rack) ou de zone (table)."""
    if ou == "rack":
        return L["place"] or place_de(L["centre"], calib["slots"])
    return zone_de(L["centre"], calib["zones"])


def nom_region(ou, k):
    return f"{'place' if ou == 'rack' else 'zone'} {k}"


# ---------------------------------------------------------------------------
# Images annotées
# ---------------------------------------------------------------------------
def _texte(img, s, org, couleur, echelle=0.42):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, echelle, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur, 1, cv2.LINE_AA)


def _ascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def annote(img, calib=None, ou=None, lectures=(), mot=None, cible=None, titre="", rect=None):
    """Places/zones calibrées, boîtes lues (texte + confiance), cible en vert, titre en haut.
    ou=None : places et zones ; ou='' : ni l'une ni l'autre (caméra poignet)."""
    out = img.copy()
    calib = calib or {"slots": [], "zones": []}
    if ou in (None, "rack"):
        for k, (x1, y1, x2, y2) in enumerate(calib["slots"], 1):
            c = VERT if (ou, cible) == ("rack", k) else BLEU
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2 if c == VERT else 1)
            t = rect_trou((x1, y1, x2, y2))
            cv2.rectangle(out, t[:2], t[2:], c, 1)
            _texte(out, str(k), (x1, y2 + 12), c)
    if ou in (None, "table"):
        for k, (x, y) in enumerate(calib["zones"], 1):
            c = VERT if (ou, cible) == ("table", k) else BLEU
            cv2.circle(out, (x, y), 6, c, 2 if c == VERT else 1)
            _texte(out, f"Z{k}", (x + 8, y + 4), c)
    if rect:
        cv2.rectangle(out, tuple(rect[:2]), tuple(rect[2:]), ORANGE, 1)
    for L in lectures:
        s, t = score(L, mot) if mot else (0.0, "")
        if s >= SEUIL:
            couleur, etiquette = VERT, f"{t} {s:.2f}"
        elif est_mot(L):
            couleur, etiquette = ORANGE, f"{L['texte']} {L['conf']:.2f}"
        else:
            continue  # bruit : pas affiché
        if L["place"] is None or s >= SEUIL:
            cv2.polylines(out, [L["quad"].astype(np.int32)], True, couleur, 1)
        reg = ""
        if ou is None:  # voir : place si dans le rack, sinon zone
            k = L["place"] or place_de(L["centre"], calib["slots"])
            reg = f" P{k}" if k else (f" Z{zone_de(L['centre'], calib['zones'])}" if calib["zones"] else "")
        elif ou and region_de(L, calib, ou):
            reg = f" {'P' if ou == 'rack' else 'Z'}{region_de(L, calib, ou)}"
        x, y = L["quad"].min(0).astype(int)
        _texte(out, etiquette + reg, (int(x), max(32, int(y) - 4)), couleur)
    if titre:
        cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
        _texte(out, _ascii(titre), (5, 14), (255, 255, 255), 0.45)
    return out


# ---------------------------------------------------------------------------
# Vision : recherche, vérifications
# ---------------------------------------------------------------------------
def cherche(lecteur, images, mot, calib, ou):
    """Lit plusieurs images ; pour chaque région (place ou zone), la meilleure ressemblance au mot.

    Renvoie (trouve, regions, lectures) ; regions[k] = {k, score, texte, vu (images où le mot y est
    lu), quad, lu (meilleur mot lisible, conf), boite (texte détecté dans la région)} ; trouve = la
    région gagnante ou None ; lectures = toutes les lectures (toutes images).
    """
    regions, toutes = {}, []
    slots = calib["slots"] if ou == "rack" else ()
    for img in images:
        lectures = [L for L in lecteur.lit(img, slots) if region_de(L, calib, ou)]
        toutes += lectures
        vus = set()
        for L in lectures:
            k = region_de(L, calib, ou)
            s, t = score(L, mot)
            r = regions.setdefault(k, {"k": k, "score": 0.0, "texte": "", "vu": 0, "quad": None,
                                       "lu": None, "boite": False})
            if s >= SEUIL and k not in vus:
                vus.add(k)
                r["vu"] += 1
            if s > r["score"]:
                r.update(score=s, texte=t, quad=L["quad"])
            if L["place"] is None:
                r["boite"] = True
            if est_mot(L) and L["conf"] > (r["lu"] or ("", 0))[1]:
                r["lu"] = (L["texte"], L["conf"])
    ok = [r for r in regions.values() if r["score"] >= SEUIL]
    trouve = max(ok, key=lambda r: (r["vu"], r["score"])) if ok else None
    return trouve, regions, toutes


def mots_lus(regions, ou):
    """Bilan lisible : « place 2 : ARNICA (0.91) ; ... »."""
    morceaux = [f"{nom_region(ou, k)} : {r['lu'][0]} ({r['lu'][1]:.2f})"
                for k, r in sorted(regions.items()) if r.get("lu")]
    return " ; ".join(morceaux) or "aucun"


def rect_objet(cible, calib, ou, w, h):
    """Région surveillée : trou de la place du rack, ou autour de l'étiquette trouvée sur la table."""
    if ou == "rack":
        x1, y1, x2, y2 = rect_trou(calib["slots"][cible["k"] - 1])
    else:
        # l'étiquette elle-même (+30 %) : un tube transparent change peu l'image, son étiquette blanche si
        (x1, y1), (x2, y2) = cible["quad"].min(0), cible["quad"].max(0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = max(12, (x2 - x1) * 0.8), max(12, (y2 - y1) * 0.8)
        x1, y1, x2, y2 = cx - dx, cy - dy, cx + dx, cy + dy
    return int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))


def changement(avant, apres, rect):
    """Part des pixels nettement changés dans le rectangle (0 = identique, 1 = tout a changé) :
    écart de couleur Lab > 20 (bruit entre deux images d'une même scène : moins de 2 % des pixels)."""
    x1, y1, x2, y2 = (int(v) for v in rect)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a, b = (cv2.GaussianBlur(cv2.cvtColor(i[y1:y2, x1:x2], cv2.COLOR_BGR2Lab).astype(np.float32), (5, 5), 0)
            for i in (avant, apres))
    return float((np.sqrt(((a - b) ** 2).sum(2)) > 20).mean())


def a_sa_place(lectures, mot, cible, calib, ou):
    """Le mot est-il encore lu à l'endroit d'origine de l'objet ?"""
    for L in lectures:
        if score(L, mot)[0] < SEUIL:
            continue
        if ou == "rack" and region_de(L, calib, ou) == cible["k"]:
            return True
        if ou == "table":
            rayon = max(25.0, 0.75 * float(np.ptp(cible["quad"], 0).max()))
            if np.hypot(*(L["centre"] - cible["quad"].mean(0))) <= rayon:
                return True
    return False


def nouvelle(L, statiques, rayon=25):
    """Boîte détectée là où rien n'était lu avant : texte apporté par le bras (objet montré)."""
    return L["place"] is None and all(np.hypot(*(L["centre"] - c)) >= rayon for c in statiques)


class Vigie(threading.Thread):
    """Lit l'OAK en continu pendant un geste (le pont OAK est partagé avec le modèle) et garde le
    meilleur résultat : le mot cherché, ou à défaut les autres mots nouveaux lus avec confiance."""

    def __init__(self, oak, lecteur, mot, statiques):
        super().__init__(daemon=True)
        self.oak, self.lecteur, self.mot, self.statiques = oak, lecteur, mot, statiques
        self.fin = threading.Event()
        self.n, self.erreur = 0, None
        self.best = (0.0, "", None, [])  # ressemblance, texte, image, lectures
        self.partiel = 0.0               # meilleure ressemblance sous le seuil (lecture partielle ?)
        self.autres = {}                 # mot lu -> confiance
        self.derniere = (None, [])

    def examine(self, img, lectures):
        self.n += 1
        self.derniere = (img, lectures)
        for L in lectures:
            if not nouvelle(L, self.statiques):
                continue
            s, t = score(L, self.mot)
            if s >= SEUIL and s > self.best[0]:
                self.best = (s, t, img, lectures)
            self.partiel = max(self.partiel, s)
            if L["conf"] >= 0.8 and len(L["texte"]) >= 4 and ressemblance(L["texte"], self.mot) < 0.5:
                self.autres[L["texte"]] = max(self.autres.get(L["texte"], 0), L["conf"])

    def run(self):
        try:
            while not self.fin.is_set():
                img = self.oak.frame(timeout_s=5)
                self.examine(img, self.lecteur.lit(img))
        except Exception as e:  # noqa: BLE001 : la lecture ne doit jamais casser l'épreuve
            self.erreur = e

    def arrete(self):
        self.fin.set()
        self.join()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
class Etiq:
    def __init__(self, args, mot, calib, journal, lecteur, oak):
        self.a, self.mot, self.calib, self.j, self.lecteur, self.oak = args, mot, calib, journal, lecteur, oak
        self.ou = args.ou
        self.slots = calib["slots"] if self.ou == "rack" else []
        self.policy_show = args.policy_show or args.policy
        self.policy_place = args.policy_place or args.policy
        self.policy_look = args.policy_look or self.policy_show
        self.gestes = 0
        self.img0 = None      # image de départ (bras au repos) : référence « place pleine »
        self.statiques = []   # centres des textes de la scène fixe (pour reconnaître l'objet montré)

    # -- outils
    def images(self, n=2, pause=0.2):
        out = []
        for i in range(n):
            if i:
                time.sleep(pause)
            out.append(self.oak.frame())
        return out

    def geste(self, policy, texte, vigie=None):
        self.gestes += 1
        if vigie:
            vigie.start()
        try:
            commun.run_policy(policy, texte, self.a.seconds, self.a.fps, self.a.async_server, self.a.dry_run, self.j)
            time.sleep(ATTENTE_S)
        finally:
            if vigie:
                vigie.arrete()

    def photo(self, img, nom, titre, lectures=(), cible=None, rect=None):
        self.j.image(annote(img, self.calib, self.ou, lectures, self.mot, cible, titre, rect), nom)

    def lit(self, imgs, places=True):
        return [L for img in imgs for L in self.lecteur.lit(img, self.slots if places else ())]

    def fin(self, ok, msg, voix):
        self.j.log(f"BILAN : {msg}")
        say(voix)
        if self.a.dry_run and not ok and self.gestes:
            self.j.log("test à blanc : le bras n'a pas bougé, l'échec est attendu ; workflow terminé")
            return 0
        return 0 if ok else 1

    # -- niveau 1 : lecture à distance
    def niveau1(self):
        t0 = time.time()
        imgs = self.images(self.a.images)
        self.img0 = imgs[0]
        cible, regions, lectures = cherche(self.lecteur, imgs, self.mot, self.calib, self.ou)
        self.statiques = [L["centre"] for L in lectures if L["place"] is None]
        self.j.log(f"niveau 1 (lecture OAK à distance, {len(imgs)} images, {time.time() - t0:.1f} s) : "
                   f"mots lus : {mots_lus(regions, self.ou)}")
        if self.ou == "rack":
            b, occ = rack_bouge(self.img0, self.calib), occupees(self.img0, self.calib)
            if b is None:
                self.j.log("rack vide non mémorisé (make etiquette-calib, touche v) : toutes les places sont candidates")
            elif b > RACK_BOUGE:
                self.j.log(f"ATTENTION : le rack semble avoir bougé depuis la calibration ({b:.0%} de sa zone "
                           f"différente du rack vide) : places peut-être fausses, refaire make etiquette-calib")
                say("Attention, le rack a bougé")
            else:
                self.j.log(f"places occupées : {[k for k, v in occ.items() if v]}")
        if cible:
            self.j.log(f"niveau 1 : {self.mot} lu « {cible['texte']} » ({cible['score']:.2f}) en "
                       f"{nom_region(self.ou, cible['k'])}, sur {cible['vu']}/{len(imgs)} images")
        else:
            best = max(regions.values(), key=lambda r: r["score"], default=None)
            if best and best["score"] >= 0.4:
                self.j.log(f"niveau 1 : meilleur candidat {best['texte']} ({best['score']:.2f} < seuil {SEUIL}) "
                           f"en {nom_region(self.ou, best['k'])}")
        self.photo(imgs[-1], "niveau1", f"niveau 1 : {self.mot} " +
                   (nom_region(self.ou, cible["k"]) if cible else "non lu"), lectures, cible and cible["k"])
        if self.a.lecture == "poignet":
            self.poignet("depart")
        return cible, regions

    # -- sous-gestes
    def saisir(self, cible, mot_ici, budget, deja=0):
        """Geste SAISIR, jusqu'à budget essais (numérotés après les deja essais faits sur cette place) :
        l'objet a-t-il quitté sa place ? Renvoie (ok, essais faits, lectures après)."""
        k = cible["k"]
        texte = (self.a.text_slot if self.ou == "rack" else self.a.text_zone).replace("{k}", str(k))
        lectures = []
        for n in range(deja + 1, deja + budget + 1):
            say(f"Saisie, essai {n}")
            avant = self.oak.frame()
            self.photo(avant, f"saisir{k}_avant{n}", f"avant saisie {n} : {texte}", cible=k)
            self.geste(self.a.policy, texte)
            imgs = self.images(2)
            lectures = self.lit(imgs)
            h, w = imgs[-1].shape[:2]
            rect = rect_objet(cible, self.calib, self.ou, w, h)
            chg = changement(self.img0, imgs[-1], rect)
            encore = mot_ici and a_sa_place(lectures, self.mot, cible, self.calib, self.ou)
            parti = not encore and chg >= CHANGEMENT
            self.j.log(f"saisir {nom_region(self.ou, k)}, essai {n} : mot encore lu à sa place : "
                       f"{'oui' if encore else ('non' if mot_ici else '-')}, région changée à {chg:.0%} "
                       f"(seuil {CHANGEMENT:.0%}) -> {'pris' if parti else 'raté'}")
            self.photo(imgs[-1], f"saisir{k}_apres{n}", f"apres saisie {n} : {'pris' if parti else 'rate'}",
                       lectures, k, rect)
            if parti:
                say("Saisi")
                return True, n - deja, lectures
            if self.ou == "table" and mot_ici and not encore:  # l'objet a pu bouger sans être pris
                ail = [L for L in lectures if score(L, self.mot)[0] >= SEUIL]
                if ail:
                    k2 = region_de(ail[0], self.calib, self.ou)
                    self.j.log(f"{self.mot} a bougé : relu en zone {k2}, je vise cette zone")
                    cible.update(k=k2, quad=ail[0]["quad"])
                    k = k2
                    texte = self.a.text_zone.replace("{k}", str(k))
            if n < deja + budget:
                say("Raté, je recommence")
        return False, budget, lectures

    def montrer(self, cible, lectures_apres):
        """Geste MONTRER, OAK lue en continu pendant le geste puis bras posé.
        Renvoie (verdict, texte, image) : verdict 'bon' | 'autre' | 'pas_pris' | 'rien'."""
        statiques = self.statiques + [L["centre"] for L in lectures_apres if L["place"] is None]
        img = None
        for n in range(1, self.a.max_tries + 1):
            say("Je montre l'objet" if n == 1 else f"Je montre encore, essai {n}")
            v = Vigie(self.oak, self.lecteur, self.mot, statiques)
            t0 = time.time()
            self.geste(self.policy_show, self.a.text_show, v)
            img = self.oak.frame()
            v.examine(img, self.lecteur.lit(img))
            if v.erreur:
                self.j.log(f"montrer : lecture continue interrompue ({v.erreur})")
            h, w = img.shape[:2]
            reste = changement(self.img0, img, rect_objet(cible, self.calib, self.ou, w, h)) < CHANGEMENT
            autres = ", ".join(f"{t} ({c:.2f})" for t, c in sorted(v.autres.items(), key=lambda x: -x[1]))
            self.j.log(f"montrer, essai {n} : {v.n} images lues en {time.time() - t0:.0f} s ; {self.mot} : "
                       f"{v.best[1] + f' ({v.best[0]:.2f})' if v.best[1] else f'non lu (au mieux {v.partiel:.2f})'}"
                       f" ; autres mots nouveaux : {autres or 'aucun'} ; place d'origine inchangée : {'oui' if reste else 'non'}")
            if v.best[1]:
                self.j.image(annote(v.best[2], lectures=v.best[3], mot=self.mot, titre=f"montrer : {v.best[1]}"),
                             f"montrer{cible['k']}_{n}")
                return "bon", v.best[1], img
            vue = v.derniere[0] if v.derniere[0] is not None else img
            self.j.image(annote(vue, lectures=v.derniere[1], mot=self.mot, titre=f"montrer {n} : {self.mot} non lu"),
                         f"montrer{cible['k']}_{n}")
            if reste:
                return "pas_pris", "", img
            if v.autres and v.partiel < 0.5:
                return "autre", max(v.autres, key=v.autres.get), img
            if n < self.a.max_tries:
                say("Je ne lis rien, je recommence")
        return "rien", "", img

    def reposer(self, k, vide):
        """Geste REPOSER : la place k est-elle de nouveau occupée ? (comparée à l'image de départ,
        place pleine, et à une image de la place vide)."""
        rect = rect_trou(self.calib["slots"][k - 1])
        texte = self.a.text_place.replace("{k}", str(k))
        ref_vide = image_vide(self.calib, self.img0.shape[:2])
        vide = ref_vide if ref_vide is not None else vide
        for n in range(1, self.a.max_tries + 1):
            say(f"Je repose le tube, place {k}" if n == 1 else f"Je repose, essai {n}")
            self.geste(self.policy_place, texte)
            img = self.oak.frame()
            d_plein, d_vide = changement(self.img0, img, rect), changement(vide, img, rect)
            ok = d_vide >= CHANGEMENT / 2 and d_plein < d_vide
            self.j.log(f"reposer place {k}, essai {n} : écart à la place pleine {d_plein:.0%}, "
                       f"à la place vide {d_vide:.0%} -> {'reposé' if ok else 'raté'}")
            self.photo(img, f"reposer{k}_{n}", f"reposer {n} : {'ok' if ok else 'rate'}", cible=k)
            if ok:
                return True
        return False

    def regarder(self, regions):
        """Niveau 2 : geste REGARDER, puis lecture poignet (plusieurs images).
        Renvoie (place sûre ou None, places probables, {place: autre mot lu})."""
        occ = occupees(self.img0, self.calib)
        lectures = []
        for n in range(1, self.a.max_tries + 1):
            say("Je regarde le rack de haut")
            self.geste(self.policy_look, self.a.text_look)
            lectures, img = [], None
            for _ in range(3):
                try:
                    img = commun.wrist_frame()
                except RuntimeError as e:
                    self.j.log(f"regarder : {e}")
                    break
                lectures += self.lecteur.lit(img)
            if img is not None:
                self.j.image(annote(img, lectures=lectures, mot=self.mot, titre=f"regarder {n} (poignet)"),
                             f"regarder_{n}")
            if lectures:
                break
            self.j.log(f"regarder, essai {n} : aucun texte vu par la caméra poignet")
        if not lectures:
            return None, [], {}
        # une étiquette = un groupe de boîtes à la même position horizontale
        groupes = []
        for L in sorted(lectures, key=lambda L: L["centre"][0]):
            if groupes and L["centre"][0] - groupes[-1][-1]["centre"][0] < 20:
                groupes[-1].append(L)
            else:
                groupes.append([L])
        if POIGNET_INVERSE:
            groupes.reverse()
        etiq = []
        for g in groupes:
            s, t = max(score(L, self.mot) for L in g)
            lu = max(((L["texte"], L["conf"]) for L in g if est_mot(L)), key=lambda x: x[1], default=None)
            etiq.append({"score": s, "texte": t, "lu": lu, "x": float(np.mean([L["centre"][0] for L in g]))})
        places = [k for k in range(1, NB_PLACES + 1) if occ is None or occ[k]]
        sur = len(etiq) == len(places)
        self.j.log(f"niveau 2 : {len(etiq)} étiquettes vues par le poignet (gauche -> droite) : "
                   + ", ".join(f"{e['lu'][0] if e['lu'] else '?'}" for e in etiq)
                   + f" ; places {'occupées' if occ else 'supposées occupées'} : {places} ; attribution "
                   + ("par l'ordre (approximatif)" if sur else "ambiguë (nombres différents)"))
        i_mot = max(range(len(etiq)), key=lambda i: etiq[i]["score"])
        trouve = etiq[i_mot]["score"] >= SEUIL
        autres = {}
        if sur:
            autres = {places[i]: e["lu"] for i, e in enumerate(etiq)
                      if e["lu"] and e["lu"][1] >= 0.8 and ressemblance(e["lu"][0], self.mot) < 0.5}
            if trouve:
                k = places[i_mot]
                self.j.log(f"niveau 2 : {self.mot} lu « {etiq[i_mot]['texte']} » ({etiq[i_mot]['score']:.2f}), "
                           f"{i_mot + 1}e étiquette -> place {k} (approximatif)")
                return k, [], autres
        if not trouve:
            self.j.log(f"niveau 2 : {self.mot} non lu (au mieux {etiq[i_mot]['score']:.2f})")
            return None, [], autres
        # mot vu, place incertaine : places classées par position relative dans la rangée
        f = 0.5 if len(etiq) == 1 else i_mot / (len(etiq) - 1)
        probables = sorted(places, key=lambda k: abs((places.index(k) / max(1, len(places) - 1)) - f))
        self.j.log(f"niveau 2 : {self.mot} lu, place incertaine ; ordre d'essai : {probables}")
        return None, probables, autres

    def poignet(self, etape):
        """Lecture poignet de confirmation (--lecture poignet), pour information."""
        try:
            img = commun.wrist_frame()
        except RuntimeError as e:
            self.j.log(f"poignet : {e}")
            return False
        lectures = self.lecteur.lit(img)
        s, t = max((score(L, self.mot) for L in lectures), default=(0.0, ""))
        mots = sorted({L["texte"] for L in lectures if est_mot(L)})
        self.j.image(annote(img, lectures=lectures, mot=self.mot, titre=f"poignet {etape}"), f"poignet_{etape}")
        self.j.log(f"poignet ({etape}) : {self.mot} {'lu : ' + t if s >= SEUIL else 'non lu'} ({s:.2f}) ; "
                   f"mots lus : {', '.join(mots) or 'aucun'}")
        return s >= SEUIL

    # -- une place (ou zone) : saisir, montrer, reposer si ce n'est pas le bon objet
    def traite(self, cible, mot_ici, sur):
        """mot_ici : le mot a été lu à cette place par l'OAK ; sur : le mot a été localisé ici
        (niveau 1 ou 2) : sans MONTRER, ou si l'étiquette reste illisible de près, on garde l'objet.
        Renvoie (statut, texte lu de près) ; statut : 'bon', 'garde', 'autre', 'illisible',
        'rate' (saisie impossible), 'bloque' (objet tenu, impossible à reposer)."""
        fait = 0
        while fait < self.a.max_tries:
            ok, n, lectures = self.saisir(cible, mot_ici, self.a.max_tries - fait, fait)
            fait += n
            if not ok:
                return "rate", ""
            if self.a.lecture == "poignet":
                self.poignet(f"saisie_{nom_region(self.ou, cible['k']).replace(' ', '')}")
            if not self.a.text_show:
                return "garde", ""
            verdict, texte, img = self.montrer(cible, lectures)
            if verdict == "bon":
                return "bon", texte
            if verdict == "pas_pris":
                self.j.log("montrer : l'objet est resté à sa place, la saisie a échoué")
                say("Rien dans la pince, je recommence")
                continue
            if verdict == "rien" and sur:
                self.j.log("montrer : étiquette illisible de près ; localisée avant, je garde l'objet (non confirmé)")
                return "garde", ""
            if verdict == "autre":
                say(f"C'est {texte}, ce n'est pas {self.mot}")
            if self.ou != "rack" or not self.a.text_place:
                return "bloque", texte
            if not self.reposer(cible["k"], img):
                return "bloque", texte
            return ("autre" if verdict == "autre" else "illisible"), texte
        return "rate", ""

    # -- épreuve
    def rack(self):
        cible, regions = self.niveau1()
        faites = set()
        if cible:
            say(f"{self.mot} trouvé, place {cible['k']}")
            st, t = self.traite(cible, mot_ici=True, sur=True)
            r = self.conclut(st, t, cible["k"], 1)
            if r is not None:
                return r
            faites.add(cible["k"])
        probables, autres = [], {}
        if self.a.text_look:
            k, probables, autres = self.regarder(regions)
            if k and k not in faites:
                say(f"{self.mot} vu de haut, place {k}")
                st, t = self.traite({"k": k, "quad": None}, mot_ici=False, sur=True)
                r = self.conclut(st, t, k, 2)
                if r is not None:
                    return r
                faites.add(k)
        if not self.a.text_show:
            return self.fin(False, f"{self.mot} introuvable (niveau 3 désactivé : pas de geste MONTRER)",
                            "Mot introuvable")
        occ = occupees(self.img0, self.calib)

        def rang(k):  # (groupe, ordre dans le groupe, place, raison)
            r = regions.get(k, {})
            lu = autres.get(k) or r.get("lu")
            if k in probables:
                return 0, probables.index(k), k, "mot vu de haut"
            if lu and lu[1] >= 0.8:
                return 3, k, k, f"autre mot lu : {lu[0]}"
            if (occ and occ[k]) or r.get("boite") or lu:
                return 1, k, k, "étiquette illisible"
            return 2, k, k, "inconnue (peut-être vide)"
        candidats = [c[2:] for c in sorted(rang(k) for k in range(1, NB_PLACES + 1)
                                           if k not in faites and (occ is None or occ[k]))]
        if not self.a.text_place:
            self.j.log("pas de geste REPOSER : une seule place peut être vérifiée")
            candidats = candidats[:1]
        self.j.log("niveau 3 : places à vérifier : " + (" ; ".join(f"{k} ({why})" for k, why in candidats) or "aucune"))
        if candidats:
            say("Je vérifie les tubes un par un")
        for k, _ in candidats:
            st, t = self.traite({"k": k, "quad": None}, mot_ici=False, sur=False)
            r = self.conclut(st, t, k, 3)
            if r is not None:
                return r
        return self.fin(False, f"{self.mot} introuvable (niveaux 1 à 3)", "Mot introuvable")

    def table(self):
        cible, regions = self.niveau1()
        if not cible:
            return self.fin(False, f"{self.mot} introuvable sur la table", "Mot introuvable")
        say(f"{self.mot} trouvé, zone {cible['k']}")
        st, t = self.traite(cible, mot_ici=True, sur=True)
        r = self.conclut(st, t, cible["k"], 1)
        return r if r is not None else self.fin(False, f"mauvais objet en zone {cible['k']} ({t})", "Mauvais objet")

    def conclut(self, st, texte, k, niveau):
        """Fin de l'épreuve si le statut le permet, sinon None (on passe à la suite)."""
        reg = nom_region(self.ou, k)
        quoi = {1: "lecture OAK à distance", 2: "caméra poignet de haut", 3: "objet montré à l'OAK"}[niveau]
        if st == "bon":
            return self.fin(True, f"{self.mot} saisi et tenu ({reg}), trouvé au niveau {niveau} ({quoi}), "
                            f"confirmé de près : « {texte} » ; {self.gestes} gestes", f"J'ai {self.mot}")
        if st == "garde":
            return self.fin(True, f"{self.mot} saisi ({reg}), trouvé au niveau {niveau} ({quoi}), "
                            f"non confirmé de près ; {self.gestes} gestes", "Saisi")
        if st == "bloque":
            return self.fin(False, f"objet de la {reg} tenu ({texte or 'illisible'}) mais impossible à reposer",
                            "Mauvais objet")
        if st == "rate" and niveau == 1:  # au niveau 2 l'attribution est approximative : on continue
            return self.fin(False, f"{self.mot} localisé ({reg}, niveau {niveau}) mais saisie ratée "
                            f"après {self.a.max_tries} essais", f"Échec après {self.a.max_tries} essais")
        self.j.log(f"{reg} : " + {"rate": "saisie ratée" + ("" if self.calib.get("vide") else " (place vide ?)"),
                                  "autre": f"c'est {texte}, reposé",
                                  "illisible": "étiquette illisible même de près, reposé"}[st])
        return None


# ---------------------------------------------------------------------------
# Commandes
# ---------------------------------------------------------------------------
def run(args):
    if args.consigne == "cube":
        say("Consigne cube retourné : non gérée dans cette version")
        print("La variante « étiquette à l'envers sur le cube rouge » n'est pas encore gérée.")
        return 2
    mot = normalise(args.mot or "")
    if not mot:
        commun.die("mot vide : make etiquette MOT=IODURE")
    calib = charge_calib()
    if args.ou == "rack" and len(calib["slots"]) != NB_PLACES:
        commun.die(f"calibration du rack incomplète ({len(calib['slots'])} places) : make etiquette-calib")
    if args.ou == "table" and not calib["zones"]:
        commun.die("aucune zone de table calibrée : make etiquette-calib")
    for nom, phrase in (("--text-slot", args.text_slot), ("--text-zone", args.text_zone),
                        ("--text-place", args.text_place or "{k}")):
        if "{k}" not in phrase:
            commun.die(f"{nom} doit contenir {{k}} : « {phrase} »")

    journal = commun.Journal("etiq")
    gestes = [f"saisir « {args.text_slot if args.ou == 'rack' else args.text_zone} » ({args.policy})"]
    for nom, phrase, pol in (("montrer", args.text_show, args.policy_show or args.policy),
                             ("reposer", args.text_place, args.policy_place or args.policy),
                             ("regarder", args.text_look, args.policy_look or args.policy_show or args.policy)):
        gestes.append(f"{nom} « {phrase} » ({pol})" if phrase else f"{nom} désactivé")
    journal.log(f"ETIQ : mot {mot}, objets {args.ou}, lecture {args.lecture}, {args.fps} fps, {args.seconds} s "
                f"par geste, {args.max_tries} essais max, seuil {SEUIL}" + (" [TEST À BLANC]" if args.dry_run else ""))
    journal.log("gestes : " + " ; ".join(gestes))
    if calib.get("provisoire"):
        journal.log(f"ATTENTION : {calib['provisoire']}")
    say(f"Je cherche {mot}")
    lecteur = Lecteur()
    oak = commun.Oak()
    try:
        e = Etiq(args, mot, calib, journal, lecteur, oak)
        return e.rack() if args.ou == "rack" else e.table()
    except KeyboardInterrupt:
        journal.log("arrêt demandé (Ctrl+C)")
        say("Arrêt")
        return 130
    finally:
        oak.close()
        print(f"Journal et images : {journal.dir}")


def _parse_liste(s, n):
    """'x1,y1,x2,y2;...' -> [[x1,y1,x2,y2], ...]"""
    out = []
    for morceau in filter(None, (m.strip() for m in s.split(";"))):
        v = [int(round(float(x))) for x in morceau.split(",")]
        if len(v) != n:
            sys.exit(f"ERREUR : « {morceau} » : {n} nombres attendus")
        out.append(v)
    return out


def calib(args):
    old = charge_calib(obligatoire=False)
    if args.slots is None and args.zones is None and args.vide is None:
        return calib_souris(commun.Oak(), old)
    # sans souris (tests, ou valeurs connues)
    slots = [[min(a, c), min(b, d), max(a, c), max(b, d)] for a, b, c, d in _parse_liste(args.slots, 4)] \
        if args.slots is not None else old["slots"]
    zones = _parse_liste(args.zones, 2) if args.zones is not None else old["zones"]
    if slots and len(slots) != NB_PLACES:
        sys.exit(f"ERREUR : {len(slots)} places données, {NB_PLACES} attendues")
    img = cv2.imread(args.image) if args.image else commun.Oak().frame()
    vide = old.get("vide") if slots == old["slots"] else None
    if args.vide:
        if not slots:
            sys.exit("ERREUR : --vide demande les places du rack")
        vide = capture_vide(img if args.vide == "oak" else cv2.imread(args.vide), slots)
    sauve_calib(slots, zones, img.shape[1::-1], args.provisoire, vide)
    controle(img)
    return 0


def controle(img):
    out = commun.OUTPUTS / "etiq_calib.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cal = charge_calib()
    occ = occupees(img, cal)
    vis = annote(img, cal, titre="calibration ETIQ" + (" ; occupees : " + str([k for k, v in occ.items() if v])
                                                         if occ is not None else ""))
    cv2.imwrite(str(out), vis)
    print(f"Image de contrôle : {out}")


def calib_souris(oak, old):
    win = "ETIQ calibration"
    e = {"phase": "places", "places": [], "coin": None, "zones": [], "vide": None}

    def clic(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = x // ZOOM_CALIB, y // ZOOM_CALIB
        if e["phase"] == "places":
            if e["coin"] is None:
                e["coin"] = (x, y)
            else:
                (a, b), e["coin"] = e["coin"], None
                e["places"].append([min(a, x), min(b, y), max(a, x), max(b, y)])
                if len(e["places"]) == NB_PLACES:
                    e["phase"] = "zones"
        elif e["phase"] == "zones":
            e["zones"].append([x, y])

    print("Calibration ETIQ (fenêtre OpenCV), image de l'OAK :\n"
          "  1) rack : pour chaque place 1 à 6 (de gauche à droite vu du robot), 2 clics = coins opposés\n"
          "     du rectangle où se trouve l'étiquette d'un tube rangé : du haut de l'étiquette jusqu'au\n"
          "     trou du rack (le bas du rectangle sert à juger si la place est occupée) ;\n"
          "  2) table : un clic par centre de zone (zones 1..n dans l'ordre des clics), Entrée pour finir ;\n"
          "  3) facultatif mais conseillé : enlève les tubes du rack et appuie sur v (rack vide mémorisé) ;\n"
          "  s = sauver, u = annuler le dernier clic, Entrée sans clic = garder l'ancienne calibration, q = quitter.")
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, clic)
    while True:
        img = oak.frame(timeout_s=2, fresh=False)
        slots = e["places"] if e["places"] or e["phase"] == "places" else old["slots"]
        zones = e["zones"] if e["zones"] or e["phase"] != "fin" else old["zones"]
        vis = annote(img, {"slots": slots, "zones": zones if e["phase"] != "places" else []})
        if e["coin"]:
            cv2.circle(vis, e["coin"], 3, ORANGE, -1)
        if e["phase"] == "places":
            msg = [f"RACK place {len(e['places']) + 1}/6 (gauche->droite vu du robot) : 2 coins opposes de l'etiquette",
                   "Entree sans clic = garder l'ancien rack ; u = annuler ; q = quitter"]
        elif e["phase"] == "zones":
            msg = [f"TABLE : clique le centre de la zone {len(e['zones']) + 1}, Entree = fini",
                   "Entree sans clic = garder les anciennes zones ; u = annuler ; q = quitter"]
        else:
            vide = e["vide"] or (old.get("vide") if slots == old["slots"] else None)
            msg = [f"{len(slots)} places, {len(zones)} zones, rack vide {'memorise' if vide else 'NON memorise'}",
                   "v = memoriser le rack vide (tubes enleves) ; s = sauver ; u = reprendre ; q = quitter"]
        vis = cv2.resize(vis, None, fx=ZOOM_CALIB, fy=ZOOM_CALIB, interpolation=cv2.INTER_NEAREST)
        for i, m in enumerate(msg):
            _texte(vis, m, (8, vis.shape[0] - 12 - 22 * (len(msg) - 1 - i)), (255, 255, 255), 0.6)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("Calibration abandonnée (rien sauvé).")
            break
        if key == ord("u"):
            if e["phase"] == "fin":
                e["phase"] = "zones"
            elif e["phase"] == "zones" and e["zones"]:
                e["zones"].pop()
            elif e["coin"]:
                e["coin"] = None
            elif e["places"]:
                e["places"].pop()
                e["phase"] = "places"
        elif key in (10, 13):
            if e["phase"] == "places" and not e["places"] and not e["coin"]:
                e["places"] = [list(s) for s in old["slots"]]
                e["phase"] = "zones"
            elif e["phase"] == "zones":
                if not e["zones"]:
                    e["zones"] = [list(z) for z in old["zones"]]
                e["phase"] = "fin"
        elif key == ord("v") and e["phase"] == "fin" and len(slots) == NB_PLACES:
            e["vide"] = capture_vide(img, slots)
            print("Rack vide mémorisé.")
        elif key == ord("s") and e["phase"] == "fin":
            if len(slots) not in (0, NB_PLACES):
                print(f"Il faut {NB_PLACES} places (ou aucune) : {len(slots)} cliquées.")
                continue
            sauve_calib(slots, zones, img.shape[1::-1],
                        vide=e["vide"] or (old.get("vide") if slots == old["slots"] else None))
            controle(img)
            break
    cv2.destroyAllWindows()
    return 0


def voir(args):
    calib = charge_calib(obligatoire=False)
    lecteur = Lecteur()
    fichier = None
    if args.image:
        fichier = cv2.imread(args.image)
        if fichier is None:
            sys.exit(f"ERREUR : image illisible : {args.image}")
    vue_fixe = args.lecture == "oak"  # OAK (ou fichier vu comme une image OAK) : places et zones
    oak = commun.Oak() if vue_fixe and fichier is None else None
    mot = normalise(args.mot) if args.mot else None
    try:
        while True:
            img = fichier if fichier is not None else (oak.frame() if oak else commun.wrist_frame())
            t = time.time()
            lectures = lecteur.lit(img, calib["slots"] if vue_fixe else ())
            dt = time.time() - t
            vis = annote(img, calib if vue_fixe else None, None if vue_fixe else "", lectures, mot,
                         titre=f"{args.lecture} : lecture {dt:.1f} s" + (f", cherche {mot}" if mot else "")
                         + ("" if args.once else " (q = quitter)"))
            for L in sorted(lectures, key=lambda L: -L["conf"]):
                s = score(L, mot) if mot else (0, "")
                if est_mot(L) or s[0] >= SEUIL:
                    reg = ""
                    if vue_fixe:
                        k = L["place"] or place_de(L["centre"], calib["slots"])
                        reg = f"place {k}" if k else (f"zone {zone_de(L['centre'], calib['zones'])}" if calib["zones"] else "")
                    print(f"  {L['texte']:<12} conf {L['conf']:.2f}  ({int(L['centre'][0])},{int(L['centre'][1])}) "
                          f"{reg}" + (f"  {mot} : {s[1]} {s[0]:.2f}" if mot else ""))
            print(f"--- lecture {dt:.1f} s", flush=True)
            if args.once:
                cv2.imwrite(args.once, vis)
                print(f"Image annotée : {args.once}")
                return 0
            cv2.imshow("ETIQ lecture", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if oak:
            oak.close()
        if not args.once:
            cv2.destroyAllWindows()
    return 0


def main():
    p = argparse.ArgumentParser(description="Épreuve ETIQ : saisir l'objet dont l'étiquette porte le mot donné")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = commun.parse_common_args(sub.add_parser("run", help="épreuve complète"))
    r.add_argument("--mot", help="mot de l'étiquette à trouver (consigne 1)")
    r.add_argument("--consigne", choices=["mot", "cube"], default="mot",
                   help="mot = mot fourni ; cube = étiquette à l'envers sur le cube rouge (non géré)")
    r.add_argument("--ou", choices=["rack", "table"], default="rack", help="tubes du rack ou objets sur la table")
    r.add_argument("--lecture", choices=["oak", "poignet"], default="oak",
                   help="poignet : lecture de confirmation par la caméra poignet après chaque saisie "
                        "(la localisation reste à l'OAK)")
    r.add_argument("--text-slot", default="Pick up the tube in slot {k}.", help="SAISIR (rack), {k} = place 1..6")
    r.add_argument("--text-zone", default="Pick up the object in zone {k}.", help="SAISIR (table), {k} = zone")
    r.add_argument("--text-show", default="", help="MONTRER l'objet tenu à l'OAK (vide = désactivé)")
    r.add_argument("--text-place", default="", help="REPOSER le tube en place {k} (vide = désactivé)")
    r.add_argument("--text-look", default="", help="REGARDER le rack de haut, caméra poignet (vide = désactivé)")
    r.add_argument("--policy-show", default="", help="modèle de MONTRER (défaut : --policy)")
    r.add_argument("--policy-place", default="", help="modèle de REPOSER (défaut : --policy)")
    r.add_argument("--policy-look", default="", help="modèle de REGARDER (défaut : --policy-show)")
    r.add_argument("--images", type=int, default=3, help="images OAK lues pour localiser le mot")

    c = sub.add_parser("calib", help="places du rack et zones de la table (souris, ou --slots/--zones)")
    c.add_argument("--slots", help="'x1,y1,x2,y2;...' : les 6 places, de gauche à droite vu du robot")
    c.add_argument("--zones", help="'x,y;...' : centres des zones 1..n")
    c.add_argument("--vide", help="image du rack sans tube : 'oak' (image actuelle) ou un fichier")
    c.add_argument("--image", help="image de contrôle (défaut : image OAK actuelle)")
    c.add_argument("--provisoire", action="store_true", help="marque la calibration comme provisoire")

    v = sub.add_parser("voir", help="lecture des étiquettes en direct")
    v.add_argument("--lecture", choices=["oak", "poignet"], default="oak")
    v.add_argument("--mot", help="mot à surligner")
    v.add_argument("--once", metavar="OUT.jpg", help="une seule image annotée, sans fenêtre")
    v.add_argument("--image", help="lire ce fichier au lieu de la caméra (tests)")

    args = p.parse_args()
    if args.cmd == "run" and args.consigne == "mot" and not args.mot:
        p.error("--mot est obligatoire")
    sys.exit({"run": run, "calib": calib, "voir": voir}[args.cmd](args))


if __name__ == "__main__":
    main()







    """Épreuve ETIQ : saisir l'objet dont l'étiquette porte le mot donné.

  etiq.py run --mot IODURE --ou rack|table --lecture oak|poignet ...   l'épreuve complète
  etiq.py calib [--slots ... --zones ... --vide oak]                  places du rack, zones de la table
  etiq.py voir --lecture oak|poignet [--once out.jpg]                 lecture des étiquettes en direct

Partage du travail : la vision lit les étiquettes (RapidOCR) ; le modèle ne lit rien, il n'a appris
que des gestes, un par phrase, enchaînés par ce workflow (le bras garde sa pose entre deux gestes) :
  SAISIR    --policy        --text-slot  « Pick up the tube in slot {k}. »   (table : --text-zone)
  MONTRER   --policy-show   --text-show  « Show the tube to the camera. »    tube tenu devant l'OAK
  REPOSER   --policy-place  --text-place « Put the tube back in slot {k}. »  puis retour au repos
  REGARDER  --policy-look   --text-look  « Look at the rack from above. »    caméra poignet sur le rack
Une phrase vide désactive le geste (sans MONTRER : saisir seul, sans confirmation). Chaque geste a
ses propres essais (--max-tries), vérifiés à l'image.

Rack : on ne monte en difficulté que si l'étape d'avant échoue.
  Niveau 1 (aucun mouvement) : lecture OAK à distance sur plusieurs images. Mot trouvé en place k :
    saisir k, puis montrer pour confirmer.
  Niveau 2 (REGARDER) : le bras se lève, la caméra poignet regarde le rack de haut. Elle n'est pas
    calibrée : l'ordre gauche -> droite des étiquettes vues est rapproché de l'ordre des places
    occupées (APPROXIMATIF). Attribution sûre seulement si les nombres concordent.
  Niveau 3 (manipuler) : pour chaque place occupée (places les plus probables d'abord, puis
    étiquettes illisibles, puis inconnues, puis celles où un autre mot a été lu) : saisir k, montrer
    (l'OAK est lue en continu pendant le geste), bon mot -> FIN (tube tenu), sinon reposer k.
Table : niveau 1 seulement (zone = centre calibré le plus proche du mot lu), saisir, montrer si
  configuré ; pas de geste REPOSER sur la table : un mauvais objet arrête l'épreuve.

Caméras : l'OAK (vue fixe, pont permanent) localise, car elle seule est calibrée
(ETIQ/etiq_calib.json). La caméra poignet n'est ouverte qu'entre deux gestes (le modèle s'en sert
pendant les gestes) : niveau 2, et avec --lecture poignet une lecture de confirmation après chaque
saisie (pour information : elle voit souvent le dos du tube, son silence ne fait rien échouer).

Variante « étiquette à l'envers sur le cube rouge » : non gérée dans cette version (--consigne cube).
Réglages par variables d'environnement : ETIQ_SEUIL (0.75), ETIQ_POIGNET_INVERSE=1 (caméra poignet
montée à l'envers : l'ordre gauche -> droite des places y est inversé).
"""