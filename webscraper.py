#!/usr/bin/env python3
"""
Scrape local shops' mission statements and contact info.
Outputs:
  results_*/shops.json   — enriched structured results
  results_*/shops.csv    — spreadsheet-friendly table

"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
import re
import io
import csv
import sys
import json
import time
import math
import asyncio
import logging
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import html

import aiohttp
from bs4 import BeautifulSoup
import tldextract
import string
from urllib.parse import urlparse

# ---------------------------
# CLI / Config
# ---------------------------
import argparse

CATEGORIES_CANON = {
    "boba": [
        # generic terms
        "boba", "bubble tea", "milk tea", "pearl milk tea", "tapioca", "珍珠奶茶", "奶茶", "泡沫紅茶",
        # shop types
        "tea house", "tea shop", "茶館",
        # popular brands (helps recall in Google/Yelp/OSM)
        "Gong Cha", "Kung Fu Tea", "Chatime", "KOI Thé", "Tiger Sugar", "Machi Machi",
        "Yi Fang", "HEYTEA", "CoCo Fresh Tea", "Sharetea", "Happy Lemon", "Kungfutea",
        "The Alley", "Ten Ren Tea", "Vivi Bubble Tea", "Teazzi", "TBaar"
    ],
    "asian_tea": [
        "asian tea", "japanese tea", "korean tea", "taiwanese tea", "matcha", "sencha",
        "tea ceremony", "茶", "抹茶", "煎茶", "日式茶屋", "韓式茶屋", "台灣茶",
        # common brand names still useful here
        "Gong Cha", "Ten Ren Tea", "Muji Tea", "Ippodo", "Ito En"
    ],
    "asian_specialty": [
        "asian grocery", "asian market", "korean market", "japanese market", "chinese supermarket",
        "h-mart", "99 ranch", "mitsuwa", "japanese grocery", "korean grocery",
    ],
    "matcha": [
        "matcha", "matcha cafe", "抹茶", "matcha latte", "uchu matcha", "ceremonial matcha"
    ],
    "asian_skincare": [
        "k-beauty", "korean skincare", "asian skincare", "j-beauty", "cosrx", "innisfree", "sulwhasoo",
    ],
}

ABOUT_KEYWORDS = [
    "about", "our-story", "ourstory", "mission", "vision", "values", "who-we-are", "philosophy",
    "about-us", "story", "company", "team"
]
CONTACT_KEYWORDS = [
    "contact", "support", "help", "customer-service", "contact-us", "get-in-touch"
]
MISSION_KEYWORDS = [
    "mission", "vision", "values", "what we believe", "our purpose", "we believe", "our mission",
    "our values", "commitment", "purpose"
]
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

DEFAULT_HEADERS = {
    "User-Agent": "local-shops-mission-scraper/1.0 (+https://example.com)"
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters."""
    if None in (lat1, lon1, lat2, lon2):
        return float("inf")
    R = 6371000.0  # meters
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

_STOPWORDS = {
    "bubble", "boba", "tea", "milk", "shop", "house", "cafe", "co", "inc", "llc",
    "the", "and", "of"
}

def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return " ".join(toks)

def _domain(url: Optional[str]) -> str:
    try:
        if not url:
            return ""
        net = urlparse(url).netloc.lower()
        # strip leading www.
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return ""
    
    
COMMON_CONTACT_PATHS = [
    "/contact", "/contact-us", "/contactus", "/contacts",
    "/about", "/about-us", "/aboutus", "/support", "/help",
    "/customer-service", "/store-locator", "/locations",
    "/team", "/company", "/press", "/faq", "/policies", "/privacy", "/terms"
]

def candidate_contact_urls(root: str) -> List[str]:
    # root is like https://example.com
    out = []
    base = root.rstrip("/")
    for p in COMMON_CONTACT_PATHS:
        out.append(base + p)
        # also with trailing slash
        out.append(base + p + "/")
    # de-dupe
    seen = set(); dedup = []
    for u in out:
        if u not in seen:
            dedup.append(u); seen.add(u)
    return dedup



# ---------------------------
# Geocoding helpers (free via Nominatim)
# ---------------------------
async def geocode(session: aiohttp.ClientSession, location: str) -> Tuple[float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    async with session.get(url, params=params, headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]}) as resp:
        resp.raise_for_status()
        data = await resp.json()
        if not data:
            raise RuntimeError(f"Could not geocode '{location}'.")
        lat = float(data[0]["lat"])  # type: ignore
        lon = float(data[0]["lon"])  # type: ignore
        return lat, lon

# ---------------------------
# Providers: Google Places, Yelp Fusion, OSM Overpass
# ---------------------------
async def google_places_search(session, lat, lon, radius_m, terms: List[str]) -> List[Dict[str, Any]]:
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set.")

    bases = []
    for term in terms:
        # Text Search API
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": f"{term}",
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "key": key,
        }
        bases.append((url, params))

    results: Dict[str, Dict[str, Any]] = {}
    for url, params in bases:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logging.warning("Google TextSearch failed: %s", await resp.text())
                continue
            payload = await resp.json()
            for r in payload.get("results", []):
                place_id = r.get("place_id")
                if not place_id:
                    continue
                results[place_id] = r
            await asyncio.sleep(0.25)  # be friendly

    # Fetch website via Place Details
    enriched = []
    for place_id, r in results.items():
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {"place_id": place_id, "fields": "name,formatted_address,website,geometry", "key": key}
        async with session.get(details_url, params=params) as resp:
            if resp.status != 200:
                continue
            d = await resp.json()
            result = d.get("result", {})
            enriched.append({
                "name": result.get("name") or r.get("name"),
                "address": result.get("formatted_address") or r.get("formatted_address"),
                "lat": result.get("geometry", {}).get("location", {}).get("lat"),
                "lon": result.get("geometry", {}).get("location", {}).get("lng"),
                "website": result.get("website") or None,
                "source": "google",
            })
            await asyncio.sleep(0.25)
    return enriched

async def yelp_search(session, lat, lon, radius_m, terms: List[str]) -> List[Dict[str, Any]]:
    key = os.getenv("YELP_API_KEY")
    if not key:
        raise RuntimeError("YELP_API_KEY not set.")
    headers = {"Authorization": f"Bearer {key}", **DEFAULT_HEADERS}
    results = {}
    for term in terms:
        url = "https://api.yelp.com/v3/businesses/search"
        params = {"term": term, "latitude": lat, "longitude": lon, "radius": min(radius_m, 40000), "limit": 50}
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                logging.warning("Yelp search failed: %s", await resp.text())
                continue
            data = await resp.json()
            for b in data.get("businesses", []):
                results[b["id"]] = b
            await asyncio.sleep(0.25)
    enriched = []
    for bid, b in results.items():
        # Yelp only exposes Yelp page URL; try to infer official site from business details if present
        enriched.append({
            "name": b.get("name"),
            "address": ", ".join(filter(None, [b.get("location", {}).get("address1"), b.get("location", {}).get("city"), b.get("location", {}).get("state"), b.get("location", {}).get("zip_code")])),
            "lat": b.get("coordinates", {}).get("latitude"),
            "lon": b.get("coordinates", {}).get("longitude"),
            "website": b.get("url"),  # Yelp page if no official website given
            "source": "yelp",
        })
        await asyncio.sleep(0.1)
    return enriched

async def osm_overpass_search(session, lat, lon, radius_m, terms: List[str]) -> List[Dict[str, Any]]:
    # Best-effort: query relevant tags near location
    # We'll search for nodes/ways/relations with amenity=cafe AND name/description matching any term,
    # plus shop=beauty/cosmetics/supermarket with asian/korean/japanese keywords.
    q_terms = "|".join([re.escape(t) for t in terms])
    overpass = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node[amenity=cafe](around:{radius_m},{lat},{lon});
      way[amenity=cafe](around:{radius_m},{lat},{lon});
      node[shop~"cosmetics|beauty|supermarket|convenience|tea|beverages"](around:{radius_m},{lat},{lon});
      way[shop~"cosmetics|beauty|supermarket|convenience|tea|beverages"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    async with session.post(overpass, data={"data": query}, headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]}) as resp:
        resp.raise_for_status()
        data = await resp.json()
    enriched = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        # Filter by presence of category terms in name/description when provided
        blob = " ".join([tags.get("name", ""), tags.get("description", ""), tags.get("brand", "")]).lower()
        if q_terms and not any(t.lower() in blob for t in terms):
            continue
        lat_ = el.get("lat") or el.get("center", {}).get("lat")
        lon_ = el.get("lon") or el.get("center", {}).get("lon")
        website = tags.get("website") or tags.get("contact:website") or tags.get("url")
        addr = ", ".join(filter(None, [tags.get("addr:street"), tags.get("addr:city"), tags.get("addr:state"), tags.get("addr:postcode")]))
        enriched.append({
            "name": name,
            "address": addr or None,
            "lat": lat_,
            "lon": lon_,
            "website": website,
            "source": "osm",
        })
    return enriched

# ---------------------------
# Robots.txt checker
# ---------------------------
from urllib import robotparser
from urllib.parse import urlparse, urljoin

class RobotsCache:
    def __init__(self):
        self.cache: Dict[str, robotparser.RobotFileParser] = {}

    async def allowed(self, session: aiohttp.ClientSession, url: str, user_agent: str = DEFAULT_HEADERS["User-Agent"]) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base, "/robots.txt")
        if base not in self.cache:
            rp = robotparser.RobotFileParser()
            try:
                async with session.get(robots_url, headers=DEFAULT_HEADERS) as resp:
                    if resp.status == 200:
                        txt = await resp.text()
                        rp.parse(txt.splitlines())
                    else:
                        # If robots not accessible, default allow
                        rp.allow_all = True
            except Exception:
                rp.allow_all = True
            self.cache[base] = rp
        return self.cache[base].can_fetch(user_agent, url)

# ---------------------------
# Scraper core
# ---------------------------
@dataclass
class ScrapeResult:
    name: str
    address: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    website: Optional[str]
    source: str
    mission_excerpt: Optional[str]
    contact_email: Optional[str]
    crawled_urls: List[str]

async def fetch_text(session: aiohttp.ClientSession, url: str, *, timeout: int = 20) -> Optional[str]:
    try:
        async with session.get(url, headers=DEFAULT_HEADERS, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            # crude content-type check
            ct = resp.headers.get("Content-Type", "text/plain").lower()
            if "text/html" not in ct and "text" not in ct:
                return None
            return await resp.text(errors="ignore")
    except Exception:
        return None

def pick_candidate_links(base_url: str, soup: BeautifulSoup) -> List[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:"):
            links.append(href)
        else:
            full = urljoin(base_url, href)
            path = urlparse(full).path.lower()
            if any(k in path for k in ABOUT_KEYWORDS + CONTACT_KEYWORDS):
                links.append(full)
    # de-dup while preserving order
    seen = set(); dedup = []
    for u in links:
        if u not in seen:
            dedup.append(u); seen.add(u)
    return dedup[:12]

def extract_mission_text(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    # prioritize sections whose headings include mission-related keywords
    candidates: List[str] = []
    # gather paragraphs near keywords
    for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        low = txt.lower()
        if any(k in low for k in MISSION_KEYWORDS):
            candidates.append(txt)
    # fallback: longest paragraph text on about/mission pages
    if not candidates:
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        ps = [p for p in ps if len(p) > 60]
        if ps:
            candidates.append(max(ps, key=len))
    # reduce to ~700 chars
    if candidates:
        blob = max(candidates, key=len)
        return blob[:700]
    return None

def _cf_decrypt(hex_str: str) -> Optional[str]:
    # Cloudflare email protection: <a data-cfemail="...">
    try:
        data = bytes.fromhex(hex_str)
        key = data[0]
        return "".join(chr(b ^ key) for b in data[1:])
    except Exception:
        return None

_AT_DOT_PATTERNS = [
    (re.compile(r"\s*\[\s*at\s*\]\s*", re.I), "@"),
    (re.compile(r"\s*\(\s*at\s*\)\s*", re.I), "@"),
    (re.compile(r"\s+at\s+", re.I), "@"),
    (re.compile(r"\s*\[\s*dot\s*\]\s*", re.I), "."),
    (re.compile(r"\s*\(\s*dot\s*\)\s*", re.I), "."),
    (re.compile(r"\s+dot\s+", re.I), "."),
]

def _deobfuscate_text(txt: str) -> str:
    # HTML entities, then common obfuscations
    s = html.unescape(txt)
    for pat, repl in _AT_DOT_PATTERNS:
        s = pat.sub(repl, s)
    # strip extraneous spaces around @ and .
    s = re.sub(r"\s*@\s*", "@", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    return s

def extract_email(html_str: str) -> Optional[str]:
    soup = BeautifulSoup(html_str, "html.parser")

    # 1) Cloudflare-protected addresses
    for a in soup.select("[data-cfemail]"):
        dec = _cf_decrypt(a.get("data-cfemail"))
        if dec and EMAIL_REGEX.fullmatch(dec):
            return dec

    # 2) mailto links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().startswith("mailto:"):
            em = href.split(":",1)[1].split("?")[0].strip()
            em = _deobfuscate_text(em)
            if EMAIL_REGEX.fullmatch(em):
                return em

    # 3) JSON-LD contactPoint (schema.org)
    for script in soup.find_all("script", type=lambda t: t and "json" in t.lower()):
        try:
            data = json.loads(script.get_text(strip=True))
        except Exception:
            continue
        # normalize to list
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            # organization.contactPoint[].email
            cp = obj.get("contactPoint") if isinstance(obj, dict) else None
            if isinstance(cp, dict): cp = [cp]
            if isinstance(cp, list):
                for c in cp:
                    em = c.get("email")
                    if isinstance(em, str):
                        em = _deobfuscate_text(em)
                        if EMAIL_REGEX.fullmatch(em):
                            return em
            # fallback: obj.get("email")
            em = obj.get("email") if isinstance(obj, dict) else None
            if isinstance(em, str):
                em = _deobfuscate_text(em)
                if EMAIL_REGEX.fullmatch(em):
                    return em

    # 4) Page text (deobfuscate first)
    text = soup.get_text(" ", strip=True)
    text = _deobfuscate_text(text)
    emails = set(EMAIL_REGEX.findall(text))
    if emails:
        # prefer generic inboxes
        def score(e: str) -> int:
            e_low = e.lower()
            s = 0
            if any(k in e_low for k in ["info@", "hello@", "contact@", "support@", "hi@", "team@", "care@", "service@"]):
                s += 3
            if any(k in e_low for k in ["gmail.com", "yahoo.com", "outlook.com"]):
                s -= 1
            return s
        return sorted(emails, key=score, reverse=True)[0]

    return None


async def scrape_site(session: aiohttp.ClientSession, item: Dict[str, Any], robots: RobotsCache) -> ScrapeResult:
    name = item.get("name")
    website = item.get("website")
    crawled = []
    mission = None
    email = None

    if website and website.startswith("http"):
        # Normalize to root
        parsed = urlparse(website)
        root = f"{parsed.scheme}://{parsed.netloc}"
        start_urls = [website, root]
        start_urls.extend(candidate_contact_urls(root))
        visited = set()
        for u in start_urls:
            if u in visited:
                continue
            visited.add(u)
            if not await robots.allowed(session, u):
                continue
            html = await fetch_text(session, u)
            
            # favor footer blocks
            for footer in soup.find_all(["footer"]):
                em = extract_email(str(footer))
                if em: email = email or em
            # and any script tags (some sites embed plain emails in config)
            for script in soup.find_all("script"):
                em = extract_email(script.get_text(" ", strip=True))
                if em: email = email or em

            if not html:
                continue
            crawled.append(u)
            email = email or extract_email(html)
            if any(k in urlparse(u).path.lower() for k in ABOUT_KEYWORDS + CONTACT_KEYWORDS):
                mission = mission or extract_mission_text(html)
            # Discover more candidate links
            soup = BeautifulSoup(html, "html.parser")
            for link in pick_candidate_links(u, soup):
                if link in visited:
                    continue
                visited.add(link)
                if not await robots.allowed(session, link):
                    continue
                html2 = await fetch_text(session, link)
                if not html2:
                    continue
                crawled.append(link)
                email = email or extract_email(html2)
                mission = mission or extract_mission_text(html2)
                if email and mission:
                    break
            if email and mission:
                break

    return ScrapeResult(
        name=name or "",
        address=item.get("address"),
        lat=item.get("lat"),
        lon=item.get("lon"),
        website=website,
        source=item.get("source", ""),
        mission_excerpt=mission,
        contact_email=email,
        crawled_urls=crawled,
    )

# ---------------------------
# Search orchestration
# ---------------------------
async def search_shops(provider: str, location: str, radius_m: int, categories: List[str]) -> List[Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        lat, lon = await geocode(session, location)
        logging.info("Geocoded '%s' -> %.6f, %.6f", location, lat, lon)
        # build term list
        terms: List[str] = []
        for cat in categories:
            terms.extend(CATEGORIES_CANON.get(cat, [cat]))
        # de-dup, preserve order
        seen = set(); terms = [t for t in terms if not (t in seen or seen.add(t))]

        if provider == "google":
            found = await google_places_search(session, lat, lon, radius_m, terms)
        elif provider == "yelp":
            found = await yelp_search(session, lat, lon, radius_m, terms)
        elif provider == "osm":
            found = await osm_overpass_search(session, lat, lon, radius_m, terms)
        else:
            raise ValueError("provider must be one of: google, yelp, osm")

       # --- distance-aware de-duplication (prefer closer to center) ---
        center_lat, center_lon = lat, lon

        # annotate with distance to center
        for f in found:
            f["__dist_m"] = _haversine_m(center_lat, center_lon, f.get("lat"), f.get("lon"))
            f["__name_norm"] = _normalize_name(f.get("name"))
            f["__addr_norm"] = (f.get("address") or "").strip().lower()
            f["__domain"] = _domain(f.get("website"))

        # sort so the NEAREST entries come first; ties put items with real coords ahead of missing
        found.sort(key=lambda x: (x["__dist_m"], 0 if x.get("lat") is not None else 1))

        kept: List[Dict[str, Any]] = []
        seen_keys = set()  # coarse signatures
        PROX_THRESHOLD_M = 120.0  # treat as same place if very close and same-ish name

        kept: List[Dict[str, Any]] = []
        seen_keys = set()  # store 3-tuples consistently
        PROX_THRESHOLD_M = 120.0

        for f in found:
            name_norm = f["__name_norm"]
            addr_norm = f["__addr_norm"]
            domain = f["__domain"]

            # build consistent 3-tuples; skip empties later
            sigs = set()
            if name_norm or addr_norm:
                sigs.add(("name+addr", name_norm, addr_norm))
            if name_norm or domain:
                sigs.add(("name+domain", name_norm, domain))
            if domain:
                sigs.add(("domain", domain, ""))  # keep 3-tuple shape

            # if any signature already taken, skip this farther one
            if any(s in seen_keys for s in sigs):
                continue

            # proximity check: same normalized name and very close → treat as dup
            is_dup_by_proximity = False
            if name_norm:
                for g in kept:
                    if name_norm == g.get("__name_norm"):
                        d = _haversine_m(f.get("lat"), f.get("lon"), g.get("lat"), g.get("lon"))
                        if d <= PROX_THRESHOLD_M:
                            is_dup_by_proximity = True
                            break
            if is_dup_by_proximity:
                continue

            # keep this one (we sorted nearest-first), mark signatures as seen
            kept.append(f)
            for s in sigs:
                seen_keys.add(s)

        # clean temp fields
        for f in kept:
            for k in ("__dist_m", "__name_norm", "__addr_norm", "__domain"):
                f.pop(k, None)

        logging.info("Found %d candidate shops after de-dup.", len(kept))
        return kept


# ---------------------------
# Main
# ---------------------------
async def main():
    parser = argparse.ArgumentParser(description="Find nearby shops and scrape mission/contact.")
    parser.add_argument("--location", required=True, help="e.g., 'New York, NY' or '40.8075,-73.9626'")
    parser.add_argument("--radius-m", type=int, default=5000, help="Search radius in meters")
    parser.add_argument("--provider", choices=["google", "yelp", "osm"], default="osm")
    parser.add_argument("--categories", default="boba,asian_specialty,matcha,asian_skincare",
                        help="Comma-separated categories (keys in CATEGORIES_CANON or free text)")
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    # Normalize categories
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    shops = await search_shops(args.provider, args.location, args.radius_m, cats)

    os.makedirs(args.out, exist_ok=True)

    robots = RobotsCache()
    sem = asyncio.Semaphore(args.max_concurrency)

    results: List[ScrapeResult] = []

    async def worker(item):
        async with sem:
            async with aiohttp.ClientSession() as session:
                try:
                    res = await scrape_site(session, item, robots)
                except Exception as e:
                    logging.warning("scrape failed for %s: %s", item.get("website"), e)
                    res = ScrapeResult(
                        name=item.get("name") or "",
                        address=item.get("address"), lat=item.get("lat"), lon=item.get("lon"),
                        website=item.get("website"), source=item.get("source", ""),
                        mission_excerpt=None, contact_email=None, crawled_urls=[])
                results.append(res)

    await asyncio.gather(*(worker(s) for s in shops))

    # Write JSON
    json_path = os.path.join(args.out, "shops.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # Write CSV
    csv_path = os.path.join(args.out, "shops.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name","address","lat","lon","website","source","contact_email","mission_excerpt","crawled_urls"])
        for r in results:
            w.writerow([
                r.name, r.address or "", r.lat or "", r.lon or "", r.website or "", r.source,
                r.contact_email or "", (r.mission_excerpt or "").replace("\n"," ")[:1000], " ".join(r.crawled_urls)
            ])

    logging.info("Saved %d rows to %s and %s", len(results), json_path, csv_path)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass