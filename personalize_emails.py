#!/usr/bin/env python3
"""
personalize_emails.py

Reads shops.csv/shops.json produced by your scraper and generates personalized
emails using an LLM (OpenAI-compatible Chat Completions).

Key features:
- Supports multiple <dynamic>...</dynamic> sections in the template; each is rewritten by the LLM.
- Preserves immutable placeholders like {{event_date}}, {{event_time}}, {{event_place}} exactly.
- Fills {{shop_name}} and {{team_or_inquiries}} (LLM-suggested) at render time.
- Computes distance from event location to shop and generates a proximity phrase.
- Produces 'mail_merge.csv' (to_email,subject,body,shop_name,website).

Environment variables (required for LLM):
  LLM_API_URL   (e.g., https://api.openai.com/v1/chat/completions)
  LLM_API_KEY
  LLM_MODEL     (e.g., gpt-4o-mini)

Usage example:
  python personalize_emails.py \
    --csv results_nyc/shops.csv \
    --template sponsorships_template.txt \
    --event-location "Columbia University, New York, NY" \
    --event-date "10/25" \
    --event-time "7:00 PM" \
    --event-place "Lerner Hall" \
    --out out_emails
"""
from __future__ import annotations

import os, csv, json, math, argparse, asyncio, logging, re
from typing import List, Dict, Any, Tuple, Optional
import aiohttp
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
load_dotenv()

# ------------------- Geo helpers -------------------
async def geocode(session: aiohttp.ClientSession, q: str) -> Tuple[float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    async with session.get(url, params=params, headers={"User-Agent": "email-personalizer/1.0"}) as r:
        r.raise_for_status()
        data = await r.json()
        if not data:
            raise RuntimeError(f"Could not geocode '{q}'")
        return float(data[0]["lat"]), float(data[0]["lon"])

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2): return float("inf")
    R=6371000.0
    p1=math.radians(lat1); p2=math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlmb=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def proximity_phrase(distance_m: float) -> str:
    if distance_m == float("inf"): return "in New York City"
    if distance_m < 250: return "right by Columbia’s campus"
    if distance_m < 800: return "a short walk from Columbia"
    if distance_m < 2000: return "within a quick walk from Columbia"
    if distance_m < 5000: return "a short subway ride from Columbia"
    return "in New York City"

# ------------------- IO helpers -------------------
def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        for i,row in enumerate(csv.DictReader(f)):
            row["_rownum"]=i+1
            rows.append(row)
    return rows

def read_json(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path): return {}
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def load_template(path: str) -> str:
    with open(path, encoding="utf-8") as f: return f.read()

# ------------------- Template helpers -------------------
def extract_subject_and_body(template_text: str) -> Tuple[str, str]:
    # First line "Subject: ..." optional; else synthesize a subject
    lines = template_text.splitlines()
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":",1)[1].strip()
        body = "\n".join(lines[1:]).lstrip("\n")
    else:
        subject = "Sponsorship for Columbia Wushu event on {{event_date}}"
        body = template_text
    return subject, body

def fill_placeholders(text: str, vars: Dict[str,str]) -> str:
    # simple {{key}} substitution only (no logic)
    for k,v in vars.items():
        text = text.replace("{{"+k+"}}", v)
    return text

# Match multiple <dynamic> blocks (non-greedy)
DYN_RE = re.compile(r"<dynamic>(.*?)</dynamic>", re.DOTALL|re.IGNORECASE)

def split_dynamic_blocks(body: str) -> List[Tuple[str, Optional[str]]]:
    """
    Returns a list of (literal_text, dynamic_text|None) alternating segments.
    For example: [("Hello ", None), ("", "dyn1"), (" world ", None), ("", "dyn2"), ("!", None)]
    """
    parts: List[Tuple[str, Optional[str]]] = []
    last_end = 0
    for m in DYN_RE.finditer(body):
        parts.append((body[last_end:m.start()], None))
        parts.append(("", m.group(1).strip()))
        last_end = m.end()
    parts.append((body[last_end:], None))
    return parts

def reassemble_from_parts(parts: List[Tuple[str, Optional[str]]], dyn_rewrites: List[str]) -> str:
    out = []
    di = 0
    for literal, dyn in parts:
        if dyn is None:
            out.append(literal)
        else:
            out.append(dyn_rewrites[di])
            di += 1
    return "".join(out)

# ------------------- LLM -------------------
async def call_llm(session: aiohttp.ClientSession, system: str, user: str, *, temperature: float=0.4) -> str:
    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model   = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_url or not api_key:
        raise RuntimeError("Set LLM_API_URL and LLM_API_KEY (and optional LLM_MODEL).")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role":"system","content": system},
            {"role":"user","content": user}
        ]
    }
    async with session.post(api_url, headers=headers, json=payload) as r:
        txt = await r.text()
        if r.status != 200:
            raise RuntimeError(f"LLM HTTP {r.status}: {txt[:300]}")
        data = json.loads(txt)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            raise RuntimeError(f"Unexpected LLM response: {txt[:500]}")

def build_dyn_prompt(dynamic_text: str, facts: Dict[str, Any]) -> Tuple[str,str]:
    system = (
        "You are a precise email copywriter. Rewrite ONLY the provided dynamic paragraph(s) "
        "to tailor for the given shop and event context. Keep tone professional and warm. "
        "Do NOT change any content outside the <dynamic> blocks. Do NOT invent facts. "
        "Keep length similar to the original dynamic text (1–2 sentences)."
    )
    user = (
        "Shop facts (use only if relevant):\n"
        f"- shop_name: {facts.get('shop_name')}\n"
        f"- address: {facts.get('address')}\n"
        f"- website: {facts.get('website')}\n"
        f"- distance_to_event_m: {facts.get('distance_m')}\n"
        f"- proximity_phrase: {facts.get('proximity_phrase')}\n"
        f"- mission_excerpt: {facts.get('mission_excerpt')}\n"
        f"- product_type_guess: {facts.get('product_type')}\n"
        "\nRewrite the dynamic block below. Output ONLY the rewritten text; no greetings:\n\n"
        f"{dynamic_text}"
    )
    return system, user

def build_header_prompt(facts: Dict[str, Any]) -> Tuple[str,str]:
    # Ask LLM for a polite team/inquiries label (short)
    system = "You craft concise addressee labels for business emails."
    user = (
        "Given the following shop, propose a short addressee label suitable for 'Dear {shop} <label>,'\n"
        "Examples: 'General Inquiries Team', 'Marketing Team', 'Sponsorships', 'Community Team'.\n"
        "Return only the label (no punctuation).\n\n"
        f"Shop name: {facts.get('shop_name')}\n"
        f"Website: {facts.get('website')}"
    )
    return system, user

def guess_product(row: Dict[str,Any]) -> str:
    name = (row.get("name") or "").lower()
    website = (row.get("website") or "").lower()
    src = (row.get("source") or "").lower()
    blob = " ".join([name, website, src, (row.get("mission_excerpt") or "").lower()])
    if any(k in blob for k in ["matcha","tea","bubble","boba","milk tea"]): return "tea/boba drinks"
    if any(k in blob for k in ["skincare","beauty","cosmetic","k-beauty","j-beauty"]): return "skincare/beauty"
    if any(k in blob for k in ["asian market","grocery","supermarket","h-mart","mitsuwa","99 ranch"]): return "asian grocery/specialty"
    return "local business"

# ------------------- Main pipeline -------------------
async def personalize(args):
    os.makedirs(args.out, exist_ok=True)
    template_text = load_template(args.template)
    subject_tpl, body_tpl = extract_subject_and_body(template_text)

    # read CSV
    rows = read_csv_rows(args.csv)

    async with aiohttp.ClientSession() as session:
        evt_lat, evt_lon = await geocode(session, args.event_location)

        out_rows = []

        for row in rows:
            email = (row.get("contact_email") or "").strip()
            if not email:
                continue

            # Coords if present
            lat = row.get("lat"); lon = row.get("lon")
            try:
                lat = float(lat) if lat not in (None,"") else None
                lon = float(lon) if lon not in (None,"") else None
            except Exception:
                lat = lon = None

            distance_m = haversine_m(evt_lat, evt_lon, lat, lon)
            prox = proximity_phrase(distance_m)

            facts = {
                "shop_name": row.get("name") or "",
                "address": row.get("address") or "",
                "website": row.get("website") or "",
                "mission_excerpt": (row.get("mission_excerpt") or "")[:450],
                "distance_m": int(distance_m) if distance_m != float("inf") else None,
                "proximity_phrase": prox,
                "product_type": guess_product(row)
            }

            # Ask LLM for a short addressee label (team_or_inquiries)
            try:
                sys_h, usr_h = build_header_prompt(facts)
                team_or_inquiries = await call_llm(session, sys_h, usr_h, temperature=0.2)
                team_or_inquiries = team_or_inquiries.replace(",", "").strip()
                if len(team_or_inquiries) > 48:  # keep it short
                    team_or_inquiries = "Team"
            except Exception as e:
                logging.warning("Header LLM failed: %s", e)
                team_or_inquiries = "Team"

            # Fill immutable placeholders NOW
            subject = fill_placeholders(subject_tpl, {
                "event_date": args.event_date,
                "event_time": args.event_time,
                "event_place": args.event_place,
                "shop_name": facts["shop_name"]
            })

            body_base = fill_placeholders(body_tpl, {
                "event_date": args.event_date,
                "event_time": args.event_time,
                "event_place": args.event_place,
                "shop_name": facts["shop_name"],
                "team_or_inquiries": team_or_inquiries,
            })

            # Process multiple <dynamic> blocks
            parts = split_dynamic_blocks(body_base)
            rewrites: List[str] = []
            for literal, dyn in parts:
                if dyn is None:
                    continue
                try:
                    sys_d, usr_d = build_dyn_prompt(dyn, facts)
                    rewritten = await call_llm(session, sys_d, usr_d, temperature=0.4)
                    # very small guard: keep it <= 3 sentences
                    sentences = re.split(r'(?<=[.!?])\s+', rewritten.strip())
                    if len(sentences) > 3:
                        rewritten = " ".join(sentences[:3])
                except Exception as e:
                    logging.warning("Dynamic LLM failed on %s: %s", facts["shop_name"], e)
                    rewritten = dyn
                rewrites.append(rewritten)

            if rewrites:
                body = reassemble_from_parts(parts, rewrites)
            else:
                body = body_base

            out_rows.append({
                "to_email": email,
                "subject": subject,
                "body": body,
                "shop_name": facts["shop_name"],
                "website": facts["website"]
            })

    out_csv = os.path.join(args.out, "mail_merge.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["to_email","subject","body","shop_name","website"])
        w.writeheader()
        w.writerows(out_rows)

    out_json = os.path.join(args.out, "mail_merge.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    logging.info("Wrote %d personalized emails → %s", len(out_rows), out_csv)

def parse_args():
    p = argparse.ArgumentParser(description="Personalize sponsorship emails with an LLM.")
    p.add_argument("--csv", required=True, help="Path to shops.csv")
    p.add_argument("--json", help="Optional shops.json (unused but accepted)")
    p.add_argument("--template", required=True, help="Path to email template text file")
    p.add_argument("--event-location", required=True, help="Geocodable address (for distance)")
    p.add_argument("--event-date", required=True)
    p.add_argument("--event-time", required=True)
    p.add_argument("--event-place", required=True)
    p.add_argument("--llm-api-url")
    p.add_argument("--llm-api-key")
    p.add_argument("--llm-model")
    p.add_argument("--out", default="out_emails")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(personalize(args))
