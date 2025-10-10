#!/usr/bin/env python3
"""
personalize_emails.py (Gemini version)

Reads shops.csv and generates personalized sponsorship emails using
Google’s Gemini API (Generative Language API).

Environment variables (required for LLM):
  GEMINI_API_KEY   (or GOOGLE_API_KEY)
  LLM_MODEL        (e.g., gemini-1.5-flash)

Usage example:
  python personalize_emails.py \
    --csv results_nyc/shops.csv \
    --template sponsorship_template.txt \
    --event-location "Columbia University, New York, NY" \
    --event-date "10/25" \
    --event-time "7:00 PM" \
    --event-place "Lerner Hall" \
    --out personalized_emails
"""

import os, csv, json, math, argparse, asyncio, logging, re, aiohttp
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
load_dotenv()
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ------------------------------------------------------------
# GEO HELPERS
# ------------------------------------------------------------
async def geocode(session: aiohttp.ClientSession, q: str) -> Tuple[float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    async with session.get(url, params=params, headers={"User-Agent": "email-personalizer/1.0"}) as r:
        r.raise_for_status()
        data = await r.json()
        if not data:
            raise RuntimeError(f"Could not geocode '{q}'")
        return float(data[0]["lat"]), float(data[0]["lon"])


def haversine_m(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return float("inf")
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def proximity_phrase(distance_m: float) -> str:
    if distance_m == float("inf"):
        return "in New York City"
    if distance_m < 250:
        return "right by Columbia’s campus"
    if distance_m < 800:
        return "a short walk from Columbia"
    if distance_m < 2000:
        return "within a quick walk from Columbia"
    if distance_m < 5000:
        return "a short subway ride from Columbia"
    return "in New York City"


# ------------------------------------------------------------
# FILE HELPERS
# ------------------------------------------------------------
def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def fill_placeholders(text: str, vars: Dict[str, str]) -> str:
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", v)
    return text


# ------------------------------------------------------------
# GEMINI API CALL
# ------------------------------------------------------------
async def call_llm(session: aiohttp.ClientSession, system: str, user: str, *, temperature: float = 0.4) -> str:
    """
    Gemini (Generative Language API) async call using REST.
    Endpoint:
      https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=API_KEY
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"role": "user", "parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature},
    }

    async with session.post(url, headers=headers, json=payload) as r:
        txt = await r.text()
        if r.status != 200:
            raise RuntimeError(f"Gemini HTTP {r.status}: {txt[:300]}")
        data = json.loads(txt)
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates: {txt[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_out = " ".join([p.get("text", "") for p in parts]).strip()
        return text_out or "(No output from Gemini)"


# ------------------------------------------------------------
# PROMPT BUILDERS
# ------------------------------------------------------------
def build_dyn_prompt(dynamic_text: str, facts: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You are a precise email copywriter. Rewrite ONLY the provided dynamic paragraph(s) "
        "to tailor for the given shop and event context. Keep tone professional and warm. "
        "Do NOT change any content outside the <dynamic> blocks. Do NOT invent facts. "
        "Keep length similar to the original dynamic text (1–2 sentences)."
    )
    user = (
        f"Shop facts:\n"
        f"- shop_name: {facts.get('shop_name')}\n"
        f"- address: {facts.get('address')}\n"
        f"- website: {facts.get('website')}\n"
        f"- proximity_phrase: {facts.get('proximity_phrase')}\n"
        f"- mission_excerpt: {facts.get('mission_excerpt')}\n"
        f"- product_type_guess: {facts.get('product_type')}\n\n"
        f"Rewrite the following dynamic section:\n{dynamic_text}"
    )
    return system, user


def build_header_prompt(facts: Dict[str, Any]) -> Tuple[str, str]:
    system = "You craft concise addressee labels for business emails."
    user = (
        "Given this shop, propose a short addressee label suitable for 'Dear {shop} <label>,' "
        "(e.g., 'General Inquiries Team', 'Marketing Team', 'Sponsorships'). "
        "Return only the label text.\n\n"
        f"Shop name: {facts.get('shop_name')}\nWebsite: {facts.get('website')}"
    )
    return system, user


def guess_product(row: Dict[str, Any]) -> str:
    blob = " ".join(
        (row.get("name", ""), row.get("website", ""), row.get("source", ""), row.get("mission_excerpt", ""))
    ).lower()
    if any(k in blob for k in ["matcha", "tea", "bubble", "boba", "milk tea"]):
        return "tea/boba drinks"
    if any(k in blob for k in ["skincare", "beauty", "cosmetic", "k-beauty", "j-beauty"]):
        return "skincare/beauty"
    if any(k in blob for k in ["asian market", "grocery", "supermarket", "h-mart", "mitsuwa", "99 ranch"]):
        return "asian grocery/specialty"
    return "local business"


# ------------------------------------------------------------
# MAIN PERSONALIZATION LOGIC
# ------------------------------------------------------------
async def personalize(args):
    os.makedirs(args.out, exist_ok=True)
    template_text = load_template(args.template)

    async with aiohttp.ClientSession() as session:
        evt_lat, evt_lon = await geocode(session, args.event_location)
        rows = read_csv_rows(args.csv)
        out_rows = []

        rows = read_csv_rows(args.csv)
        total = len(rows)
        
        out_rows = []
        start_time = time.time()
        success = 0
        skipped_no_email = 0

        print(f"[INFO] Starting personalization with Gemini (model: {os.getenv('LLM_MODEL','gemini-1.5-flash')})")
        print(f"[INFO] Event: {args.event_location} on {args.event_date} at {args.event_time} ({args.event_place})")
        print(f"[INFO] Total rows in CSV: {total}")

        for i, row in enumerate(rows):
            # identify row
            shop_name = row.get("name") or row.get("shop_name") or "Unknown Shop"
            to_email = (row.get("contact_email") or row.get("to_email") or "").strip()

            # progress header
            print(f"\n[{i+1}/{total}] 🔄 Processing: {shop_name}  |  email={to_email or '—'}")

            # skip if no email
            if not to_email:
                print(f"[{i+1}/{total}] ⚠️ Skipped (no email found).")
                skipped_no_email += 1
                continue

            # geo + proximity
            try:
                lat = float(row.get("lat", "") or "nan")
                lon = float(row.get("lon", "") or "nan")
            except ValueError:
                lat = lon = None

            try:
                # distance & phrasing
                distance_m = haversine_m(evt_lat, evt_lon, lat, lon)
                prox = proximity_phrase(distance_m)

                # facts used in prompts
                facts = {
                    "shop_name": shop_name,
                    "address": row.get("address", ""),
                    "website": row.get("website", ""),
                    "mission_excerpt": (row.get("mission_excerpt", "") or "")[:300],
                    "proximity_phrase": prox,
                    "product_type": guess_product(row),
                }

                # header label (team/inquiries)
                try:
                    sys_h, usr_h = build_header_prompt(facts)
                    team_or_inquiries = await call_llm(session, sys_h, usr_h, temperature=0.2)
                    team_or_inquiries = (team_or_inquiries or "Team").strip()
                    # collapse to a short label (no punctuation/newlines)
                    team_or_inquiries = team_or_inquiries.replace("\n", " ").strip(" ,.;:-")
                    print(f"[{i+1}/{total}] 🧭 Addressee resolved → {team_or_inquiries}")
                except Exception as e:
                    team_or_inquiries = "Team"
                    print(f"[{i+1}/{total}] ⚠️ Header LLM fallback → Team  (reason: {e})")

                # fill immutable placeholders first
                body_filled = fill_placeholders(
                    template_text,
                    {
                        "shop_name": facts["shop_name"],
                        "event_date": args.event_date,
                        "event_time": args.event_time,
                        "event_place": args.event_place,
                        "team_or_inquiries": team_or_inquiries,
                    },
                )

                # personalize each <dynamic>…</dynamic> block
                dyn_blocks = re.findall(r"<dynamic>(.*?)</dynamic>", body_filled, re.DOTALL)
                if dyn_blocks:
                    print(f"[{i+1}/{total}] ✍️  Dynamic blocks: {len(dyn_blocks)}")
                for idx, block in enumerate(dyn_blocks, 1):
                    sys_d, usr_d = build_dyn_prompt(block.strip(), facts)
                    try:
                        new_text = await call_llm(session, sys_d, usr_d)
                        body_filled = body_filled.replace(block, new_text)
                        preview = (new_text or "").strip().replace("\n", " ")
                        print(f"[{i+1}/{total}]    ✓ Block {idx} → {preview[:90]}{'…' if len(preview)>90 else ''}")
                    except Exception as e:
                        print(f"[{i+1}/{total}]    ⚠️ Block {idx} failed, leaving original. (reason: {e})")

                # subject + output row
                subject = f"{facts['shop_name']} Support & Sponsorship for a Columbia student-run event"
                out_rows.append(
                    {
                        "to_email": to_email,
                        "subject": subject,
                        "body": body_filled,
                        "shop_name": facts["shop_name"],
                        "website": facts["website"],
                    }
                )
                success += 1

                # ETA math
                elapsed = time.time() - start_time
                avg_per = elapsed / (i + 1)
                remaining = (total - (i + 1)) * avg_per
                print(f"[{i+1}/{total}] ✅ Done. Elapsed: {elapsed:.1f}s | Avg/email: {avg_per:.2f}s | ETA: {remaining:.1f}s")

            except Exception as e:
                print(f"[{i+1}/{total}] ❌ Error: {e}")

        # write outputs
        csv_path = os.path.join(args.out, "mail_merge.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["to_email", "subject", "body", "shop_name", "website"])
            w.writeheader()
            w.writerows(out_rows)

        elapsed_all = time.time() - start_time
        print("\n[SUMMARY]")
        print(f"  Total rows          : {total}")
        print(f"  Personalized emails : {success}")
        print(f"  Skipped (no email)  : {skipped_no_email}")
        print(f"  Output CSV          : {csv_path}")
        print(f"  Total time          : {elapsed_all:.1f}s  (avg/email: {elapsed_all/max(1,success):.2f}s)")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--event-location", required=True)
    p.add_argument("--event-date", required=True)
    p.add_argument("--event-time", required=True)
    p.add_argument("--event-place", required=True)
    p.add_argument("--out", default="out_emails")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(personalize(parse_args()))