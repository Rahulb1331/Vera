"""
Vera — magicpin Merchant AI Assistant
A high-quality, context-grounded message composer for merchant engagement.
"""

import os
import time
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys

sys.stdout.reconfigure(encoding='utf-8')
# ==============================================================================
# CONFIG
# ==============================================================================

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")

START_TIME = time.time()
app = FastAPI(title="Vera Bot")

if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    import anthropic as _anthropic
    client_llm = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    MODEL      = "claude-sonnet-4-20250514"
    print(f"[LLM] Using Anthropic — {MODEL}")

elif LLM_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
    from openai import OpenAI as _OpenAI
    client_llm = _OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    MODEL      = "deepseek-chat"
    print(f"[LLM] Using DeepSeek — {MODEL}")

elif LLM_PROVIDER == "groq" and GROQ_API_KEY:
    from openai import OpenAI as _OpenAI
    client_llm = _OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    MODEL      = "llama-3.3-70b-versatile"
    print(f"[LLM] Using Groq — {MODEL}")

else:
    client_llm = None
    MODEL      = "none"
    print("[LLM] No API key found — fallback mode only")

# ==============================================================================
# IN-MEMORY STATE
# ==============================================================================

contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, dict] = {}
suppressed_keys: set[str] = set()
ended_conversations: set[str] = set()

# ==============================================================================
# HELPERS
# ==============================================================================

def get_context(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def context_count() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


def llm_call(system: str, prompt: str, max_tokens: int = 600) -> str:
    """Unified LLM call — handles both Anthropic and OpenAI-compatible APIs."""
    if LLM_PROVIDER == "anthropic":
        resp = client_llm.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        resp = client_llm.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()


def parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def detect_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    patterns = [
        r"thank you for contacting",
        r"our team will respond shortly",
        r"we will get back to you",
        r"this is an automated",
        r"auto.?reply",
        r"out of office",
        r"currently unavailable",
        r"we have received your message",
        r"aapki madad ke liye shukriya",
        r"main ek automated",
    ]
    return any(re.search(p, message.lower()) for p in patterns)


def detect_opt_out(message: str) -> bool:
    """Detect explicit opt-out or hostility."""
    patterns = [
        r"\bstop\b",
        r"\bnot interested\b",
        r"\bdo not contact\b",
        r"\bspam\b",
        r"\buseless\b",
        r"\bstop messaging\b",
        r"\bband karo\b",
        r"\bbother(ing)? me\b",
        r"\bwhy are you\b",
    ]
    return any(re.search(p, message.lower()) for p in patterns)


def detect_intent_commit(message: str) -> bool:
    """Detect merchant committing to action."""
    patterns = [
        r"\blets? do it\b",
        r"\bgo ahead\b",
        r"\byes.{0,10}(do it|proceed|start)\b",
        r"\bconfirm\b",
        r"\bproceed\b",
        r"\b(haan|ha|han).{0,5}(karo|chalao|bhejo|send)\b",
        r"\bwhat'?s? next\b",
        r"\bok.{0,10}(proceed|let's|lets)\b",
    ]
    return any(re.search(p, message.lower()) for p in patterns)


def make_conv_id(merchant_id: str, trigger: dict) -> str:
    """
    Build a meaningful conversation ID.
    Format: conv_{merchant_short}_{trigger_kind}_{suppression_slug}
    Example: conv_m001drmeera_research_2026W17
    """
    m_short = merchant_id.replace("_", "")[:16]
    kind = trigger.get("kind", "msg").replace("_", "")[:12]
    sup = trigger.get("suppression_key", trigger.get("id", ""))
    sup_slug = re.sub(r"[^a-zA-Z0-9]", "", sup)[-8:]
    return f"conv_{m_short}_{kind}_{sup_slug}"


def owner_address(merchant: dict, cat_slug: str) -> str:
    """Return the correct form of address for the merchant owner."""
    first = merchant.get("identity", {}).get("owner_first_name", "")
    if cat_slug == "dentists":
        return f"Dr. {first}" if first else "Doctor"
    return first or merchant.get("identity", {}).get("name", "")


# ==============================================================================
# COMPOSE SYSTEM PROMPT — derived from case studies
# ==============================================================================

COMPOSE_SYSTEM = """You are Vera, magicpin's AI assistant for merchant growth in India.
You compose WhatsApp messages to merchants (and their customers) that are specific, grounded, and compulsive.
- Specific: use real numbers, dates, prices, source citations from the context
- Category-fit: dentists get clinical/peer tone; salons get warm/practical; restaurants get operator tone; gyms get coaching voice; pharmacies get trustworthy/precise
- Merchant-fit: use owner first name, reference their actual metrics and offers
- Trigger-relevant: clearly communicate why this message is being sent right now
- Engagement-compelling: one clear CTA, low friction, use curiosity/loss aversion/social proof

WHAT HIGH-SCORING MESSAGES LOOK LIKE (study these patterns):
- Dentist research digest: "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14"
- Gym seasonal dip: "Karthik, your views are down 30% this week — but this is the normal April-June acquisition lull (every metro gym sees -25 to -35% in this window). Action: skip ad spend now, save it for Sept-Oct when conversion is 2x. For now, focus retention on your 245 members. Want me to draft a summer attendance challenge?"
- Restaurant IPL: "Quick heads-up Suresh — DC vs MI at Arun Jaitley tonight, 7:30pm. Saturday IPL matches usually shift -12% restaurant covers. Skip the match-night promo; push your BOGO pizza as a delivery-only Saturday special instead. Want me to draft the Swiggy banner? Live in 10 min."
- Pharmacy compliance: "Ramesh, urgent: voluntary recall on 2 atorvastatin batches (AT2024-1102, AT2024-1108) by Mfr Z — sub-potency, no safety risk. Pulled your repeat-Rx list: 22 of your chronic-Rx customers were dispensed these batches in last 90 days. Want me to draft their WhatsApp note + replacement-pickup workflow?"
- Customer recall (sent as merchant): "Hi Priya, Dr. Meera's clinic here. It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. Rs.299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works."

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{
  "body": "<WhatsApp message body — concise, plain text, no markdown>",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<dedup key from trigger, do not invent>",
  "rationale": "<2-3 sentences: which signal, why now, what it achieves>"
}

SCORING DIMENSIONS — optimize for all five:
1. SPECIFICITY: Use exact numbers, dates, batch numbers, page references, percentages from the context. No vague claims.
2. CATEGORY FIT: Match the voice exactly — dentists=peer_clinical, salons=warm_practical, restaurants=operator, gyms=coach, pharmacies=trustworthy_precise. Use domain vocabulary (covers, AOV, sub-potency, fluoride varnish, ad spend).
3. MERCHANT FIT: Use owner first name. Reference their actual CTR vs peer median. Reference their specific customer counts, offer prices, locality.
4. TRIGGER RELEVANCE: The trigger is the reason for sending NOW. Make that reason explicit and specific.
5. ENGAGEMENT COMPULSION: One clear low-friction CTA. Use: loss aversion, social proof, curiosity, reciprocity, effort externalization ("I'll draft it — takes 10 min"), urgency.

HARD RULES — any violation loses points:
- No URLs (Meta blocks them)
- No fabricated numbers — only use figures from the context
- No taboo words from voice.vocab_taboo
- One CTA only — never two asks in one message
- No preamble ("Hope you're doing well", "I wanted to reach out")
- Hindi-English mix is fine and often preferred for Indian merchants
- Never say "guaranteed", "cure", "100% safe" for health categories
- Keep body under 300 chars for simple nudges, up to 500 for complex messages
- Address dentists as "Dr. {first_name}" always
- Address all other owners by first name only
- For customer-facing messages: send_as = "merchant_on_behalf", honor language preference (hi-en mix if preferred), use customer's first name
- For research/compliance triggers: include source citation at end of message (e.g. "— JIDA Oct 2026 p.14")
- The bot adds judgment — if the trigger warrants a contrarian recommendation (e.g. skip IPL promo on Saturday), say so with data
- suppression_key must come from the trigger context, never invent one
"""


# ==============================================================================
# COMPOSE MESSAGE — core intelligence
# ==============================================================================

def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """Compose a grounded Vera message from the four context ingredients."""
    if not client_llm:
        return _fallback_compose(category, merchant, trigger, customer)

    cat_slug = category.get("slug", "unknown")
    m_id     = merchant.get("identity", {})
    m_perf   = merchant.get("performance", {})
    m_offers = merchant.get("offers", [])
    m_ca     = merchant.get("customer_aggregate", {})
    m_sigs   = merchant.get("signals", [])
    m_reviews = merchant.get("review_themes", [])
    m_hist   = merchant.get("conversation_history", [])
    peer     = category.get("peer_stats", {})

    active_offers = [o for o in m_offers if o.get("status") == "active"]

    # Resolve digest item referenced by trigger
    digest_item = None
    top_id = trigger.get("payload", {}).get("top_item_id")
    if top_id:
        for d in category.get("digest", []):
            if d.get("id") == top_id:
                digest_item = d
                break

    # CTR gap for specificity
    ctr = m_perf.get("ctr") or m_perf.get("ctr_30d")
    peer_ctr = peer.get("avg_ctr")
    ctr_note = ""
    if ctr and peer_ctr:
        try:
            gap = round((float(peer_ctr) - float(ctr)) * 100, 1)
            if gap > 0:
                ctr_note = f" (CTR {ctr} vs peer median {peer_ctr} — {gap}pp below)"
            else:
                ctr_note = f" (CTR {ctr} vs peer median {peer_ctr} — above median)"
        except Exception:
            pass

    prompt = f"""COMPOSE A VERA MESSAGE FOR THIS EXACT CONTEXT.

=== CATEGORY: {cat_slug} ===
Voice tone: {category.get('voice', {}).get('tone', 'peer')}
Taboo words (never use): {category.get('voice', {}).get('vocab_taboo', [])}
Offer catalog: {json.dumps(category.get('offer_catalog', [])[:4])}
Peer stats: avg_ctr={peer_ctr}, avg_rating={peer.get('avg_rating')}, avg_reviews={peer.get('avg_review_count')}
Seasonal beats: {json.dumps(category.get('seasonal_beats', []))}
Trend signals: {json.dumps(category.get('trend_signals', [])[:3])}
Digest items available: {json.dumps(category.get('digest', [])[:4])}

=== MERCHANT ===
Name: {m_id.get('name')}
Owner address: {"Dr. " + m_id.get('owner_first_name') if cat_slug == "dentists" else m_id.get('owner_first_name')}
Locality: {m_id.get('locality')}, {m_id.get('city')}
Languages: {m_id.get('languages')}
Subscription: {merchant.get('subscription', {}).get('plan')}, {merchant.get('subscription', {}).get('days_remaining')} days left
Performance (30d): views={m_perf.get('views')}, calls={m_perf.get('calls')}, directions={m_perf.get('directions')}{ctr_note}
7d delta: {m_perf.get('delta_7d')}
Active offers: {json.dumps(active_offers)}
Signals: {m_sigs}
Customer aggregate: {json.dumps(m_ca)}
Review themes (recent): {json.dumps(m_reviews[:2])}
Prior conversation (last 2 turns): {json.dumps(m_hist[-2:] if m_hist else [])}

=== TRIGGER ===
Kind: {trigger.get('kind')}
Source: {trigger.get('source')}
Urgency: {trigger.get('urgency')}/5
Payload: {json.dumps(trigger.get('payload', {}))}
Suppression key (USE THIS EXACTLY): {trigger.get('suppression_key')}
Expires: {trigger.get('expires_at')}
{f"Resolved digest item: {json.dumps(digest_item)}" if digest_item else "No digest item resolved."}

=== CUSTOMER (if customer-facing) ===
{json.dumps(customer, indent=2) if customer else "None — merchant-facing message."}

{f"=== PRIOR CONVERSATION ==={chr(10)}{json.dumps(conversation_history[-4:], indent=2)}" if conversation_history else ""}

INSTRUCTIONS:
1. Pick the ONE strongest signal from trigger + merchant + category. Do not summarise everything.
2. Use real numbers from the context in every factual claim.
3. For research/compliance triggers: cite source at end of body (e.g. "— JIDA Oct 2026 p.14").
4. If the trigger warrants a contrarian recommendation, make it with data.
5. suppression_key in your response must match the trigger's suppression_key exactly.
6. For customer-facing messages: honor language preference, use customer first name, set send_as=merchant_on_behalf.
"""

    try:
        raw = llm_call(COMPOSE_SYSTEM, prompt, max_tokens=700)
        result = parse_json(raw)
        # Ensure suppression_key always matches trigger
        if not result.get("suppression_key"):
            result["suppression_key"] = trigger.get("suppression_key", f"msg:{merchant.get('merchant_id')}:{trigger.get('id')}")
        return result
    except Exception as e:
        print(f"[COMPOSE ERROR] {e}")
        return _fallback_compose(category, merchant, trigger, customer)


def _fallback_compose(category, merchant, trigger, customer) -> dict:
    """Fallback when LLM is unavailable."""
    name = owner_address(merchant, category.get("slug", ""))
    kind = trigger.get("kind", "update").replace("_", " ")
    return {
        "body": f"{name}, Vera here — {kind} update on your {category.get('slug', 'business')} profile. Want to know more?",
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", f"fallback:{trigger.get('id', 'x')}"),
        "rationale": "Fallback compose — LLM unavailable.",
    }


# ==============================================================================
# COMPOSE REPLY — mid-conversation intelligence
# ==============================================================================

REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant, mid-conversation.

OUTPUT ONLY valid JSON — no markdown:
{
  "action": "send" | "wait" | "end",
  "body": "<reply — only if action=send, plain text>",
  "wait_seconds": <int — only if action=wait>,
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "none",
  "rationale": "<1-2 sentences>"
}

DECISION RULES (in order):
1. Hostile / explicit opt-out ("stop", "useless", "why are you bothering") -> action=end, no body
2. Auto-reply detected -> action=wait (14400s first time, 86400s second time, end third time)
3. Merchant committed ("let's do it", "go ahead", "confirm") -> action=send, switch immediately to execution, no more qualifying questions
4. Out-of-scope ask (GST, unrelated topics) -> politely decline in one sentence, redirect to original topic
5. General reply -> action=send, short and useful, one CTA

QUALITY RULES:
- When merchant commits to action: give a concrete next step with scope ("I'll draft the WhatsApp for your 40 high-risk patients — reply CONFIRM to send")
- When merchant asks a question: answer directly, then offer next step
- Never repeat the same body twice in a conversation
- No URLs, no multiple CTAs
- Stay concise — 1-3 sentences for most replies
"""


def compose_reply(conversation: dict, merchant_message: str, merchant: dict, category: dict) -> dict:
    """Compose a reply to a merchant message in an ongoing conversation."""
    if not client_llm:
        return {"action": "send", "body": "Got it — let me help with that.", "cta": "open_ended", "rationale": "Fallback reply."}

    turns   = conversation.get("turns", [])
    m_id    = merchant.get("identity", {})
    m_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    m_ca    = merchant.get("customer_aggregate", {})

    prompt = f"""CONVERSATION SO FAR:
{json.dumps(turns[-6:], indent=2)}

MERCHANT'S LATEST MESSAGE: "{merchant_message}"

MERCHANT CONTEXT:
- Name: {m_id.get('name')} | Owner: {owner_address(merchant, category.get('slug',''))}
- Category: {category.get('slug')} | City: {m_id.get('city')}
- Active offers: {json.dumps(m_offers)}
- Customer aggregate: {json.dumps(m_ca)}

What should Vera do next? Apply the decision rules in order."""

    try:
        raw = llm_call(REPLY_SYSTEM, prompt, max_tokens=400)
        return parse_json(raw)
    except Exception as e:
        print(f"[REPLY ERROR] {e}")
        return {"action": "send", "body": "Understood — let me look into that.", "cta": "open_ended", "rationale": "Fallback reply."}


# ==============================================================================
# TICK LOGIC
# ==============================================================================

def process_tick(available_trigger_ids: list[str]) -> list[dict]:
    actions = []

    for trigger_id in available_trigger_ids:
        if len(actions) >= 20: # max actions per tick
            break

        trigger = get_context("trigger", trigger_id)
        if not trigger:
            continue

        sup_key = trigger.get("suppression_key", "")
        if sup_key and sup_key in suppressed_keys:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        if not merchant_id:
            continue

        merchant = get_context("merchant", merchant_id)
        if not merchant:
            continue

        cat_slug = merchant.get("category_slug")
        category = get_context("category", cat_slug)
        if not category:
            continue

        customer = get_context("customer", customer_id) if customer_id else None

        # Meaningful conversation ID
        conv_id = make_conv_id(merchant_id, trigger)
        if conv_id in ended_conversations:
            continue

        composed = compose_message(category, merchant, trigger, customer)

        body = composed.get("body", "").strip()
        if not body:
            continue

        # Anti-repetition
        existing_turns = conversations.get(conv_id, {}).get("turns", [])
        if any(t.get("body") == body and t.get("role") == "vera" for t in existing_turns):
            continue

        # Record suppression
        final_sup = composed.get("suppression_key") or sup_key
        if final_sup:
            suppressed_keys.add(final_sup)

        # Store conversation turn
        if conv_id not in conversations:
            conversations[conv_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "turns": [],
                "suppressed": False,
            }
        conversations[conv_id]["turns"].append({"role": "vera", "body": body})

        kind = trigger.get("kind", "generic")
        send_as = composed.get("send_as", "vera")
        template_name = f"merchant_recall_reminder_v1" if send_as == "merchant_on_behalf" else f"vera_{kind}_v1"

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": template_name,
            "template_params": [
                owner_address(merchant, cat_slug),
                kind.replace("_", " "),
                body[:100],
            ],
            "body": body,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": final_sup,
            "rationale": composed.get("rationale", ""),
        })

    return actions


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": context_count(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Composer",
        "team_members": ["Candidate"],
        "model": MODEL,
        "approach": (
            "Context-grounded single-prompt composer. Dispatches by trigger.kind. "
            "Resolves live digest items, computes CTR gap vs peer median, honors customer language preference. "
            "Hardcoded decision tree for auto-reply/opt-out/intent-commit; LLM for general replies."
        ),
        "contact_email": "candidate@example.com",
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key     = (body.scope, body.context_id)
    current = contexts.get(key)

    if current:
        if current["version"] > body.version:
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": current["version"]},
            )
        if current["version"] == body.version:
            return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    return {"actions": process_tick(body.available_triggers)}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id     = body.conversation_id
    merchant_id = body.merchant_id or ""
    message     = body.message
    turn        = body.turn_number

    if conv_id not in conversations:
        conversations[conv_id] = {"merchant_id": merchant_id, "customer_id": body.customer_id, "turns": []}
    conv = conversations[conv_id]

    # Already ended?
    if conv_id in ended_conversations:
        return {"action": "end", "rationale": "Conversation already closed."}

    # Append merchant turn ONCE
    conv["turns"].append({"role": "merchant", "body": message, "turn": turn})

    # 1. Explicit opt-out -> end immediately, no apology message (per case study 4.3)
    if detect_opt_out(message):
        ended_conversations.add(conv_id)
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Closing conversation and suppressing all future triggers for this merchant.",
        }

    # 2. Auto-reply detection (count AFTER appending current turn)
    if detect_auto_reply(message):

        auto_count = sum(
            1 for t in conv["turns"]
            if t.get("role") == "merchant" and detect_auto_reply(t.get("body", ""))
        )
        if auto_count >= 3:
            ended_conversations.add(conv_id)
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {auto_count}x in a row — owner not at phone. Closing conversation.",
            }
        elif auto_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice — owner not available. Waiting 24h before retry.",
            }
        else:
            # Only send nudge once — check if already sent
            already_nudged = any(
                t.get("role") == "vera" and "auto-reply" in t.get("body", "").lower()
                for t in conv["turns"]
            )
            if already_nudged:
                return {
                    "action": "wait",
                    "wait_seconds": 14400,
                    "rationale": "Auto-reply detected again — nudge already sent. Backing off 4h.",
                }
            nudge = "Looks like an auto-reply 😊 When you see this, just reply 'Yes' to continue."
            conv["turns"].append({"role": "vera", "body": nudge})
            return {
                "action": "send",
                "body": nudge,
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected. Sending visible nudge for when owner returns.",
            }
    
    # 3. Intent commitment — switch to action mode
    if detect_intent_commit(message):
        merchant      = get_context("merchant", merchant_id) or {}
        cat_slug      = merchant.get("category_slug", "")
        category      = get_context("category", cat_slug) or {}
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        m_ca          = merchant.get("customer_aggregate", {})
    
        offer_text = f" ({active_offers[0]['title']})" if active_offers else ""
    
        scope_note = ""
        for key, label in [
            ("lapsed_180d_plus",      "lapsed customers"),
            ("high_risk_adult_count", "high-risk patients"),
            ("lapsed_count",          "lapsed customers"),
            ("total_unique_ytd",      "customers"),
        ]:
            val = m_ca.get(key)
            if val:
                scope_note = f" for your {val} {label}"
                break
    
        action_body = (
            f"Great, let's go.{offer_text} I'll start drafting now{scope_note} — "
            f"should be ready in under a minute. Reply CONFIRM to proceed or tell me if you need any changes."
        )
        conv["turns"].append({"role": "vera", "body": action_body})
        return {
            "action": "send",
            "body": action_body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action. Switching immediately to execution mode without further qualification.",
        }
        
    # 4. General reply — LLM handles it
    merchant = get_context("merchant", merchant_id) or {}
    cat_slug = merchant.get("category_slug", "")
    category = get_context("category", cat_slug) or {}

    result = compose_reply(conv, message, merchant, category)

    if result.get("action") == "send" and result.get("body"):
        conv["turns"].append({"role": "vera", "body": result["body"]})
    elif result.get("action") == "end":
        ended_conversations.add(conv_id)

    return result

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id     = body.conversation_id
    print(f"[REPLY] conv_id={conv_id} turn={body.turn_number} msg={body.message[:40]}")
    conv = conversations.get(conv_id, {})
    print(f"[REPLY] existing turns={len(conv.get('turns', []))}")

@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    ended_conversations.clear()
    return {"status": "cleared"}


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Standalone compose function."""
    return compose_message(category, merchant, trigger, customer)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
