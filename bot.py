"""
Vera — magicpin Merchant AI Assistant
A high-quality, context-grounded message composer for merchant engagement.
"""

import os
import time
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ==============================================================================
# CONFIG
# ==============================================================================

# Switch provider via environment variable:
#   LLM_PROVIDER=deepseek   (default — free/cheap, good for development)
#   LLM_PROVIDER=anthropic  (best quality, use for final submission)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek").lower()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
 
START_TIME = time.time()
app = FastAPI(title="Vera Bot")
 
# --- Build the right client depending on provider ---
if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
    import anthropic as _anthropic
    client_llm  = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    MODEL       = "claude-sonnet-4-20250514"
    print(f"[LLM] Using Anthropic — {MODEL}")
 
elif LLM_PROVIDER == "deepseek" and DEEPSEEK_API_KEY:
    from openai import OpenAI as _OpenAI
    client_llm  = _OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    MODEL       = "deepseek-chat"
    print(f"[LLM] Using DeepSeek — {MODEL}")
 
elif LLM_PROVIDER == "groq" and GROQ_API_KEY:
    from openai import OpenAI as _OpenAI
    client_llm  = _OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    MODEL       = "llama-3.3-70b-versatile"
    print(f"[LLM] Using Groq — {MODEL}")

else:
    client_llm  = None
    MODEL       = "none"
    print("[LLM] No API key found — fallback mode only")

# ==============================================================================
# IN-MEMORY STATE
# ==============================================================================

# (scope, context_id) -> {version, payload}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> {merchant_id, customer_id, turns: [{role, body}], suppressed: bool}
conversations: dict[str, dict] = {}

# suppression_key -> bool (already sent)
suppressed_keys: set[str] = set()

# merchant_id -> set of conversation_ids (to avoid re-opening ended conversations)
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


def detect_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    patterns = [
        r"thank you for contacting",
        r"aapki jaankari ke liye bahut.{0,10}shukriya",
        r"main ek automated assistant",
        r"our team will respond shortly",
        r"we will get back to you",
        r"this is an automated",
        r"auto.?reply",
        r"out of office",
        r"currently unavailable",
        r"aapki madad ke liye shukriya.*automated",
        r"we have received your message",
    ]
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in patterns)


def detect_opt_out(message: str) -> bool:
    """Detect explicit opt-out or hostility."""
    patterns = [
        r"\bstop\b",
        r"\bnot interested\b",
        r"\bdo not contact\b",
        r"\bblock\b",
        r"\bspam\b",
        r"\buseless\b",
        r"\bstop messaging\b",
        r"\bband karo\b",
        r"\bbother me\b",
    ]
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in patterns)


def detect_intent_commit(message: str) -> bool:
    """Detect merchant committing to action."""
    patterns = [
        r"\blets? do it\b",
        r"\bgo ahead\b",
        r"\byes.{0,10}(do it|proceed|start|let's|lets)\b",
        r"\bconfirm\b",
        r"\bproceed\b",
        r"\b(haan|ha|han).{0,5}(karo|chalao|bhejo|send)\b",
        r"\bwhat'?s? next\b",
        r"\bok.{0,10}proceed\b",
    ]
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in patterns)


# ==============================================================================
# COMPOSER — the core intelligence
# ==============================================================================

COMPOSE_SYSTEM = """You are Vera, magicpin's AI assistant for merchant growth in India.
You compose WhatsApp messages to merchants (and occasionally their customers) that are:
- Specific: use real numbers, dates, prices, source citations from the context
- Category-fit: dentists get clinical/peer tone; salons get warm/practical; restaurants get operator tone; gyms get coaching voice; pharmacies get trustworthy/precise
- Merchant-fit: use owner first name, reference their actual metrics and offers
- Trigger-relevant: clearly communicate why this message is being sent right now
- Engagement-compelling: one clear CTA, low friction, use curiosity/loss aversion/social proof

OUTPUT FORMAT — respond ONLY with valid JSON:
{
  "body": "<WhatsApp message body — concise, no markdown formatting, plain text>",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<unique dedup key>",
  "rationale": "<2-3 sentences: signal chosen, why now, what it should achieve>"
}

HARD RULES:
1. No URLs (Meta blocks them in templates)
2. No fabricated data — only use what's in the context
3. No taboo words for the category (check voice.vocab_taboo)
4. One CTA per message — not multiple asks
5. Hindi-English mix is fine and often preferred for Indian merchants
6. Never say "guaranteed", "cure", "100% safe" for health categories
7. Keep body under 300 chars for simple nudges, up to 500 for complex messages
8. No preamble like "I hope you're doing well"
9. For customer-facing messages (recall, winback), set send_as = "merchant_on_behalf"
10. Use owner_first_name or Dr. {first_name} — never generic "Dear Merchant"
"""


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """Core compose function. Returns the full ComposedMessage dict."""
    if not client_llm:
        return _fallback_compose(category, merchant, trigger, customer)

    # Build a rich, structured prompt
    cat_slug = category.get("slug", "unknown")
    m_id = merchant.get("identity", {})
    m_perf = merchant.get("performance", {})
    m_offers = merchant.get("offers", [])
    m_signals = merchant.get("signals", [])
    m_ca = merchant.get("customer_aggregate", {})
    m_reviews = merchant.get("review_themes", [])
    m_hist = merchant.get("conversation_history", [])

    active_offers = [o for o in m_offers if o.get("status") == "active"]

    # Resolve digest item if trigger references one
    digest_item = None
    if trigger.get("payload", {}).get("top_item_id"):
        item_id = trigger["payload"]["top_item_id"]
        for d in category.get("digest", []):
            if d.get("id") == item_id:
                digest_item = d
                break

    peer_stats = category.get("peer_stats", {})

    prompt = f"""COMPOSE A VERA MESSAGE.

=== CATEGORY: {cat_slug} ===
Voice tone: {category.get('voice', {}).get('tone', 'peer')}
Taboo words: {category.get('voice', {}).get('vocab_taboo', [])}
Offer catalog: {json.dumps(category.get('offer_catalog', [])[:4])}
Peer stats: avg_ctr={peer_stats.get('avg_ctr')}, avg_rating={peer_stats.get('avg_rating')}, avg_reviews={peer_stats.get('avg_review_count')}
Seasonal beats: {json.dumps(category.get('seasonal_beats', []))}
Trend signals: {json.dumps(category.get('trend_signals', [])[:3])}

=== MERCHANT ===
Name: {m_id.get('name')}
Owner first name: {m_id.get('owner_first_name')}
City/Locality: {m_id.get('city')}, {m_id.get('locality')}
Languages: {m_id.get('languages')}
Subscription: {merchant.get('subscription', {}).get('status')}, {merchant.get('subscription', {}).get('plan')}, {merchant.get('subscription', {}).get('days_remaining')} days left
Performance (30d): views={m_perf.get('views')}, calls={m_perf.get('calls')}, directions={m_perf.get('directions')}, ctr={m_perf.get('ctr')} (peer median: {peer_stats.get('avg_ctr')})
7d delta: {m_perf.get('delta_7d')}
Active offers: {json.dumps(active_offers)}
Signals: {m_signals}
Customer aggregate: {json.dumps(m_ca)}
Review themes (recent): {json.dumps(m_reviews[:2])}
Last conversation: {json.dumps(m_hist[-2:] if m_hist else [])}

=== TRIGGER ===
Kind: {trigger.get('kind')}
Source: {trigger.get('source')}
Urgency: {trigger.get('urgency')}/5
Payload: {json.dumps(trigger.get('payload', {}))}
Suppression key: {trigger.get('suppression_key')}
Expires: {trigger.get('expires_at')}
{f'Digest item: {json.dumps(digest_item)}' if digest_item else ''}

=== CUSTOMER (if customer-facing) ===
{json.dumps(customer) if customer else 'None — this is a merchant-facing message'}

{f"=== PRIOR CONVERSATION TURNS ==={chr(10)}{json.dumps(conversation_history[-4:])}" if conversation_history else ""}

Now compose the message. Pick the ONE strongest signal from trigger+merchant+category.
Make every sentence earn its place. Use real numbers from the context.
"""

    try:
        if LLM_PROVIDER == "anthropic":
            resp = client_llm.messages.create(
                model=MODEL,
                max_tokens=600,
                temperature=0,
                system=COMPOSE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:  # deepseek (openai-compatible)
            resp = client_llm.chat.completions.create(
                model=MODEL,
                max_tokens=600,
                temperature=0,
                messages=[
                    {"role": "system", "content": COMPOSE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        # Ensure suppression_key fallback
        if not result.get("suppression_key"):
            result["suppression_key"] = trigger.get("suppression_key", f"msg:{merchant.get('merchant_id')}:{trigger.get('id')}")
        return result
    except Exception as e:
        print(f"[COMPOSE ERROR] {e}")
        return _fallback_compose(category, merchant, trigger, customer)


def _fallback_compose(category, merchant, trigger, customer) -> dict:
    """Simple fallback if LLM is unavailable."""
    name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "there")
    kind = trigger.get("kind", "update")
    return {
        "body": f"Hi {name}, Vera here. Quick update on your {category.get('slug', 'business')} profile — {kind.replace('_', ' ')}. Want to know more?",
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", f"fallback:{trigger.get('id', 'x')}"),
        "rationale": "Fallback compose — LLM unavailable.",
    }


def compose_reply(
    conversation: dict,
    merchant_message: str,
    merchant: dict,
    category: dict,
) -> dict:
    """Compose a reply to a merchant message in an ongoing conversation."""
    if not client_llm:
        return {"action": "send", "body": "Got it — let me help you with that.", "cta": "open_ended", "rationale": "Fallback reply."}

    turns = conversation.get("turns", [])
    m_id = merchant.get("identity", {})

    system = """You are Vera, magicpin's AI merchant assistant. You are mid-conversation.
Respond to the merchant's latest message. Keep it short, useful, action-oriented.

OUTPUT ONLY JSON:
{
  "action": "send" | "wait" | "end",
  "body": "<reply body — only if action=send>",
  "wait_seconds": <int — only if action=wait>,
  "cta": "open_ended" | "binary_yes_no" | "none",
  "rationale": "<1-2 sentences>"
}

Rules:
- If merchant said explicit opt-out/hostility -> action=end
- If auto-reply detected -> action=wait (14400s) or end if repeated
- If merchant committed to action -> immediately execute, don't ask more qualifying questions
- Stay on topic; politely decline out-of-scope asks
- No URLs, no multiple CTAs
"""

    prompt = f"""Conversation so far:
{json.dumps(turns[-6:], indent=2)}

Merchant's latest message: "{merchant_message}"

Merchant info: {m_id.get('name')}, {m_id.get('city')}, {m_id.get('languages')}
Category: {category.get('slug')}

What should Vera do next?"""

    try:
        if LLM_PROVIDER == "anthropic":
            resp = client_llm.messages.create(
                model=MODEL,
                max_tokens=400,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:  # deepseek (openai-compatible)
            resp = client_llm.chat.completions.create(
                model=MODEL,
                max_tokens=400,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[REPLY ERROR] {e}")
        return {"action": "send", "body": "Understood, let me look into that.", "cta": "open_ended", "rationale": "Fallback reply."}


# ==============================================================================
# TICK LOGIC — decide what to send this tick
# ==============================================================================

def process_tick(available_trigger_ids: list[str]) -> list[dict]:
    """Decide which triggers to act on and compose messages."""
    actions = []
    cap = 20  # max actions per tick

    for trigger_id in available_trigger_ids:
        if len(actions) >= cap:
            break

        trigger = get_context("trigger", trigger_id)
        if not trigger:
            continue

        # Check suppression
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

        # Check if conversation already ended for this trigger
        conv_id = f"conv_{merchant_id}_{trigger_id}"
        if conv_id in ended_conversations:
            continue

        # Compose
        composed = compose_message(category, merchant, trigger, customer)

        body = composed.get("body", "").strip()
        if not body:
            continue

        # Anti-repetition: check if we sent this exact body recently
        existing_conv = conversations.get(conv_id, {})
        existing_turns = existing_conv.get("turns", [])
        if any(t.get("body") == body and t.get("role") == "vera" for t in existing_turns):
            continue

        # Record suppression
        final_sup = composed.get("suppression_key") or sup_key
        if final_sup:
            suppressed_keys.add(final_sup)

        # Store in conversation state
        if conv_id not in conversations:
            conversations[conv_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "turns": [],
                "suppressed": False,
            }
        conversations[conv_id]["turns"].append({"role": "vera", "body": body})

        # Template name based on trigger kind
        kind = trigger.get("kind", "generic")
        template_name = f"vera_{kind}_v1"

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trigger_id,
            "template_name": template_name,
            "template_params": [
                merchant.get("identity", {}).get("owner_first_name", ""),
                kind.replace("_", " "),
                body[:100],
            ],
            "body": body,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": final_sup,
            "rationale": composed.get("rationale", ""),
        }
        actions.append(action)

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
            "Context-grounded composer using Claude. Dispatches by trigger.kind. "
            "Retrieves live digest items, merchant signals, and customer state. "
            "Anti-repetition, auto-reply detection, graceful exit logic."
        ),
        "contact_email": "candidate@example.com",
        "version": "1.0.0",
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
    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current:
        if current["version"] > body.version:
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": current["version"]},
            )
        if current["version"] == body.version:
            # Idempotent — same version is a no-op
            return {
                "accepted": False,
                "reason": "stale_version",
                "current_version": current["version"],
            }

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
    actions = process_tick(body.available_triggers)
    return {"actions": actions}


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
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or ""
    message = body.message
    turn = body.turn_number

    # Get or create conversation
    if conv_id not in conversations:
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": body.customer_id,
            "turns": [],
        }
    conv = conversations[conv_id]

    # Track merchant message
    conv["turns"].append({"role": "merchant", "body": message, "turn": turn})

    # Already ended?
    if conv_id in ended_conversations:
        return {"action": "end", "rationale": "Conversation already closed."}

    # --- Decision tree ---

    # 1. Explicit opt-out
    if detect_opt_out(message):
        ended_conversations.add(conv_id)
        return {
            "action": "send",
            "body": "Apologies for the interruption — won't message again. If you'd like to reconnect later, just say 'Hi Vera'. 🙏",
            "cta": "none",
            "rationale": "Merchant explicitly opted out. Sending brief apology then ending conversation.",
        }

    # 2. Auto-reply detection
    if detect_auto_reply(message):
        conv["turns"].append({"role": "merchant", "body": message, "turn": turn})
        # Count auto-replies in this conversation
        auto_count = sum(
            1 for t in conv["turns"]
            if t.get("role") == "merchant" and detect_auto_reply(t.get("body", ""))
        )
        if auto_count >= 2:
            ended_conversations.add(conv_id)
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {auto_count}× — owner not at phone. Closing conversation.",
            }
        elif auto_count == 1:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Second consecutive auto-reply — waiting 24h before retry.",
            }
        else:
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Auto-reply detected (canned WhatsApp response). Backing off 4h for owner to return.",
            }

    # 3. Intent commitment — switch to action mode
    if detect_intent_commit(message):
        merchant = get_context("merchant", merchant_id) or {}
        cat_slug = merchant.get("category_slug", "")
        category = get_context("category", cat_slug) or {}
        m_name = merchant.get("identity", {}).get("owner_first_name", "")
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        offer_text = f" Your active offer: {active_offers[0]['title']}." if active_offers else ""

        action_body = (
            f"Great, let's go.{offer_text} I'll start drafting now — "
            f"should be ready in under a minute. Reply CONFIRM to proceed or tell me if you need any changes."
        )
        conv["turns"].append({"role": "vera", "body": action_body})
        return {
            "action": "send",
            "body": action_body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action. Switching immediately to execution mode without further qualification.",
        }

    # 4. General reply — let LLM handle it
    merchant = get_context("merchant", merchant_id) or {}
    cat_slug = merchant.get("category_slug", "")
    category = get_context("category", cat_slug) or {}

    result = compose_reply(conv, message, merchant, category)

    # Track Vera's response
    if result.get("action") == "send" and result.get("body"):
        conv["turns"].append({"role": "vera", "body": result["body"]})
    elif result.get("action") == "end":
        ended_conversations.add(conv_id)

    return result


@app.post("/v1/teardown")
async def teardown():
    """Wipe all state at end of test."""
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    ended_conversations.clear()
    return {"status": "cleared"}


# ==============================================================================
# STANDALONE COMPOSE (for bot.py interface)
# ==============================================================================

def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Standalone compose function for direct use / submission.jsonl generation."""
    return compose_message(category, merchant, trigger, customer)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
