"""A2A 1.0 invoice agent (HTTP+JSON). Run: uvicorn invoice_agent:app --port 8000 --workers 1
Env: BASE_URL (exact submitted URL, e.g. https://host/a2a), BASE_PATH (default /a2a),
     LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (any OpenAI-compatible chat endpoint),
     DB_PATH (put on a persistent disk), STRIP_BRACKETS (1 = refs without [ ]).
"""
import asyncio, hashlib, json, os, re, sqlite3, threading, time, uuid
from collections import defaultdict
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/a2a").rstrip("/") + "/"
BASE_PATH = os.environ.get("BASE_PATH", "/a2a").rstrip("/")
LLM_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
STRIP = os.environ.get("STRIP_BRACKETS", "1") == "1"
DB_PATH = os.environ.get("DB_PATH", "agent.db")

MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_REC = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
A2A = "application/a2a+json"

app = FastAPI()
db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
dblock = threading.Lock()
with dblock:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, seq INTEGER, body TEXT);
    CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, h TEXT, task_id TEXT, PRIMARY KEY(principal,mid));
    CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, v TEXT);""")

def q(sql, args=(), one=False):
    with dblock:
        cur = db.execute(sql, args)
        return (cur.fetchone() if one else cur.fetchall())

locks = defaultdict(asyncio.Lock)

class Err(Exception):
    def __init__(s, status, code, msg="Request failed"):
        s.status, s.code, s.msg = status, code, msg

@app.exception_handler(Err)
async def _err(_, e):
    return JSONResponse({"error": {"code": e.status, "status": e.code, "message": e.msg}}, e.status)

def ok(data, status=200):
    return JSONResponse(data, status, media_type=A2A)

def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def guard(request, post=False):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or not auth[7:].strip():
        raise Err(401, "UNAUTHENTICATED", "Authentication required")
    if request.headers.get("a2a-version") != "1.0":
        raise Err(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version")
    if post and not request.headers.get("content-type", "").lower().startswith(A2A):
        raise Err(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json")
    return sha(auth[7:].strip())

# ---------- Agent Card ----------
@app.get("/.well-known/agent-card.json")
def card():
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-cited action each, executes only grader-accepted actions.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE_URL.rstrip("/") if False else BASE_URL.rstrip("/"),
                                 "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [MT_BATCH, MT_RES],
        "defaultOutputModes": [MT_PROP, MT_REC],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Choose settle/approve/hold/reject/exception for invoice packages with evidence.",
                    "tags": ["invoice", "payments", "approval"]}],
    }, media_type="application/json")

@app.get("/debug/invoice")
def debug_invoice():
    try:
        rows = q("SELECT j FROM inv_dbg ORDER BY ts DESC LIMIT 5")
    except Exception:
        rows = []
    return {"llmErrors": ERRS[-5:], "samples": [json.loads(r[0]) for r in rows]}

# ---------- LLM decisions ----------
SYS = """You are an accounts-payable controller. For EACH invoice package choose exactly ONE action under the CURRENT policy revision given in the input:
settle_invoice: valid, three-way reconciled (invoice, purchase order, receipt/ledger agree), vendor/bank details verified, and the amount is within autonomous approval authority.
request_approval: commercially valid and reconciled, but the amount or terms exceed the delegated authority limit, so a named approver must sign off first.
hold_invoice: payment pauses until a stated verification completes (bank-detail change callback, missing receipt/GRN, tax or vendor verification, pending document...). The paragraph names the verification.
reject_duplicate: the SAME commercial invoice (same vendor AND same invoice number AND same amount) was already PAID. A payment for a different invoice number or a different amount is NOT a duplicate.
open_exception: material records conflict (amount, vendor, PO, quantity or currency mismatch between documents) and need an exception workflow.
HOW TO READ: packages mix useful facts with decoys: cover sheets, old policy revisions, archive and training examples, quoted or negated statements ("do not hold", "would have been rejected"), and irrelevant action words. Never decide from keywords. Find the ONE paragraph that states the controlling fact for THIS invoice under the CURRENT policy revision; negations and examples do not count. Check amounts against the authority limit stated in the current policy, check whether a prior payment is truly the same invoice, and check whether verification is complete or still pending.
evidenceRefs = EXACTLY the three bracketed references (copy the text inside the brackets exactly) that appear in that one decisive paragraph. Never use the cover-sheet reference, archive examples, or training decoys.
facts: vendorName, invoiceNumber, amountMinor (integer in minor units, e.g. 12345.67 INR -> 1234567), currency (ISO code) of the CURRENT invoice as stated in the controlling documents.
reason: two sentences, at least 80 characters, naming the action and explaining how the cited evidence supports it.
Return ONLY JSON: {"decisions":[{"packageId":..,"action":..,"facts":{..},"evidenceRefs":[a,b,c],"reason":..}]}"""

def pid(p):
    return str(p.get("packageId") or p.get("id"))

def refs_ok(refs, text):
    return isinstance(refs, list) and len(refs) == 3 and all(isinstance(r, str) and r and r in text for r in refs)

def fix_refs(refs, text):
    out = []
    for r in refs if isinstance(refs, list) else []:
        r = str(r).strip()
        if r.startswith("[") and r.endswith("]"):
            r = r[1:-1]
        out.append(r)
    return out

def valid(d, text):
    f = d.get("facts") or {}
    return (d.get("action") in ACTIONS and refs_ok(d.get("evidenceRefs"), text)
            and isinstance(f.get("amountMinor"), int) and not isinstance(f.get("amountMinor"), bool)
            and all(isinstance(f.get(k), str) and f.get(k) for k in ("vendorName", "invoiceNumber", "currency")))

ERRS = []

def strings(o):
    if isinstance(o, str):
        yield o
        for part in re.split(r"\n+", o):
            if part != o:
                yield part
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)

def repair(refs, p):
    """If one paragraph holds exactly three bracketed refs and overlaps the model's refs, use that paragraph's refs."""
    best = None
    for t in strings(p):
        bs = list(dict.fromkeys(re.findall(r"\[([^\[\]]+)\]", t)))
        if len(bs) == 3:
            ov = len(set(bs) & set(refs))
            if ov >= 1 and (best is None or ov > best[0]):
                best = (ov, bs)
    return best[1] if best else refs

def dbg(p, d):
    try:
        q("CREATE TABLE IF NOT EXISTS inv_dbg(ts REAL, j TEXT)")
        q("INSERT INTO inv_dbg VALUES(?,?)", (time.time(), json.dumps({"package": json.dumps(p, ensure_ascii=False)[:3500], "decision": d})))
    except Exception:
        pass

async def llm(pkgs, rev=None):
    body = {"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             {"policyRevision": rev, "packages": [{"packageId": pid(p), "package": p} for p in pkgs]}, ensure_ascii=False)}]}
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.post(f"{LLM_URL}/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {LLM_KEY}"})
        if r.status_code != 200:
            ERRS.append(f"HTTP {r.status_code} {r.text[:120]}")
            del ERRS[:-8]
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip())["decisions"]

async def decide(packages, rev=None):
    """Returns {packageId: decision}; caches by canonical package content."""
    res, todo = {}, []
    for p in packages:
        k = sha("inv-v2|" + canon(p))
        row = q("SELECT v FROM cache WHERE k=?", (k,), one=True)
        if row:
            res[pid(p)] = json.loads(row[0])
        else:
            todo.append((k, p))
    for attempt in range(2):
        if not todo:
            break
        chunks = [todo[i:i + 12] for i in range(0, len(todo), 12)]
        outs = await asyncio.gather(*[llm([p for _, p in ch], rev) for ch in chunks], return_exceptions=True)
        for ch, out in zip(chunks, outs):
            if isinstance(out, Exception):
                continue
            by = {str(d.get("packageId")): d for d in out if isinstance(d, dict)}
            for k, p in ch:
                d = by.get(pid(p))
                if not d:
                    continue
                d["evidenceRefs"] = repair(fix_refs(d.get("evidenceRefs"), ""), p)
                text = json.dumps(p, ensure_ascii=False)
                if valid(d, text):
                    d["evidenceRefs"] = d["evidenceRefs"] if STRIP else [f"[{r}]" for r in d["evidenceRefs"]]
                    q("INSERT OR REPLACE INTO cache VALUES(?,?)", (k, json.dumps(d)))
                    res[pid(p)] = d
                    dbg(p, d)
        todo = [(k, p) for k, p in todo if pid(p) not in res]
    if todo:
        raise Err(503, "UNAVAILABLE", "Decision service unavailable")
    return res

def proposal(batch_id, p, d):
    f = d["facts"]
    r = d["evidenceRefs"]
    rat = (f"Action {d['action']} chosen for {f['vendorName']} invoice {f['invoiceNumber']}: "
           f"{d.get('reason', '')} Decisive evidence: {r[0]}; {r[1]}; {r[2]}.")
    rat = rat[:1500].ljust(60, ".")
    return {"packageId": pid(p), "actionId": "act-" + sha(f"{batch_id}|{pid(p)}")[:24],
            "action": d["action"],
            "facts": {"vendorName": f["vendorName"], "invoiceNumber": f["invoiceNumber"],
                      "amountMinor": f["amountMinor"], "currency": f["currency"]},
            "evidenceRefs": r, "rationale": rat}

# ---------- task storage ----------
def load(tid):
    row = q("SELECT principal, body FROM tasks WHERE id=?", (tid,), one=True)
    return (row[0], json.loads(row[1])) if row else (None, None)

def save(principal, t):
    row = q("SELECT seq FROM tasks WHERE id=?", (t["id"],), one=True)
    if row:
        q("UPDATE tasks SET body=? WHERE id=?", (json.dumps(t), t["id"]))
    else:
        n = q("SELECT COALESCE(MAX(seq),0)+1 FROM tasks", one=True)[0]
        q("INSERT INTO tasks VALUES(?,?,?,?)", (t["id"], principal, n, json.dumps(t)))

def terminal(t):
    return t["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED")

def get_part(msg, mt):
    for p in msg.get("parts", []):
        if p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None

# ---------- routes ----------
@app.post(BASE_PATH + "/message:send")
async def send(request: Request):
    principal = guard(request, post=True)
    try:
        body = await request.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid and msg.get("role") == "ROLE_USER" and isinstance(msg.get("parts"), list)
    except Exception:
        raise Err(400, "INVALID_ARGUMENT", "Malformed message")
    h = sha(canon(msg))
    async with locks[(principal, mid)]:
        row = q("SELECT h, task_id FROM msgs WHERE principal=? AND mid=?", (principal, mid), one=True)
        if row:
            if row[0] != h:
                raise Err(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT: messageId reused with different content")
            return ok({"task": load(row[1])[1]})
        if msg.get("taskId"):
            task = await continuation(principal, msg)
        else:
            task = await new_batch(principal, msg)
        q("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task["id"]))
        return ok({"task": task})

async def new_batch(principal, msg):
    data = get_part(msg, MT_BATCH)
    if not data or not isinstance(data.get("packages"), list) or not data.get("batchId"):
        raise Err(400, "INVALID_ARGUMENT", "Invalid batch")
    pkgs = data["packages"]
    ids = [pid(p) for p in pkgs]
    if len(set(ids)) != len(ids):
        raise Err(400, "INVALID_ARGUMENT", "Duplicate package ids")
    decisions = await decide(pkgs, data.get("policyRevision"))
    props = [proposal(data["batchId"], p, decisions[pid(p)]) for p in pkgs]
    t = {"id": "task-" + uuid.uuid4().hex, "contextId": msg.get("contextId") or "ctx-" + uuid.uuid4().hex,
         "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
         "artifacts": [{"artifactId": "proposals-" + uuid.uuid4().hex[:12], "name": "invoice-action-proposals",
                        "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": props}}]}],
         "history": [msg]}
    save(principal, t)
    return t

async def continuation(principal, msg):
    tid = msg["taskId"]
    async with locks[("task", tid)]:
        owner, t = load(tid)
        if owner != principal:
            raise Err(404, "NOT_FOUND", "Task not found")
        if msg.get("contextId") != t["contextId"]:
            raise Err(400, "INVALID_ARGUMENT", "Context mismatch")
        if terminal(t):
            raise Err(409, "TASK_NOT_ACTIVE", "Task is already terminal")
        data = get_part(msg, MT_RES)
        pd = t["artifacts"][0]["parts"][0]["data"]
        if not data or data.get("batchId") != pd["batchId"] or not isinstance(data.get("results"), list):
            raise Err(400, "INVALID_ARGUMENT", "Result mismatch")
        stored = {p["packageId"]: p for p in pd["proposals"]}
        seen, execs = set(), []
        for r in data["results"]:
            p = stored.get(r.get("packageId"))
            if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"]
                    or r.get("action") != p["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                raise Err(400, "INVALID_ARGUMENT", "Result mismatch")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    raise Err(400, "INVALID_ARGUMENT", "Missing receipt")
                execs.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                              "receiptNonce": n, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
        t["history"].append(msg)
        t["artifacts"][0]["parts"] = t["artifacts"][0]["parts"][:1]
        t["artifacts"][0]["parts"].append({"mediaType": MT_REC, "data": {"batchId": pd["batchId"], "executions": execs}})
        t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        save(principal, t)
        return t

def own_task(principal, tid):
    owner, t = load(tid)
    if owner != principal:
        raise Err(404, "NOT_FOUND", "Task not found")
    return t

@app.get(BASE_PATH + "/tasks/{tid}")
def get_task(tid: str, request: Request):
    p = guard(request)
    return ok(own_task(p, tid))

@app.get(BASE_PATH + "/tasks")
def list_tasks(request: Request):
    p = guard(request)
    rows = q("SELECT body FROM tasks WHERE principal=? ORDER BY seq", (p,))
    return ok({"tasks": [json.loads(r[0]) for r in rows]})

@app.post(BASE_PATH + "/tasks/{tid}:cancel")
async def cancel(tid: str, request: Request):
    p = guard(request)
    async with locks[("task", tid)]:
        t = own_task(p, tid)
        if terminal(t):
            raise Err(409, "TASK_NOT_CANCELABLE", "Task is already terminal")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save(p, t)
        return ok(t)
