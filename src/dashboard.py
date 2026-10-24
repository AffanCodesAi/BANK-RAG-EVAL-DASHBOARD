"""Local dashboard routes, SQLite evaluation history, structured logs and SSE."""
import json, logging, os, queue, re, sqlite3, threading, time, uuid
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.search import ROOT, GROQ_MODEL, build_context, generate_answer, retrieve_documents

router = APIRouter(prefix="/dashboard", tags=["evaluation-dashboard"])
resources: dict[str, Any] = {}
inference_lock = None
eval_guard = threading.Lock()
sub_lock = threading.Lock()
subscribers: set[queue.Queue] = set()
recent_logs = deque(maxlen=300)
logger = logging.getLogger("bank_rag_dashboard")
LOG_DIR = ROOT / "logs"
DB_PATH = Path(os.getenv("DASHBOARD_DB_PATH", str(ROOT / "dashboard_data" / "evaluation_history.sqlite3")))
if not DB_PATH.is_absolute(): DB_PATH = ROOT / DB_PATH
LOG_DIR.mkdir(parents=True, exist_ok=True); DB_PATH.parent.mkdir(parents=True, exist_ok=True)

TESTS = [
 {"name":"Raast transaction fee","question":"Does JS Bank charge a fee for Raast transactions?","keywords":["charge"],"refuse":False},
 {"name":"Raast per-transaction limit","question":"What is the maximum amount per Raast transaction?","keywords":["500,000","500000"],"refuse":False},
 {"name":"Raast daily limit","question":"What is the daily Raast transaction limit?","keywords":["1,000,000","1000000"],"refuse":False},
 {"name":"Unsupported corporate package limit","question":"What is the special transaction limit for the corporate package?","keywords":[],"refuse":True},
]
REFUSALS = ("could not confirm","cannot confirm","do not specify","does not specify","not enough information","cannot verify","unable to confirm","not provided","not available in the documents","could not find any information","do not contain any information","does not contain any information","do not mention","does not mention","supplied documents","provided documents","contact js bank")
CITATION = re.compile(r"(?:【\s*Source\s+\d+\s*】|\[\s*Source\s+\d+\s*\])", re.I)

class JsonFormatter(logging.Formatter):
 def format(self, r):
  d={"timestamp":datetime.fromtimestamp(r.created,timezone.utc).isoformat(timespec="milliseconds"),"level":r.levelname,"event":getattr(r,"event","application_event"),"message":r.getMessage()}
  for k in ("request_id","run_id","case_name","status","status_code","response_time_seconds","source_count","indexed_chunks","device","model","error_type","provider_status","case_index"):
   v=getattr(r,k,None)
   if v is not None: d[k]=v
  return json.dumps(d,ensure_ascii=False)
class Broadcast(logging.Handler):
 def emit(self,r):
  try:
   item=json.loads(self.format(r))
   with sub_lock: recent_logs.append(item); targets=list(subscribers)
   for q in targets:
    try:q.put_nowait(item)
    except queue.Full:
     try:q.get_nowait(); q.put_nowait(item)
     except (queue.Empty,queue.Full):pass
  except Exception:self.handleError(r)
def _setup_logs():
 if logger.handlers:return
 logger.setLevel(logging.INFO); logger.propagate=False; fmt=JsonFormatter()
 for h in (RotatingFileHandler(LOG_DIR/"app.jsonl",maxBytes=2_000_000,backupCount=3,encoding="utf-8"),logging.StreamHandler(),Broadcast()): h.setFormatter(fmt); logger.addHandler(h)
def log_event(level:int,event:str,**fields):
 _setup_logs(); logger.log(level,event,extra={"event":event,**fields})
def configure_dashboard(res,lock):
 global resources,inference_lock
 resources=res; inference_lock=lock; _setup_logs(); init_db()
def connect():
 c=sqlite3.connect(DB_PATH,timeout=15); c.row_factory=sqlite3.Row; return c
def init_db():
 with connect() as c:
  c.execute("CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,kind TEXT,status TEXT,started_at TEXT,completed_at TEXT,total_cases INTEGER DEFAULT 0,passed_cases INTEGER DEFAULT 0,failed_cases INTEGER DEFAULT 0,pass_rate REAL,total_latency REAL DEFAULT 0,error_message TEXT)")
  c.execute("CREATE TABLE IF NOT EXISTS cases(case_id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,case_index INTEGER,name TEXT,question TEXT,expected TEXT,answer TEXT,sources_json TEXT,status TEXT,retrieval_ok INTEGER,answer_ok INTEGER,citation_ok INTEGER,latency REAL,error_message TEXT)")
  c.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)")
def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def ready():
 if not resources or inference_lock is None: raise HTTPException(503,"RAG backend is not ready yet.")
def run_case(question,keywords=None,refuse=False,expected_answer=None):
 ready(); start=time.perf_counter()
 with inference_lock:
  docs=retrieve_documents(question,resources["embedding_model"],resources["collection"])
  context,sources=build_context(docs)
  if not context.strip(): raise RuntimeError("No usable reference documents were retrieved")
  answer=generate_answer(question,context,resources["groq_client"])
 latency=round(time.perf_counter()-start,3); lower_context=context.lower(); lower_answer=answer.lower()
 if refuse: retrieval_ok=True; answer_ok=any(p in lower_answer for p in REFUSALS); expected="Refusal required"
 elif expected_answer: retrieval_ok=True; answer_ok=expected_answer.lower() in lower_answer; expected=expected_answer
 else:
  keywords=keywords or []; retrieval_ok=(not keywords) or any(k.lower() in lower_context for k in keywords); answer_ok=(not keywords) or any(k.lower() in lower_answer for k in keywords); expected=", ".join(keywords) or "Manual review"
 citation_ok=bool(CITATION.search(answer)) if not refuse else True
 passed=answer_ok if (refuse or expected_answer) else retrieval_ok and answer_ok and citation_ok
 status="passed" if passed else ("needs_review" if not keywords and not expected_answer and not refuse else "failed")
 return {"question":question,"answer":answer,"sources":sources,"expected":expected,"status":status,"retrieval_ok":retrieval_ok,"answer_ok":answer_ok,"citation_ok":citation_ok,"latency_seconds":latency}
def insert_run(rid,kind):
 with connect() as c:c.execute("INSERT INTO runs(run_id,kind,status,started_at) VALUES(?,?,?,?)",(rid,kind,"running",now()))
def insert_case(rid,i,name,r):
 with connect() as c:c.execute("INSERT INTO cases(run_id,case_index,name,question,expected,answer,sources_json,status,retrieval_ok,answer_ok,citation_ok,latency,error_message) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(rid,i,name,r.get("question",""),r.get("expected"),r.get("answer"),json.dumps(r.get("sources",[]),ensure_ascii=False),r.get("status","failed"),None if r.get("retrieval_ok") is None else int(r["retrieval_ok"]),None if r.get("answer_ok") is None else int(r["answer_ok"]),None if r.get("citation_ok") is None else int(r["citation_ok"]),r.get("latency_seconds"),r.get("error_message")))
def finish_run(rid,status,total,passed,failed,latency,error=None):
 with connect() as c:c.execute("UPDATE runs SET status=?,completed_at=?,total_cases=?,passed_cases=?,failed_cases=?,pass_rate=?,total_latency=?,error_message=? WHERE run_id=?",(status,now(),total,passed,failed,round(passed/total*100,1) if total else None,round(latency,3),error,rid))
def suite_worker(rid):
 total_latency=passed=failed=0; pause=max(0,min(int(os.getenv("DASHBOARD_EVAL_PAUSE_SECONDS","15")),60)); log_event(logging.INFO,"evaluation_suite_started",run_id=rid)
 try:
  for i,t in enumerate(TESTS,1):
   log_event(logging.INFO,"evaluation_case_started",run_id=rid,case_index=i,case_name=t["name"])
   try:
    r=run_case(t["question"],t["keywords"],t["refuse"]); insert_case(rid,i,t["name"],r); total_latency+=r["latency_seconds"]
    if r["status"]=="passed":passed+=1
    else:failed+=1
    log_event(logging.INFO,"evaluation_case_completed",run_id=rid,case_index=i,case_name=t["name"],status=r["status"],response_time_seconds=r["latency_seconds"])
   except Exception as e:
    failed+=1; insert_case(rid,i,t["name"],{"question":t["question"],"expected":", ".join(t["keywords"]) or "Refusal required","status":"failed","error_message":type(e).__name__})
    log_event(logging.ERROR,"evaluation_case_failed",run_id=rid,case_index=i,case_name=t["name"],error_type=type(e).__name__)
   if i<len(TESTS) and pause:time.sleep(pause)
  finish_run(rid,"completed",len(TESTS),passed,failed,total_latency); log_event(logging.INFO,"evaluation_suite_completed",run_id=rid,status="completed",response_time_seconds=total_latency)
 except Exception as e:
  finish_run(rid,"failed",len(TESTS),passed,failed,total_latency,type(e).__name__); log_event(logging.ERROR,"evaluation_suite_failed",run_id=rid,error_type=type(e).__name__)
 finally:eval_guard.release()
def details(rid):
 with connect() as c:
  run=c.execute("SELECT * FROM runs WHERE run_id=?",(rid,)).fetchone()
  if not run:return None
  cases=c.execute("SELECT * FROM cases WHERE run_id=? ORDER BY case_index",(rid,)).fetchall()
 out=dict(run); out["cases"]=[]
 for row in cases:
  item=dict(row); item["sources"]=json.loads(item.pop("sources_json") or "[]"); item["latency_seconds"]=item.pop("latency")
  for k in ("retrieval_ok","answer_ok","citation_ok"):
   if item[k] is not None:item[k]=bool(item[k])
  out["cases"].append(item)
 out["total_latency_seconds"]=out.pop("total_latency")
 return out

class CustomRequest(BaseModel):
 question:str=Field(min_length=1,max_length=2000)
 expected_answer:str|None=Field(default=None,max_length=2000)

@router.get("/overview")
def overview():
 ready()
 with connect() as c:
  latest=c.execute("SELECT * FROM runs WHERE kind='suite' AND status='completed' ORDER BY started_at DESC LIMIT 1").fetchone()
  count=c.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"]
 return {"status":"ok","indexed_chunks":resources["collection"].count(),"embedding_device":resources["device"],"answer_model":GROQ_MODEL,"evaluation_runs":count,"latest_suite":dict(latest) if latest else None}
@router.post("/evaluate/suite",status_code=202)
def start_suite():
 ready()
 if not eval_guard.acquire(blocking=False):raise HTTPException(409,"An evaluation suite is already running.")
 rid=uuid.uuid4().hex[:12]
 try:insert_run(rid,"suite"); threading.Thread(target=suite_worker,args=(rid,),daemon=True).start()
 except Exception:eval_guard.release(); raise
 log_event(logging.INFO,"evaluation_run_queued",run_id=rid)
 return {"run_id":rid,"status":"running","case_count":len(TESTS)}
@router.post("/evaluate/custom")
def custom(req:CustomRequest):
 ready(); question=req.question.strip()
 if not question:raise HTTPException(422,"Question must not be empty or whitespace.")
 rid=uuid.uuid4().hex[:12]; insert_run(rid,"custom"); log_event(logging.INFO,"custom_evaluation_started",run_id=rid)
 try:
  r=run_case(question,[],False,req.expected_answer.strip() if req.expected_answer else None); r["name"]="Custom question"
  if not req.expected_answer:r["status"]="needs_review"
  insert_case(rid,1,"Custom question",r); finish_run(rid,"completed",1,int(r["status"]=="passed"),int(r["status"]=="failed"),r["latency_seconds"])
  log_event(logging.INFO,"custom_evaluation_completed",run_id=rid,status=r["status"],response_time_seconds=r["latency_seconds"])
  return {"run_id":rid,"result":r}
 except Exception as e:
  finish_run(rid,"failed",1,0,1,0,type(e).__name__); log_event(logging.ERROR,"custom_evaluation_failed",run_id=rid,error_type=type(e).__name__)
  raise HTTPException(502,"The custom evaluation failed. Check logs and try again.")
@router.get("/evaluations")
def history(limit:int=Query(25,ge=1,le=100)):
 with connect() as c:rows=c.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",(limit,)).fetchall()
 return {"items":[{**dict(r),"total_latency_seconds":dict(r).get("total_latency")} for r in rows]}
@router.get("/evaluations/{rid}")
def evaluation_detail(rid:str):
 r=details(rid)
 if r is None:raise HTTPException(404,"Evaluation run not found.")
 return r
@router.get("/logs")
def logs(limit:int=Query(100,ge=1,le=300)):
 with sub_lock:return {"items":list(recent_logs)[-limit:]}
@router.get("/logs/stream")
def log_stream():
 q=queue.Queue(maxsize=100)
 with sub_lock:subscribers.add(q)
 def generate():
  try:
   yield "retry: 3000\n\n"
   yield f"data: {json.dumps({'event':'connected','message':'Live log stream connected','timestamp':now()})}\n\n"
   while True:
    try:yield f"data: {json.dumps(q.get(timeout=15),ensure_ascii=False)}\n\n"
    except queue.Empty:yield ": keepalive\n\n"
  finally:
   with sub_lock:subscribers.discard(q)
 return StreamingResponse(generate(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

dashboard_router = router
