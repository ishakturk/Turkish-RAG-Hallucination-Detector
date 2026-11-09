# %% [markdown]
# # 00 · Setup and smoke test
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Goal: check that the Colab environment works end to end before any real experiment.
#
# 1. GPU, library versions
# 2. Tokenizers: how TabiBERT, mmBERT and the Newmind model split Turkish text (casing, apostrophes)
# 3. RAGTruth-TR: which string the label offsets refer to
# 4. Two existing detectors on 5 test examples (speed, output format)
# 5. TabiBERT memory and speed at long input lengths
# 6. One call to the DeepSeek API (the augmentation data generator)
#
# **Runtime:** L4 GPU. **Secrets needed:** `GH_TOKEN` and `DEEPSEEK_API_KEY`.
# **Output:** `MyDrive/hdtr/results/00_smoke.json` — send this file back.

# %% [markdown]
# ## Setup
# Mount Drive, clone the private repo with the `GH_TOKEN` secret and install the package.

# %%
import os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)

token = userdata.get("GH_TOKEN")
auth_url = f"https://x-access-token:{token}@github.com/{REPO}.git"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", "--depth", "1", auth_url, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "-q", auth_url, "main"], check=True)
# Keep the token out of .git/config.
subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", f"https://github.com/{REPO}.git"], check=True)
commit = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[colab]"], check=True)
sys.path.insert(0, f"{REPO_DIR}/src")  # editable installs are not visible to a running kernel
print("repo commit:", commit)

# %% [markdown]
# ## 1. Environment

# %%
import json, platform, time
import numpy as np
import torch, transformers, datasets

results = {"notebook": "00_setup_smoke", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M")}
results["env"] = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
    "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
}
try:
    import flash_attn
    results["env"]["flash_attn"] = flash_attn.__version__
except ImportError:
    results["env"]["flash_attn"] = None
print(json.dumps(results["env"], indent=2))

# %% [markdown]
# ## 2. Tokenizers on Turkish text
# We check three things for each tokenizer:
# - Does it keep upper case (`İ`, `I`) or lowercase internally?
# - How does it split apostrophe suffixes such as "Ocak'a"?
# - How many tokens does the same text need? (fewer tokens = more context fits)
#
# Offsets must always point into the original text; we never lowercase ourselves.

# %%
from transformers import AutoTokenizer

TOKENIZERS = {
    "tabibert": "boun-tabilab/TabiBERT",
    "mmbert": "jhu-clsp/mmBERT-base",
    "newmind_modernbert_tr": "newmindai/ModernBERT-tr-uncased-stsb-HD",
}
SAMPLE = "Başvurular 25 Ocak'a kadar İstanbul'daki IŞIK Merkezi'ne yapılır; ücret 1.250 TL’dir."

tokenizers, tok_report = {}, {}
for name, model_id in TOKENIZERS.items():
    tok = AutoTokenizer.from_pretrained(model_id)
    tokenizers[name] = tok
    enc = tok(SAMPLE, return_offsets_mapping=True, add_special_tokens=False)
    pieces = [SAMPLE[s:e] for s, e in enc["offset_mapping"]]
    tok_report[name] = {
        "n_tokens": len(enc["input_ids"]),
        "pieces": pieces,
        "decoded_keeps_case": "İstanbul" in tok.decode(enc["input_ids"]),
        "offsets_valid": all(0 <= s <= e <= len(SAMPLE) for s, e in enc["offset_mapping"]),
        "is_fast": tok.is_fast,
    }
    print(f"\n{name} ({tok_report[name]['n_tokens']} tokens, keeps case: {tok_report[name]['decoded_keeps_case']})")
    print(" | ".join(pieces))
results["tokenizers"] = tok_report

# %% [markdown]
# ## 3. RAGTruth-TR: which text do the label offsets index?
# The dataset card does not say. If the offsets refer to `answer`, the labeled text should be
# a sensible phrase that starts and ends on word boundaries inside the answer.

# %%
from hdtr.data import load_ragtruth_tr

df = load_ragtruth_tr()
results["ragtruth_tr"] = {
    "rows": len(df),
    "split_counts": df["split"].value_counts().to_dict(),
    "task_counts": df["task_type"].value_counts().to_dict(),
}

def boundary_share(texts, spans_col):
    ok = total = 0
    for text, spans in zip(texts, spans_col):
        for s in spans:
            total += 1
            st, en = s["start"], s["end"]
            inside = 0 <= st < en <= len(text)
            left = st == 0 or not (text[st - 1].isalnum() and text[st].isalnum()) if inside else False
            right = en == len(text) or not (text[en - 1].isalnum() and text[en].isalnum()) if inside else False
            ok += inside and left and right
    return ok / max(total, 1)

labeled = df[df["labels"].map(len) > 0]
results["offset_check"] = {
    "share_aligned_if_answer": boundary_share(labeled["answer"], labeled["labels"]),
    "share_aligned_if_prompt": boundary_share(labeled["prompt"], labeled["labels"]),
}
print(results["offset_check"])
for _, row in labeled.head(3).iterrows():
    s = row["labels"][0]
    print("\nANSWER SPAN:", repr(row["answer"][s["start"]:s["end"]]), "| label:", s["label"])

# %% [markdown]
# ## 4. Two existing detectors on 5 test examples
# - `newmindai/ModernBERT-tr-uncased-stsb-HD`: the best Turkish detector from prior work
# - `KRLabsOrg/lettucedect-v2-mmbert-base`: multilingual detector, never trained on Turkish
#
# We only check that loading, prediction and our span conversion work, and measure speed.

# %%
from hdtr.infer import load_detector, predict_many
from hdtr.postprocess import char_probs_to_spans

sample = df[(df["split"] == "test") & (df["labels"].map(len) > 0)].head(5)
pairs = list(zip(sample["prompt"], sample["answer"]))
results["detectors"] = {}

for name, model_id in {
    "newmind_modernbert_tr": "newmindai/ModernBERT-tr-uncased-stsb-HD",
    "krlabs_v2_mmbert": "KRLabsOrg/lettucedect-v2-mmbert-base",
}.items():
    tok, model = load_detector(model_id)
    predict_many(model, tok, pairs[:1])  # warm-up
    probs, timing = predict_many(model, tok, pairs)
    shown = []
    for (prompt, answer), p, gold in zip(pairs, probs, sample["labels"]):
        pred = char_probs_to_spans(answer, p, threshold=0.5)
        shown.append({"gold": [answer[g["start"]:g["end"]] for g in gold],
                      "pred": [s["text"] for s in pred]})
    results["detectors"][name] = {"timing": timing, "id2label": model.config.id2label, "examples": shown}
    print(f"\n{name}: {timing['sec_per_example']:.3f} s/example, truncated {timing['truncated']}")
    for item in shown[:3]:
        print("  gold:", item["gold"], "\n  pred:", item["pred"])
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. TabiBERT at long input lengths
# Forward + backward pass on random token ids at several lengths, to plan batch sizes for training.

# %%
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "boun-tabilab/TabiBERT", num_labels=2, torch_dtype=torch.bfloat16
).cuda()
model.gradient_checkpointing_enable()
long_report = {}
for length in (1024, 2048, 4096, 8192):
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(5, model.config.vocab_size - 1, (1, length), device="cuda")
    labels = torch.randint(0, 2, (1, length), device="cuda")
    try:
        start = time.perf_counter()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        torch.cuda.synchronize()
        long_report[length] = {"sec": round(time.perf_counter() - start, 3),
                               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except torch.cuda.OutOfMemoryError:
        long_report[length] = "OOM"
    model.zero_grad(set_to_none=True)
    print(length, long_report[length])
results["tabibert_long_input"] = long_report
del model
torch.cuda.empty_cache()

# %% [markdown]
# ## 6. DeepSeek API check
# One small request to `deepseek-flash` (the data generator for augmentation) to confirm that:
# - the `DEEPSEEK_API_KEY` secret works,
# - **thinking mode is really off** (it is on by default and would multiply the cost),
# - JSON output mode returns valid JSON in Turkish.
#
# Cost of this cell: a fraction of a cent. Skipped if the secret is not set.

# %%
try:
    deepseek_key = userdata.get("DEEPSEEK_API_KEY")
except Exception:
    deepseek_key = None

if deepseek_key:
    from openai import OpenAI
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "Yalnızca geçerli JSON döndür. Örnek: {\"cevap\": \"...\"}"},
            {"role": "user", "content": "Bağlam: Başvurular 15 Ocak'ta kapanır.\n"
             "Soru: Başvurular ne zaman kapanır? Tek cümlelik Türkçe yanıtı json olarak ver."},
        ],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.7,
        max_tokens=100,
    )
    usage = resp.usage.model_dump() if resp.usage else {}
    details = usage.get("completion_tokens_details") or {}
    message = resp.choices[0].message
    try:
        parsed = json.loads(message.content)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    results["deepseek"] = {
        "model": resp.model,
        "content": message.content,
        "valid_json": parsed is not None,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "has_reasoning_content": bool(getattr(message, "reasoning_content", None)),
        "usage": usage,
        "seconds": round(time.perf_counter() - start, 2),
    }
    print(json.dumps(results["deepseek"], ensure_ascii=False, indent=2))
    assert not results["deepseek"]["has_reasoning_content"], "Thinking mode is still on: check the request."
else:
    results["deepseek"] = None
    print("DEEPSEEK_API_KEY not set; skipped.")

# %% [markdown]
# ## Save results
# Download `MyDrive/hdtr/results/00_smoke.json` and send it back.

# %%
out_path = f"{DRIVE_DIR}/results/00_smoke.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print("saved", out_path)
