import fitz  # PyMuPDF
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-Reranker-4B"  # or "Qwen/Qwen3-Reranker-0.6B" / "-8B"
NUM_PAGES = 2                        # how many pages from the start of each PDF to use
MAX_LENGTH = 3072                    # max tokens per (query + pages) pair
BATCH_SIZE = 1                       # lower this if you hit out-of-memory
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# ----------------------------------------------------------------------
# Load model once (4-bit quantized)
# ----------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading {MODEL_ID} in 4-bit ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
).eval()
print("Model loaded.")

token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")

prefix = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer can only be "
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)


# ----------------------------------------------------------------------
# Reranker functions
# ----------------------------------------------------------------------
def _format(query, doc, instruction):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _tokenize(pairs):
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens
    inputs = tokenizer.pad(
        inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH
    )
    return {k: v.to(model.device) for k, v in inputs.items()}


@torch.no_grad()
def _score(pairs):
    inputs = _tokenize(pairs)
    logits = model(**inputs).logits[:, -1, :]
    two = torch.stack(
        [logits[:, token_false_id], logits[:, token_true_id]], dim=1
    )
    probs = torch.nn.functional.log_softmax(two, dim=1)[:, 1].exp()
    return probs.float().cpu().tolist()


def score_docs(query, doc_texts, instruction, progress=None):
    """Return a relevance score (0..1) for every document, in original order."""
    pairs = [_format(query, d, instruction) for d in doc_texts]

    # Sort by length so each batch has similar-length items (less padding waste)
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]))
    scores = [0.0] * len(pairs)

    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        batch_scores = _score([pairs[i] for i in idx])
        for i, s in zip(idx, batch_scores):
            scores[i] = s
        if progress is not None:
            progress(min(start + BATCH_SIZE, len(order)) / len(order),
                     desc="Reranking")
    return scores


# ----------------------------------------------------------------------
# PDF handling: first NUM_PAGES pages only
# ----------------------------------------------------------------------
def extract_first_pages(pdf_path, num_pages=NUM_PAGES):
    doc = fitz.open(pdf_path)
    try:
        parts = []
        for i in range(min(num_pages, doc.page_count)):
            parts.append(doc[i].get_text("text"))
        return " ".join(" ".join(parts).split())  # clean whitespace
    finally:
        doc.close()


# ----------------------------------------------------------------------
# Gradio callback
# ----------------------------------------------------------------------
def run_rerank(files, query, top_n, instruction, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload at least one PDF.")
    if not query or not query.strip():
        raise gr.Error("Please enter a query.")

    instruction = (instruction or "").strip() or DEFAULT_INSTRUCTION

    docs, skipped = [], []
    for f in files:
        path = f if isinstance(f, str) else f.name  # handles old/new Gradio
        name = path.replace("\\", "/").split("/")[-1]
        text = extract_first_pages(path)
        if text:
            docs.append({"file": name, "text": text})
        else:
            skipped.append(name)

    if not docs:
        raise gr.Error(
            f"No text found in the first {NUM_PAGES} pages. These may be scanned PDFs "
            "(images) - they need OCR first."
        )

    scores = score_docs(
        query.strip(), [d["text"] for d in docs], instruction, progress
    )

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ranked = ranked[: int(top_n)]

    rows = [
        [rank, round(score, 4), d["file"],
         d["text"][:400] + ("..." if len(d["text"]) > 400 else "")]
        for rank, (d, score) in enumerate(ranked, start=1)
    ]

    status = f"Reranked {len(docs)} PDF(s) using their first {NUM_PAGES} pages."
    if skipped:
        status += f" Skipped (no text found): {', '.join(skipped)}"
    return rows, status


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
with gr.Blocks(title="Qwen3 PDF Reranker") as demo:
    gr.Markdown("# PDF Reranker (Qwen3-Reranker, 4-bit)\n"
                "Upload PDFs and type a query. PDFs are ranked by relevance "
                f"using only their **first {NUM_PAGES} pages**.")

    with gr.Row():
        with gr.Column():
            files = gr.File(
                label="Upload PDFs",
                file_count="multiple",
                file_types=[".pdf"],
            )
            query = gr.Textbox(label="Query", lines=2,
                               placeholder="What do you want to find?")
            top_n = gr.Slider(1, 50, value=10, step=1, label="Top N results")
            with gr.Accordion("Advanced", open=False):
                instruction = gr.Textbox(
                    label="Task instruction",
                    value=DEFAULT_INSTRUCTION,
                    lines=2,
                )
            btn = gr.Button("Rerank", variant="primary")

        with gr.Column():
            status = gr.Markdown()
            out = gr.Dataframe(
                headers=["Rank", "Score", "File", "Text preview"],
                datatype=["number", "number", "str", "str"],
                wrap=True,
                interactive=False,
            )

    btn.click(
        run_rerank,
        inputs=[files, query, top_n, instruction],
        outputs=[out, status],
    )

if __name__ == "__main__":
    demo.launch()