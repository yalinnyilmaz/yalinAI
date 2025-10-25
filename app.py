# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, jsonify
from duckduckgo_search import DDGS
from openai import OpenAI

app = Flask(__name__)

# -------------------------------
# API Config
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Use: export OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------------
# Web search helper
# -------------------------------
def smart_web_search(query, max_results=4):
    """Search only if query clearly requests online info."""
    web_keywords = ["internet", "ara", "google", "web", "online", "search", "find", "lookup"]
    if not any(k.lower() in query.lower() for k in web_keywords):
        return []  # No web search unless user asks
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):
            results.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body")
            })
    return results


# -------------------------------
# AI response generator
# -------------------------------
def yalin_ai_response(prompt):
    """Generates multilingual, context-aware, abbreviation-friendly answers."""
    context_prompt = (
        "You are YalinAI, an intelligent, multilingual assistant. "
        "Understand all languages, emojis, abbreviations, and slang. "
        "If the question implies needing internet info, integrate search context. "
        "Respond clearly, professionally, and naturally like a real person."
    )

    web_ctx = smart_web_search(prompt)
    if web_ctx:
        context_prompt += "\n\nWeb results:\n" + "\n".join(
            [f"[{i+1}] {r['title']} - {r['snippet']} ({r['url']})" for i, r in enumerate(web_ctx)]
        )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.55,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html", page="home")

@app.route("/about")
def about():
    return render_template("about.html", page="about")

@app.route("/contact")
def contact():
    return render_template("contact.html", page="contact")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    answer = yalin_ai_response(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
