import time
import os
import requests
import xml.etree.ElementTree as ET
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime, timedelta
import json


import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import math
from datetime import datetime

def compute_recency_score(pub_date):
    """発表日から Recency スコアを計算（新しいほど高い）"""
    try:
        d = datetime.strptime(pub_date, "%Y-%m-%d")
    except:
        return 0.0

    days = (datetime.now() - d).days
    years = days / 365.0

    return 1 / (1 + years)  # 新しいほど 1 に近い

def compute_citation_score(citation_count):
    """引用数から Citation スコアを計算（ログスケール）"""
    if citation_count is None:
        return 0.0
    return math.log(1 + citation_count)

def compute_novelty_score(ai_summary_text):
    """LLM の出力から Novelty スコア（1〜5）を抽出"""
    for line in ai_summary_text.splitlines():
        if "Novelty" in line:
            nums = [int(s) for s in line.split() if s.isdigit()]
            if nums:
                return nums[0] / 5.0  # 0〜1 に正規化
    return 0.5  # デフォルト

def compute_final_score(similarity, novelty, recency, citation):
    return (
        0.45 * similarity +
        0.25 * novelty +
        0.20 * recency +
        0.10 * citation
    )



# ============================================================
# 1. FAISS 永続化（保存・読み込み）
# ============================================================

EMBED_DIM = 384  # MiniLM-L6-v2 の埋め込み次元

def save_faiss_index(index, paper_ids,
                     index_path="faiss.index",
                     ids_path="paper_ids.json"):
    faiss.write_index(index, index_path)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f, ensure_ascii=False, indent=2)

def load_faiss_index(index_path="faiss.index",
                     ids_path="paper_ids.json"):
    if not os.path.exists(index_path) or not os.path.exists(ids_path):
        return None, []
    index = faiss.read_index(index_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        paper_ids = json.load(f)
    return index, paper_ids

# ============================================================
# 2. 埋め込みモデル
# ============================================================

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# 3. FAISS インデックス初期化（新規 or 復元）
# ============================================================

index, paper_ids = load_faiss_index()

if index is None:
    print("FAISS index not found. Creating new index...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    paper_ids = []
else:
    print(f"Loaded FAISS index with {len(paper_ids)} papers.")

# ============================================================
# 4. 論文を FAISS に追加する関数
# ============================================================

def add_paper_to_index(paper):
    text = paper["title"] + " " + paper["abstract"]
    vec = embedder.encode(text, normalize_embeddings=True)
    vec = np.array([vec]).astype("float32")

    index.add(vec)
    paper_ids.append(paper["id"])

# ============================================================
# 5. 類似度検索
# ============================================================

def search_similar_papers(query_text, top_k=5):
    vec = embedder.encode(query_text, normalize_embeddings=True)
    vec = np.array([vec]).astype("float32")

    D, I = index.search(vec, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        pid = paper_ids[idx]
        results.append({"pid": pid, "score": float(score)})
    return results

# ============================================================
# 6. 敦郎さんの興味プロファイル
# ============================================================

interest_text = """
Restormer, MDTA, GDFN, PromptIR, MRI画像復元, MoE, Router,
Degradation-aware learning, Transformer-based image restoration
"""

# ============================================================
# 7. ここから既存の論文取得ロジック（PubMed / arXiv）
# ============================================================

# 例：敦郎さんの既存コードで new_papers を作っている前提
# new_papers = fetch_pubmed_and_arxiv()  ← 既存の処理をそのまま使う

# new_papers は以下の形式を想定：
# {
#   "id": "arxiv:2401.12345",
#   "title": "...",
#   "abstract": "...",
#   "url": "...",
#   "pdf_url": "..."
# }


# ============================================================
# 11. FAISS インデックスを保存（永続化）
# ============================================================

save_faiss_index(index, paper_ids)
print("FAISS index saved.")

# ============================================================
# 10. FAISS結果を Slack に送る（改善フォーマット）
# ============================================================

def summarize_with_ai(ai_config, title, abstract):
    """LLMで Summary / Tags / Novelty を生成する"""
    prompt = f"""
以下の論文について、3つの情報を生成してください。

1. Summary（日本語で簡潔に要約）
2. Tags（技術タグを3〜6個）
3. Novelty（新規性を1〜5で評価し、理由を1行）

論文タイトル:
{title}

要旨:
{abstract}
"""

    if ai_config["provider"] == "gemini":
        response = ai_config["client"].generate_content(prompt)
        text = response.text
    else:
        response = ai_config["client"].chat.completions.create(
            model=ai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        text = response.choices[0].message.content.strip()

    return text


# Slack送信用フォーマット
def send_faiss_result_to_slack(ai_config, faiss_results):
    parent_ts = send_notification("🔍 *FAISS 推薦論文（複合スコア版）*")

    for r in faiss_results:
        pid = r["pid"]
        similarity = r["score"]

        # SQLite からメタデータ取得
        conn = sqlite3.connect("papers.db")
        cur = conn.cursor()
        cur.execute("SELECT title, abstract, url, pub_date, citation FROM papers WHERE pid=?", (pid,))
        row = cur.fetchone()

        if not row:
            continue

        title, abstract, url, pub_date, citation_count = row

        # LLMで Summary / Tags / Novelty を生成
        ai_summary = summarize_with_ai(ai_config, title, abstract)

        # Novelty スコア抽出
        novelty_score = compute_novelty_score(ai_summary)

        # Recency スコア
        recency_score = compute_recency_score(pub_date)

        # Citation スコア
        citation_score = compute_citation_score(citation_count)

        # 複合スコア
        final_score = compute_final_score(similarity, novelty_score, recency_score, citation_score)

        # Slackへ送信
        message = f"""
        📘【FAISS 推薦論文】
        Final Score: {final_score:.3f}
        Similarity: {similarity:.3f}
        Novelty: {novelty_score:.2f}
        Recency: {recency_score:.2f}
        Citation: {citation_score:.2f}
        
        *Journal*: {journal}
        *Published*: {pub_date}
        
        {ai_summary}
        
        *Title*: {title}
        *URL*: {url}
        """
        
        send_notification(message, thread_ts=parent_ts)



# ============================================================
# 11. FAISS インデックスを保存（永続化）
# ============================================================

save_faiss_index(index, paper_ids)
print("FAISS index saved.")





# from dotenv import load_dotenv

# 環境変数読み込み
# load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEN_FILE = os.path.join(BASE_DIR, "seen_papers.json")
#SEEN_FILE = "seen_papers.json"

def load_seen_papers():
    if not os.path.exists(SEEN_FILE):
        return {"pubmed": [], "arxiv": []}
    try:
        with open(SEEN_FILE, "r") as f:
            return json.load(f)
    except:
        return {"pubmed": [], "arxiv": []}

def save_seen_papers(seen):
    with open(SEEN_FILE, "w") as f:
        json.dump(seen, f, indent=2)

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}
from datetime import datetime

def normalize_pub_date(pub_date):
    if pub_date is None or pub_date == "No date":
        return datetime.min

    # 例: "2026-Jan"
    if "-" in pub_date and pub_date.split("-")[1].isalpha():
        year, mon = pub_date.split("-")
        mon = MONTH_MAP.get(mon, "01")
        return datetime.strptime(f"{year}-{mon}-01", "%Y-%m-%d")

    # 例: "2026-06-25"
    try:
        return datetime.strptime(pub_date, "%Y-%m-%d")
    except:
        pass

    # 例: "2026-06"
    try:
        return datetime.strptime(pub_date, "%Y-%m")
    except:
        pass

    # MedlineDate など（例: "2020 Jan-Feb"）
    for m in MONTH_MAP:
        if m in pub_date:
            year = pub_date.split()[0]
            mon = MONTH_MAP[m]
            return datetime.strptime(f"{year}-{mon}-01", "%Y-%m-%d")

    return datetime.min
    
def load_keywords():
    """keywords.jsonから検索ワードを読み込み、変数を展開"""
    try:
        with open('keywords.json', 'r') as f:
            data = json.load(f)
        
        # journalsの変数をpubmedの検索ワードに展開
        journals = data.get("journals", {})
        pubmed_keywords = data.get("pubmed", [])
        arxiv_keywords = data.get("arxiv", [])
        news_keywords = data.get("news", [])
        
        # PubMed検索ワードの変数展開
        expanded_pubmed = []
        for keyword in pubmed_keywords:
            expanded = keyword
            for journal_key, journal_value in journals.items():
                expanded = expanded.replace(f"@{journal_key}", journal_value)
            expanded_pubmed.append(expanded)
        
        # arXiv検索ワードの変数展開
        expanded_arxiv = []
        for keyword in arxiv_keywords:
            expanded = keyword
            for journal_key, journal_value in journals.items():
                expanded = expanded.replace(f"@{journal_key}", journal_value)
            expanded_arxiv.append(expanded)
        
        # News検索ワードの変数展開
        expanded_news = []
        for keyword in news_keywords:
            expanded = keyword
            for journal_key, journal_value in journals.items():
                expanded = expanded.replace(f"@{journal_key}", journal_value)
            expanded_news.append(expanded)
        
        return {
            "journals": journals,
            "pubmed": expanded_pubmed,
            "arxiv": expanded_arxiv,
            "news": expanded_news
        }
    except FileNotFoundError:
        return {
            "journals": {},
            "pubmed": ["(AI OR Machine Learning) AND (research OR study)"],
            "arxiv": ["all:machine+learning"],
            "news": ["(AI OR Machine Learning) AND (research OR study)"]
        }

def get_model_name(provider: str) -> str:
    """Secretで指定されたモデル名を取得（デフォルト値付き）"""
    return {
        "openai": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
        #"gemini": os.getenv("GEMINI_MODEL", "gemini-1.0-pro"),
    }.get(provider.lower())

def init_ai_client():
    """AIクライアント初期化（モデル選択対応版）"""
    if os.getenv("DEEPSEEK_API_KEY"):
        return {
            "client": OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            ),
            "model": "deepseek-chat",
            "provider": "deepseek"
        }
    elif os.getenv("OPENAI_API_KEY"):
        return {
            "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            "model": get_model_name("openai"),
            "provider": "openai"
        }
    elif os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return {
            "client": genai.GenerativeModel(get_model_name("gemini")),
            "model": get_model_name("gemini"),
            "provider": "gemini"
        }
    raise RuntimeError("有効なAI APIキーが設定されていません")

def fetch_news():
    """NewsAPIから24時間以内の記事を取得"""
    params = {
        "q": os.getenv("SEARCH_KEYWORDS", "(AI OR Machine Learning) AND (research OR study)"),
        "from": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "language": "en",  # 英語記事のみ
        "apiKey": os.getenv("NEWS_API_KEY"),
        "pageSize": 5  # 最大5記事
    }
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"NewsAPIエラー: {str(e)}")

def fetch_ranked_news():
    """複数の検索ワードからニュースを取得"""
    keywords = load_keywords()
    news_keywords = keywords.get("news", ["(AI OR Machine Learning) AND (research OR study)"])
    select_top_n = int(os.getenv("SELECT_TOP_N", 5))
    all_articles = []

    # 各検索ワードをループ処理
    for keyword in news_keywords:
        params = {
            "q": keyword,
            "from": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": select_top_n,
            "apiKey": os.getenv("NEWS_API_KEY")
        }

        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            
            # 検索ワードを記録
            for article in articles:
                article["search_keyword"] = keyword
            
            all_articles.extend(articles)

        except requests.exceptions.RequestException as e:
            print(f"NewsAPIエラー (キーワード: {keyword}): {str(e)}")
            continue
    
    return all_articles


def fetch_pubmed_papers():
    print("[DEBUG] 現在の作業ディレクトリ:", os.getcwd())
    print("[DEBUG] seen_papers.json の保存先:", os.path.abspath(SEEN_FILE))

    """PubMedから複数キーワードで論文を取得し、重複を除外し、最後に最新5件だけ返す"""
    keywords = load_keywords()
    pubmed_keywords = keywords.get("pubmed", ["(AI OR Machine Learning) AND (research OR study)"])

    select_top_n = int(os.getenv("SELECT_TOP_N", 5))
    all_papers = []

    # 🔥 過去に出力した論文IDを読み込む
    seen = load_seen_papers()
    seen_pubmed = set(seen.get("pubmed", []))

    for selected_keyword in pubmed_keywords:

        # --- ESearch ---
        params = {
            "term": selected_keyword,
            "retmax": 100,
            "sort": "pub_date",
            "tool": "news-summarizer",
            "email": os.getenv("PUBMED_EMAIL", "your-email@example.com")
        }

        try:
            time.sleep(0.34)
            search_response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params
            )
            search_response.raise_for_status()

            root = ET.fromstring(search_response.text)
            pmids = [pmid.text for pmid in root.findall(".//Id")]
            
            print(f"[DEBUG] PubMed '{selected_keyword}' 件数: {len(pmids)}")
            
            if not pmids:
                continue

            #selected_pmids = pmids[:select_top_n]
            selected_pmids = pmids
            # --- EFetch ---
            for pmid in selected_pmids:

                # 🔥 重複チェック（過去に出力済みならスキップ）
                if pmid in seen_pubmed:
                    continue

                fetch_params = {
                    "db": "pubmed",
                    "id": pmid,
                    "rettype": "abstract",
                    "retmode": "xml",
                    "tool": "news-summarizer",
                    "email": os.getenv("PUBMED_EMAIL", "your-email@example.com")
                }

                time.sleep(0.34)
                fetch_response = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params=fetch_params
                )
                fetch_response.raise_for_status()

                root = ET.fromstring(fetch_response.text)

                for pubmed_article in root.findall(".//PubmedArticle"):
                    article = pubmed_article.find("MedlineCitation")
                    if article is None:
                        continue

                    pmid_elem = article.find(".//PMID")
                    title_elem = article.find(".//ArticleTitle")

                    # Abstract
                    abstract_texts = []
                    for abstract_part in article.findall(".//Abstract/AbstractText"):
                        text = "".join(abstract_part.itertext()).strip()
                        label = abstract_part.get("Label")
                        abstract_texts.append(f"{label}: {text}" if label else text)
                    abstract = "\n".join(abstract_texts) if abstract_texts else "No abstract available"

                    # Journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else "No journal"

                    # --- 発表日の取得 ---
                    pub_date = "No date"

                    # ① ArticleDate（Electronic）
                    article_elem = article.find("Article")
                    if article_elem is not None:
                        article_date_elem = article_elem.find("ArticleDate[@DateType='Electronic']")
                        if article_date_elem is not None:
                            year = article_date_elem.findtext("Year")
                            month = article_date_elem.findtext("Month")
                            day = article_date_elem.findtext("Day")
                            if year:
                                pub_date = f"{year}-{month or ''}-{day or ''}".strip("-")

                    # ② JournalIssue → PubDate
                    if pub_date == "No date":
                        journal_issue_elem = article.find("Article/Journal/JournalIssue/PubDate")
                        if journal_issue_elem is not None:
                            year = journal_issue_elem.findtext("Year")
                            month = journal_issue_elem.findtext("Month")
                            day = journal_issue_elem.findtext("Day")

                            if year:
                                if month and day:
                                    pub_date = f"{year}-{month}-{day}"
                                elif month:
                                    pub_date = f"{year}-{month}"
                                else:
                                    pub_date = year

                    # ③ MedlineDate
                    if pub_date == "No date":
                        medline = journal_issue_elem.findtext("MedlineDate") if journal_issue_elem is not None else None
                        if medline:
                            pub_date = medline

                    # --- 🔥 新規論文として追加（保存はここではしない） ---
                    if pmid_elem is not None:
                        print(f"[DEBUG] PubMed 新規論文追加: PMID={pmid}, pub_date={pub_date}, keyword={selected_keyword}")
                        all_papers.append({
                            "title": "".join(title_elem.itertext()).strip() if title_elem is not None else "No title",
                            "abstract": abstract,
                            "pmid": pmid,
                            "journal": journal,
                            "pub_date": pub_date,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "search_keyword": selected_keyword
                        })

        except Exception as e:
            print(f"PubMed APIエラー (キーワード: {selected_keyword}): {str(e)}")
            continue

    # --- 🔥 最新5件だけ抽出 ---
    final_papers = sorted(all_papers, key=lambda x: normalize_pub_date(x["pub_date"]), reverse=True)[:5]

    # --- 🔥 最新5件だけ保存 ---
    for p in final_papers:
        seen_pubmed.add(p["pmid"])

    seen["pubmed"] = list(seen_pubmed)
    save_seen_papers(seen)

    return final_papers

    
def fetch_arxiv_papers():
    """arXivから複数キーワードで論文を取得し、重複を除外し、最後に最新5件だけ返す"""
    keywords = load_keywords()
    arxiv_queries = keywords.get("arxiv", [])
    if not arxiv_queries:
        arxiv_queries = [os.getenv("ARXIV_QUERY", "all:machine+learning")]

    select_top_n = int(os.getenv("SELECT_TOP_N", 5))
    all_papers = []

    # 🔥 過去に出力した論文ID（URL）を読み込む
    seen = load_seen_papers()
    seen_arxiv = set(seen.get("arxiv", []))

    for query in arxiv_queries:
        time.sleep(0.3)

        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 100,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()

            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError:
                time.sleep(1)
                resp = requests.get(base_url, params=params, timeout=15)
                root = ET.fromstring(resp.text)

            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                published_elem = entry.find("atom:published", ns)
                id_elem = entry.find("atom:id", ns)

                authors = []
                for a in entry.findall("atom:author", ns):
                    name = a.findtext("atom:name", default=None, namespaces=ns)
                    if name:
                        authors.append(name.strip())

                title = "".join(title_elem.itertext()).strip() if title_elem is not None else "No title"
                abstract = "".join(summary_elem.itertext()).strip() if summary_elem is not None else "No abstract available"

                pub_date = "No date"
                if published_elem is not None and published_elem.text:
                    pub_date = published_elem.text.split("T")[0]

                # URL（arXiv ID）
                url = id_elem.text.strip() if id_elem is not None else "https://arxiv.org"

                # 🔥 重複チェック（過去に出力済みならスキップ）
                if url in seen_arxiv:
                    continue

                # 🔥 新規論文として追加（保存はここではしない）
                all_papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "pub_date": pub_date,
                    "url": url,
                    "search_keyword": query
                })

        except Exception as e:
            print(f"arXiv APIエラー (クエリ: {query}): {str(e)}")
            continue

    # --- 🔥 最新5件だけ抽出 ---
    final_papers = sorted(all_papers, key=lambda x: x["pub_date"], reverse=True)[:5]

    # --- 🔥 最新5件だけ保存 ---
    for p in final_papers:
        seen_arxiv.add(p["url"])

    seen["arxiv"] = list(seen_arxiv)
    save_seen_papers(seen)

    return final_papers

        
def translate_and_summarize(ai_config: dict, text: str, target_lang: str = "ja") -> str:
    """翻訳&要約（要約だけを返す。雑誌名/日付は外で使う）"""
    prompt = f"""以下の原文について、{target_lang}で課題が何か、その課題をどうやって解決したかをまとめてください。雑誌名や発表日は出力しないでください。箇条書きやヘッダは不要[...]

原文:
{text}
"""
    if ai_config["provider"] == "gemini":
        response = ai_config["client"].generate_content(prompt)
        # gemini のレスポンス取得方法に合わせて要約テキストを返す
        return response.text if hasattr(response, "text") else str(response)
    else:
        response = ai_config["client"].chat.completions.create(
            model=ai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

def send_notification(message: str, thread_ts: str = None):
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    channel_id = os.getenv("SLACK_CHANNEL_ID")

    if not slack_token or not channel_id:
        raise ValueError("SlackトークンまたはチャンネルIDが設定されていません")

    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "channel": channel_id,
        "text": message,
        "unfurl_links": False,
        "unfurl_media": False
    }

    if thread_ts:
        payload["thread_ts"] = thread_ts

    response = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)
    data = response.json()

    if not data.get("ok"):
        raise RuntimeError(f"Slack APIエラー: {data}")

    return data.get("ts")

def extract_primary_keyword(query: str) -> str:
    """
    検索式の最初のキーワードだけを抽出する。
    例:
      "\"image restoration\" AND (deep learning ...)" → image restoration
      "(deep learning OR machine learning) AND MRI" → deep learning OR machine learning
    """
    q = query.strip()

    # ① ダブルクォートで始まる場合 → 最初の "..." を抜き出す
    if q.startswith('"'):
        end = q.find('"', 1)
        if end != -1:
            return q[1:end]

    # ② 括弧で始まる場合 → 最初の (...) を抜き出す
    if q.startswith("("):
        end = q.find(")")
        if end != -1:
            return q[1:end]

    # ③ AND で区切る → 最初の語を取る
    if "AND" in q:
        return q.split("AND")[0].strip()

    # ④ それ以外 → 全体を返す
    return q

if __name__ == "__main__":
    try:
        print("DEBUG SLACK_CHANNEL_ID =", os.getenv("SLACK_CHANNEL_ID"))
        print("DEBUG SLACK_BOT_TOKEN =", os.getenv("SLACK_BOT_TOKEN"))
        ai_config = init_ai_client()
        print("DEBUG AI_PROVIDER =", ai_config["provider"])
        print("DEBUG AI_MODEL =", ai_config["model"])

        # 1. PubMed / arXiv の論文を取得
        papers = fetch_pubmed_papers()
        arxiv_papers = fetch_arxiv_papers()

        # 2. 取得した論文を FAISS に追加
        new_papers = []

        # PubMed
        for p in papers:
            new_papers.append({
                "id": f"pubmed:{p['pmid']}",
                "title": p["title"],
                "abstract": p["abstract"],
                "url": p["url"],
                "journal": p.get("journal", "Unknown Journal"),
                "pub_date": p.get("pub_date", "No date"),
                "citation": 0
            })


        # arXiv
        for a in arxiv_papers:
            new_papers.append({
                "id": a["url"],
                "title": a["title"],
                "abstract": a["abstract"],
                "url": a["url"],
                "journal": "arXiv",
                "pub_date": a.get("pub_date", "No date"),
                "citation": 0
            })


        # FAISS に追加
        for paper in new_papers:
            add_paper_to_index(paper)
            
        # SQLite に保存（FAISS検索で使うメタデータ）
        conn = sqlite3.connect("papers.db")
        cur = conn.cursor()
        
        # テーブル作成（初回のみ）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                pid TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                url TEXT,
                pub_date TEXT,
                citation INTEGER
            )
        """)
        
        # 論文を保存
        for paper in new_papers:
            cur.execute("""
                INSERT OR REPLACE INTO papers (pid, title, abstract, url, journal, pub_date, citation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                paper["id"],
                paper["title"],
                paper["abstract"],
                paper["url"],
                paper["journal"],
                paper["pub_date"],
                paper["citation"]
            ))

        
        conn.commit()
        conn.close()

        # 3. 類似度検索
        similar_results = search_similar_papers(interest_text, top_k=5)

        # 4. Slack通知（FAISS結果のみ）
        send_faiss_result_to_slack(ai_config, similar_results)

        # 5. FAISS インデックス保存
        save_faiss_index(index, paper_ids)
        print("FAISS index saved.")

    except Exception as e:
        error_msg = f"⚠️ 致命的なエラー: {str(e)}"
        print(error_msg)
        send_notification(error_msg)
