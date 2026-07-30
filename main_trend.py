import time
import os
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
import json
# from dotenv import load_dotenv
import sqlite3

# 環境変数読み込み
# load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEN_FILE = os.path.join(BASE_DIR, "seen_papers_trend.json")
#SEEN_FILE = "seen_papers.json"
DB_PATH = os.path.join(BASE_DIR, "papers.db")
OBSIDIAN_DIR = os.path.join(BASE_DIR, "obsidian_vault", "papers")  # 保存先フォルダ

def export_papers_to_obsidian():
    os.makedirs(OBSIDIAN_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT pid, title, abstract, journal, pub_date, source, search_keyword FROM papers ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    for pid, title, abstract, journal, pub_date, source, keyword in rows:
        filename = safe_filename(pid)
        filepath = os.path.join(OBSIDIAN_DIR, filename)

        md = f"""---
pid: {json.dumps(pid)}
source: {json.dumps(source)}
journal: {json.dumps(journal)}
pub_date: {json.dumps(pub_date)}

search_keyword: {json.dumps(keyword)}

keywords: []
modality: []
application: []
network: []
organ: []
research_type: []

processed: false
---

# {title}

## Abstract

{abstract}
"""

        with open(filepath, "w") as f:
            f.write(md)
            
def safe_filename(pid: str) -> str:
    """
    arXiv URL や PubMed URL を安全なファイル名に変換する
    """
    # arXiv の場合: https://arxiv.org/abs/2606.30049v1 → 2606.30049v1
    if "arxiv.org" in pid:
        return pid.split("/")[-1] + ".md"

    # PubMed の場合はそのまま
    return pid + ".md"
            
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pid TEXT,
            title TEXT,
            abstract TEXT,
            journal TEXT,
            pub_date TEXT,
            source TEXT,
            search_keyword TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_paper_to_db(paper):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO papers
        (pid, title, abstract, journal, pub_date, source, search_keyword, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        paper["pid"],
        paper["title"],
        paper["abstract"],
        paper.get("journal", ""),
        paper["pub_date"],
        paper["source"],
        paper["search_keyword"]
    ))
    conn.commit()
    conn.close()







def load_seen_papers():
    if not os.path.exists(SEEN_FILE):
        return {"pubmed": [], "arxiv": []}
    try:
        with open(SEEN_FILE, "r") as f:
            return json.load(f)
    except:
        return {"pubmed": [], "arxiv": []}

def save_seen_papers(seen):
    print("[DEBUG] JSON保存:", SEEN_FILE)
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
        with open('keywords_trend.json', 'r') as f:
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
    print("[DEBUG] 最終PubMed:", len(final_papers))
    for p in final_papers:
        seen_pubmed.add(p["pmid"])
    print("SEEN_FILE =", SEEN_FILE)
    print("exists =", os.path.exists(SEEN_FILE))
    seen["pubmed"] = list(seen_pubmed)
    save_seen_papers(seen)
    print("[DEBUG] save_seen_papers()")
    print("[DEBUG] 保存件数:", len(seen_pubmed))
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

        papers = fetch_pubmed_papers()
        arxiv_papers = fetch_arxiv_papers()

        # --- SQLite 初期化 ---
        init_db()

        # --- PubMed 保存 ---
        for p in papers:
            save_paper_to_db({
                "pid": p["pmid"],
                "title": p["title"],
                "abstract": p["abstract"],
                "journal": p["journal"],
                "pub_date": p["pub_date"],
                "source": "pubmed",
                "search_keyword": p["search_keyword"]
            })

        # --- arXiv 保存 ---
        for a in arxiv_papers:
            save_paper_to_db({
                "pid": a["url"],
                "title": a["title"],
                "abstract": a["abstract"],
                "journal": ", ".join(a.get("authors", [])),
                "pub_date": a["pub_date"],
                "source": "arxiv",
                "search_keyword": a["search_keyword"]
            })
        export_papers_to_obsidian()
    except Exception:
        traceback.print_exc()
        raise

