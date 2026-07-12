import os
import requests
import xml.etree.ElementTree as ET
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime, timedelta
import json
# from dotenv import load_dotenv

# 環境変数読み込み
# load_dotenv()

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
    """関連度でソートした記事を取得"""
    select_top_n = int(os.getenv("SELECT_TOP_N", 5))
    params = {
        "q": os.getenv("SEARCH_KEYWORDS", "(insect OR animal) AND (research OR study)"),
        "from": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",  # サーバー側で関連度順でソート
        "pageSize": select_top_n,  # 必要な分だけ取得
        "apiKey": os.getenv("NEWS_API_KEY")
    }

    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        # サーバー側が既にソート済みなので、そのまま返す
        return articles

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"NewsAPIエラー: {str(e)}")


def fetch_pubmed_papers():
    """PubMedから関連度の高い論文を検索・取得"""
    keywords = load_keywords()
    pubmed_keywords = keywords.get("pubmed", ["(AI OR Machine Learning) AND (research OR study)"])
    selected_keyword = pubmed_keywords[0]
    
    select_top_n = int(os.getenv("SELECT_TOP_N", 5))
    params = {
        "term": selected_keyword,
        "retmax": 100,
        "sort": "pub_date",
        "tool": "news-summarizer",
        "email": os.getenv("PUBMED_EMAIL", "your-email@example.com")
    }
    
    try:
        # ステップ1: ESearch APIで論文IDを取得
        search_response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params
        )
        search_response.raise_for_status()
        
        root = ET.fromstring(search_response.text)
        pmids = [pmid.text for pmid in root.findall(".//Id")]
        
        if not pmids:
            return []
        
        # ステップ2: 上位SELECT_TOP_N件を選択（既にサーバー側で関連度ソート済み）
        selected_pmids = pmids[:select_top_n]
        
        # ステップ3: EFetch APIで詳細情報を取得（XML形式で取得）
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(selected_pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": "news-summarizer",
            "email": os.getenv("PUBMED_EMAIL", "your-email@example.com")
        }
        
        fetch_response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=fetch_params
        )
        fetch_response.raise_for_status()
        
        papers = []
        root = ET.fromstring(fetch_response.text)
        
        # XMLから論文情報を抽出（PubmedArticle 単位で処理）
        for pubmed_article in root.findall(".//PubmedArticle"):
            article = pubmed_article.find("MedlineCitation")
            if article is None:
                continue
        
            pmid_elem = article.find(".//PMID")
            title_elem = article.find(".//ArticleTitle")

            # Abstract は複数要素の可能性があるので結合する
            abstract_texts = []
            for abstract_part in article.findall(".//Abstract/AbstractText"):
                if abstract_part is None:
                    continue
                # AbstractText はネストやタグを含む場合があるので itertext で結合
                text = "".join(abstract_part.itertext()).strip() if hasattr(abstract_part, "itertext") else (abstract_part.text or "").strip()
                label = abstract_part.get("Label")
                if label:
                    abstract_texts.append(f"{label}: {text}")
                else:
                    abstract_texts.append(text)
            abstract = "\n".join([t for t in abstract_texts if t]) if abstract_texts else "No abstract available"
            # --- Journal名 ---
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "No journal"
            # --- 発表日の取得（既存ロジックを維持しつつMedlineDateも試す） ---
            pub_year = pub_month = pub_day = None
            article_elem = article.find("Article")
            if article_elem is not None:
                article_date_elem = article_elem.find("ArticleDate[@DateType='Electronic']")
                if article_date_elem is not None:
                    pub_year = article_date_elem.findtext("Year")
                    pub_month = article_date_elem.findtext("Month")
                    pub_day = article_date_elem.findtext("Day")
            if pub_year is None:
                journal_issue_elem = article.find("Article/Journal/JournalIssue/PubDate")
                if journal_issue_elem is not None:
                    # PubDate には Year / Month / Day 以外に MedlineDate が入ることがある
                    pub_year = journal_issue_elem.findtext("Year")
                    pub_month = journal_issue_elem.findtext("Month")
                    pub_day = journal_issue_elem.findtext("Day")
                    if pub_year is None:
                        medline = journal_issue_elem.findtext("MedlineDate")
                        if medline:
                            # 例: "2023 Sep-Oct" をそのまま使う
                            pub_date = medline
                        else:
                            pub_date = "No date"
            if pub_year:
                if pub_month and pub_day:
                    pub_date = f"{pub_year}-{pub_month}-{pub_day}"
                elif pub_month:
                    pub_date = f"{pub_year}-{pub_month}"
                else:
                    pub_date = pub_year
            # 論文情報をpapersに追加
            if pmid_elem is not None:
                pmid = pmid_elem.text
                title = "".join(title_elem.itertext()).strip() if (title_elem is not None and hasattr(title_elem, "itertext")) else (title_elem.text if title_elem is not None else "No title")
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "pmid": pmid,
                    "journal": journal,
                    "pub_date": pub_date,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

        
        return papers

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"PubMed APIエラー: {str(e)}")

def fetch_arxiv_papers():
    """arXiv から論文を取得してパースする"""
    # 優先順: keywords.json の arxiv -> 環境変数 ARXIV_QUERY -> デフォルトクエリ
    keywords = load_keywords()
    arxiv_queries = keywords.get("arxiv") if keywords else None
    query = None
    if arxiv_queries and len(arxiv_queries) > 0:
        query = arxiv_queries[0]
    else:
        query = os.getenv("ARXIV_QUERY", "all:machine+learning")  # デフォルト簡易クエリ

    # arXiv API パラメータ
    select_top_n = int(os.getenv("SELECT_TOP_N", 5))
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": select_top_n,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        # Atom XML をパース（名前空間に注意）
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            published_elem = entry.find("atom:published", ns)
            id_elem = entry.find("atom:id", ns)
            # authors
            authors = []
            for a in entry.findall("atom:author", ns):
                name = a.findtext("atom:name", default=None, namespaces=ns)
                if name:
                    authors.append(name.strip())
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else "No title"
            # summary は改行や HTML 風のタグが混ざることがある -> itertext で結合
            abstract = "".join(summary_elem.itertext()).strip() if summary_elem is not None else "No abstract available"
            pub_date = None
            if published_elem is not None and published_elem.text:
                # ISO 8601 形式 (例: 2023-07-10T12:34:56Z) -> YYYY-MM-DD
                try:
                    pub_date = published_elem.text.split("T")[0]
                except Exception:
                    pub_date = published_elem.text
            else:
                pub_date = "No date"
            link = None
            # id_elem は通常記事の URL を含むが、entry 内の link rel="alternate" も確認する
            if id_elem is not None and id_elem.text:
                link = id_elem.text.strip()
            else:
                for l in entry.findall("atom:link", ns):
                    if l.get("rel") == "alternate" and l.get("href"):
                        link = l.get("href")
                        break
            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "pub_date": pub_date,
                "url": link or "https://arxiv.org",
            })
        return papers

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"arXiv API エラー: {str(e)}")
    except ET.ParseError as e:
        raise RuntimeError(f"arXiv レスポンスの XML 解析エラー: {str(e)}")
        
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

def send_notification(message: str):
    """Slack/Discordに通知"""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL") or os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("通知先Webhookが設定されていません")
    
    payload = {
        "text": message
    } if "slack" in webhook_url.lower() else {
        "content": message
    }
    requests.post(webhook_url, json=payload)

if __name__ == "__main__":
    try:
        ai_config = init_ai_client()
        target_lang = os.getenv("TARGET_LANGUAGE", "ja")
        
        # ニュースと論文の両方を取得
        #articles = fetch_ranked_news()
        papers = fetch_pubmed_papers()
        arxiv_papers = fetch_arxiv_papers() 
        all_sources = []
        
        # ニュース記事を処理
        #for article in articles:
        #    content = f"{article['title']}\n\n{article['description'] or 'No description available'}"
        #    summary = translate_and_summarize(ai_config, content, target_lang)
        #    all_sources.append({
        #        "type": "news",
        #        "summary": summary,
        #        "title": article['title'],
        #        "url": article['url']
        #    })
        
        # PubMed論文を処理（例）
        for paper in papers:
            content = f"{paper['title']}\n\n{paper['abstract']}"
            summary = translate_and_summarize(ai_config, content, target_lang)
            all_sources.append({
                "type": "paper",
                "summary": summary,
                "title": paper['title'],
                "url": paper['url'],
                "journal": paper.get("journal", "No journal"),
                "pub_date": paper.get("pub_date", "No date")
            })
            
         # arXiv を同様に処理
        for a in arxiv_papers:
            content = f"{a['title']}\n\n{a['abstract']}"
            summary = translate_and_summarize(ai_config, content, target_lang)
            all_sources.append({
                "type": "paper",
                "source": "arxiv",
                "summary": summary,
                "title": a['title'],
                "url": a['url'],
                "authors": ", ".join(a.get("authors", [])) or "No authors",
                "pub_date": a.get("pub_date", "No date")
            })       
            
        # 通知送信（例）
        for source in all_sources:
            if source["type"] == "paper":
                source_type = "📄【論文】"
                header = f"{source_type}\n*雑誌*: {source.get('journal')}\n*発表日*: {source.get('pub_date')}\n"
            else:
                source_type = "📰【ニュース】"
                header = f"{source_type}\n"
            send_notification(
                f"{header}"
                f"*翻訳要約*\n{source['summary']}\n\n"
                f"*Title*: {source['title']}\n"
                f"*URL*: {source['url']}"
            )
            
    except Exception as e:
        error_msg = f"⚠️ 致命的なエラー: {str(e)}"
        print(error_msg)
        send_notification(error_msg)
