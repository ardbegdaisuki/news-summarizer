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
        news_keywords = data.get("news", [])
        
        # PubMed検索ワードの変数展開
        expanded_pubmed = []
        for keyword in pubmed_keywords:
            expanded = keyword
            for journal_key, journal_value in journals.items():
                expanded = expanded.replace(f"@{journal_key}", journal_value)
            expanded_pubmed.append(expanded)
        
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
            "news": expanded_news
        }
    except FileNotFoundError:
        return {
            "journals": {},
            "pubmed": ["(AI OR Machine Learning) AND (research OR study)"],
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
        
        # XMLから論文情報を抽出
        for article in root.findall(".//MedlineCitation"):
            pmid_elem = article.find(".//PMID")
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//AbstractText")
            # --- 追加：Journal名 ---
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "No journal"
        
            # --- 追加：発表日（PubDate） ---
            pub_year = article.findtext(".//PubDate/Year")
            pub_month = article.findtext(".//PubDate/Month")
            pub_day = article.findtext(".//PubDate/Day")
        
            # 日付が欠けている場合のフォールバック
            if pub_year:
                if pub_month and pub_day:
                    pub_date = f"{pub_year}-{pub_month}-{pub_day}"
                elif pub_month:
                    pub_date = f"{pub_year}-{pub_month}"
                else:
                    pub_date = pub_year
            else:
                pub_date = "No date"
        
            if pmid_elem is not None:
                pmid = pmid_elem.text
                title = title_elem.text if title_elem is not None else "No title"
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "pmid": pmid,
                    "journal": journal,        # ← 追加
                    "pub_date": pub_date,      # ← 追加                    
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
        
        return papers

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"PubMed APIエラー: {str(e)}")


def translate_and_summarize(ai_config: dict, text: str, target_lang: str = "ja") -> str:
    """翻訳&要約（モデル選択対応版）"""
    prompt = f"""
    以下のテキストについて、{target_lang}の要約文、雑誌名、発表日のみを生成してください。他の解説・注意事項・装飾は一切不要です。

    原文:
    {text}
    """
    
    if ai_config["provider"] == "gemini":
        response = ai_config["client"].generate_content(prompt)
        return response.text
    else:
        response = ai_config["client"].chat.completions.create(
            model=ai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

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
        
        # PubMed論文を処理
        for paper in papers:
            content = f"{paper['title']}\n\n{paper['abstract']}"
            summary = translate_and_summarize(ai_config, content, target_lang)
            all_sources.append({
                "type": "paper",
                "summary": summary,
                "title": paper['title'],
                "url": paper['url']
            })
        
        if not all_sources:
            send_notification("⚠️ 今日の該当記事・論文が見つかりませんでした")
            exit()
        
        # 通知を送信
        for source in all_sources:
            source_type = "📄【論文】" if source["type"] == "paper" else "📰【ニュース】"
            send_notification(
                f"{source_type}\n"
                f"*翻訳要約*\n{source['summary']}\n\n"
                f"*Title*: {source['title']}\n"
                f"*URL*: {source['url']}"
            )
            
    except Exception as e:
        error_msg = f"⚠️ 致命的なエラー: {str(e)}"
        print(error_msg)
        send_notification(error_msg)
