import os
import requests
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime, timedelta
# from dotenv import load_dotenv

# 環境変数読み込み
# load_dotenv()

def get_model_name(provider: str) -> str:
    """Secretで指定されたモデル名を取得（デフォルト値付き）"""
    return {
        "openai": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-1.0-pro"),
    }.get(provider.lower())

def init_ai_client():
    """AIクライアント初期化（モデル選択対応版）"""
    if os.getenv("DEEPSEEK_API_KEY"):
        return {
            "client": OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            ),
            "model": "deepseek-chat"
        }
    elif os.getenv("OPENAI_API_KEY"):
        return {
            "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            "model": get_model_name("openai")
        }
    elif os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return {
            "client": genai.GenerativeModel(get_model_name("gemini")),  # ← ここでインスタンスを作成
            "model": get_model_name("gemini")
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
    """relevancy_scoreでランキングした記事を取得"""
    params = {
        "q": os.getenv("SEARCH_KEYWORDS", "(insect OR animal) AND (research OR study)"),
        "from": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",  # 関連度順でソート
        "pageSize": int(os.getenv("MAX_ARTICLES", 20)),  # デフォルト20件取得
        "apiKey": os.getenv("NEWS_API_KEY")
    }

    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        # relevancy_scoreで降順ソート
        ranked_articles = sorted(
            articles,
            key=lambda x: x.get("relevancy_score", 0),
            reverse=True
        )[:int(os.getenv("SELECT_TOP_N", 5))]  # デフォルト上位5件
        
        return ranked_articles

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"NewsAPIエラー: {str(e)}")


def translate_and_summarize(ai_config: dict, text: str, target_lang: str = "ja") -> str:
    """翻訳&要約（モデル選択対応版）"""
    prompt = f"""
    以下のテキストについて、{target_lang}の要約文のみを生成してください。他の解説・注意事項・装飾は一切不要です。

    原文:
    {text}
    """
    
    if isinstance(ai_config["client"], genai.GenerativeModel):
        response = ai_config["client"].generate_content(prompt)  # ← 直接使用
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
        "text": message  # Slack
    } if "slack" in webhook_url.lower() else {
        "content": message  # Discord
    }
    requests.post(webhook_url, json=payload)

if __name__ == "__main__":
    try:
        ai_config = init_ai_client()
        # articles = fetch_news()
        articles = fetch_ranked_news()
        
        if not articles:
            send_notification("⚠️ 今日の該当記事が見つかりませんでした")
            exit()

        for article in articles:
            content = f"{article['title']}\n\n{article['description'] or 'No description available'}"
            summary = translate_and_summarize(
                ai_config,
                content,
                os.getenv("TARGET_LANGUAGE", "ja")
            )
            send_notification(
                f"*【翻訳要約】*\n{summary}\n\n"
                f"*Original Title*: {article['title']}\n"
                f"*URL*: {article['url']}"
            )
            
    except Exception as e:
        error_msg = f"⚠️ 致命的なエラー: {str(e)}"
        print(error_msg)
        send_notification(error_msg)
