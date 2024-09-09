import asyncio
import os
import re
import trafilatura
import litellm
from redis import asyncio as aioredis
import hashlib
from duckduckgo_search import AsyncDDGS
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters, ApplicationBuilder
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
from pytubefix import YouTube

telegram_token = os.environ.get("TELEGRAM_TOKEN", "xxx")
model = os.environ.get("LLM_MODEL", "openrouter/openai/gpt-4o-mini")
#lang = os.environ.get("TS_LANG", "Ameri")
ddg_region = os.environ.get("DDG_REGION", "wt-wt")
chunk_size = int(os.environ.get("CHUNK_SIZE", 10000))
allowed_users = os.environ.get("ALLOWED_USERS", "")
#os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY_ENV", "")
litellm.set_verbose=True
redis_url = os.environ.get("REDIS_URL", "")
zyte_api_key = os.environ.get("ZYTE_API_KEY", "")

# Define youtube_pattern as a global variable
youtube_pattern = re.compile(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/")

# Initialize Redis
redis_client = None

def init_redis():
    global redis_client
    redis_client = aioredis.from_url(redis_url)

system_prompt =os.environ.get("SUMMARY_LLM_PROMPT", """
    Do NOT repeat unmodified content.
    Do NOT mention anything like "Here is the summary:" or "Here is a summary of the video in 2-3 sentences:" etc.
    User will only give you youtube video subtitles, For summarizing YouTube video subtitles:
    - Write a list with 3 main key points of the following text in short sentences. Start list with -.
    - Try to cover every concept that are covered in the subtitles.
    - DO NOT use any formatting like Markdown, HTML etc.

    Be helpful without directly copying content.""")

def split_user_input(text):
    print("Вызвана функция split_user_input")
    # Split the input text into paragraphs
    paragraphs = text.split('\n')

    # Remove empty paragraphs and trim whitespace
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    return paragraphs

def scrape_text_from_url(url):
    """
    Scrape the content from the URL
    """
    print("Вызвана функция scrape_text_from_url")
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_formatting=True)
        if text is None:
            return []
        text_chunks = text.split("\n")
        article_content = [text for text in text_chunks if text]
        return article_content
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error: {e}\n{error_traceback}")


async def search_results(keywords):
    print("Вызвана функция search_results")
    print(keywords, ddg_region)
    results = await AsyncDDGS().text(keywords, region=ddg_region, safesearch='off', max_results=3)
    return results

def summarize(text_array):
    """
    Summarize the text using GPT API
    """
    print("Вызвана функция summarize")

    def create_chunks(paragraphs):
        print("Вызвана функция create_chunks")
        chunks = []
        chunk = ''
        for paragraph in paragraphs:
            if len(chunk) + len(paragraph) < chunk_size:
                chunk += paragraph + ' '
            else:
                chunks.append(chunk.strip())
                chunk = paragraph + ' '
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    try:
        text_chunks = create_chunks(text_array)
        text_chunks = [chunk for chunk in text_chunks if chunk] # Remove empty chunks

        # Call the GPT API in parallel to summarize the text chunks
        summaries = []
        system_messages = [
            {"role": "system", "content": f"{system_prompt}"}
        ]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(call_gpt_api, f"{chunk}", system_messages) for chunk in text_chunks]
            for future in tqdm(futures, total=len(text_chunks), desc="Summarizing"):
                summary = future.result()
                if summary:  # Check if summary is not empty
                    summaries.append(summary)

        if not summaries:
            return "no key points"

        if len(summaries) <= 5:
            summary = ' '.join(summaries)
            with tqdm(total=1, desc="Final summarization") as progress_bar:
                final_summary = call_gpt_api(f"{summary}", system_messages)
                progress_bar.update(1)
            return final_summary if final_summary else "no key points"
        else:
            return summarize(summaries)
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error: {e}\n{error_traceback}")
        return "no key points"

def get_youtube_video_info(youtube_url):
    print("Вызвана функция get_youtube_video_info")
    try:
        yt = YouTube(youtube_url)
        video_info = {
            "author": yt.author,
            "title": yt.title,
            "duration": yt.length,
            "publish_date": yt.publish_date.strftime("%Y-%m-%d"),            
            "description": yt.description,
            "url": youtube_url,
        }
        return video_info
    except Exception as e:
        print(f"Error getting video info: {e}")
        raise ValueError("Failed to get YouTube video info")

def extract_youtube_transcript(youtube_url):
    print("Вызвана функция extract_youtube_transcript")
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            video_id_match = re.search(r"(?<=v=)[^&]+|(?<=youtu.be/)[^?|\n]+", youtube_url)
            video_id = video_id_match.group(0) if video_id_match else None
            if video_id is None:
                raise ValueError("Invalid YouTube URL")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, proxies={"http": f"http://{zyte_api_key}:@api.zyte.com:8011/","https": f"http://{zyte_api_key}:@api.zyte.com:8011/",})
            transcript = transcript_list.find_transcript(['en', 'en-US', 'ja', 'ko', 'de', 'fr', 'ru', 'it', 'es', 'pl', 'uk', 'nl', 'zh-TW', 'zh-CN', 'zh-Hant', 'zh-Hans'])
            transcript_text = ' '.join([item['text'] for item in transcript.fetch()])
            return transcript_text
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Attempt {attempt + 1} failed. Error: {e}\n{error_traceback}")
            if attempt == max_attempts - 1:
                raise ValueError("Failed to extract YouTube transcript after 3 attempts")

def retrieve_yt_transcript_from_url(youtube_url):
    print("Вызвана функция retrieve_yt_transcript_from_url")
    output = extract_youtube_transcript(youtube_url)
    # Split output into an array based on the end of the sentence (like a dot),
    # but each chunk should be smaller than chunk_size
    output_sentences = output.split(' ')
    output_chunks = []
    current_chunk = ""

    for sentence in output_sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + ' '
        else:
            output_chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '

    if current_chunk:
        output_chunks.append(current_chunk.strip())
    return output_chunks

def call_gpt_api(prompt, additional_messages=[]):
    """
    Call GPT API
    """
    print("Вызвана функция call_gpt_api")
    try:
        response = litellm.completion(
        # response = openai.ChatCompletion.create(
            model=model,
            messages=additional_messages+[
                {"role": "user", "content": prompt}
            ],

        )
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error: {e}\n{error_traceback}")
        return ""

def handle_start(update, context):
    return handle('start', update, context)

def handle_help(update, context):
    return handle('help', update, context)

def handle_showall(update, context):
    return handle('showall', update, context)

def handle_summarize(update, context):
    return handle('summarize', update, context)

def handle_file(update, context):
    return handle('file', update, context)

def handle_button_click(update, context):
    return handle('button_click', update, context)

import traceback

async def handle(command, update, context):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    print("user_id=", user_id)

    if allowed_users:
        user_ids = allowed_users.split(',')
        if str(user_id) not in user_ids:
            print(user_id, "is not allowed.")
            await context.bot.send_message(chat_id=chat_id, text="You have no permission to use this bot.")
            return

    try:
        if command == 'start':
            await context.bot.send_message(chat_id=chat_id, text="I can summarize text, URLs, PDFs and YouTube video for you.")
        elif command == 'help':
            await context.bot.send_message(chat_id=chat_id, text="Report bugs here 👉 https://github.com/tpai/summary-gpt-bot/issues", disable_web_page_preview=True)
        elif command == 'showall':
            user_requests = await get_user_requests(user_id)
            if not user_requests:
                await context.bot.send_message(chat_id=chat_id, text="У вас пока нет сохраненных видео.")
                return
            
            video_info_list = []
            for content_hash in user_requests:
                cached_data = await get_cached_data(content_hash)
                print(f"Cached data for hash {content_hash}: {cached_data}")  # Добавлен вывод в лог
                if cached_data:
                    video_info = {
                        'title': cached_data.get('title', 'Название недоступно'),
                        'author': cached_data.get('author', 'Автор неизвестен'),
                        'url': cached_data.get('url', '#')
                    }
                    video_info_list.append(video_info)
            
            if video_info_list:
                message = "Список ваших сохраненных видео:\n\n"
                for info in video_info_list:
                    message += f"• <a href='{info['url']}'>{info['title']}</a> от {info['author']}\n"
            else:
                message = "У вас есть сохраненные видео, но не удалось получить информацию о них."
            
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML", disable_web_page_preview=True)
                    
        elif command == 'summarize':
            user_input = update.message.text
            print("user_input=", user_input)

            content_hash = get_hash(user_input)
            cached_data = await get_cached_data(content_hash)

            if cached_data and 'summary' in cached_data:
                await add_user_request(user_id, content_hash)
                response_text = f"<b>Key ideas: </b>\n{cached_data['summary']}\n"
                if youtube_pattern.match(user_input):
                    text = construct_video_info_text(cached_data) + response_text
                    response_text = text 
                await context.bot.send_message(chat_id=chat_id, text=response_text, reply_to_message_id=update.message.message_id, disable_web_page_preview = True, parse_mode="HTML")
                return

            try:
                text_array, video_info = process_user_input(user_input)
                print(text_array)

                if not text_array:
                    raise ValueError("No content found to summarize.")

                await context.bot.send_chat_action(chat_id=chat_id, action="TYPING")
                summary = summarize(text_array)
                
                await cache_data(content_hash, summary, video_info)
                await add_user_request(user_id, content_hash)

                response_text = f"<b>Key ideas: </b>\n{summary}\n"
                if youtube_pattern.match(user_input):
                    text = construct_video_info_text(video_info) + response_text
                    response_text = text

                await context.bot.send_message(chat_id=chat_id, text=response_text, reply_to_message_id=update.message.message_id, parse_mode="HTML",disable_web_page_preview = True)
            except ValueError as e:
                error_message = str(e)
                if error_message == "Please try again":
                    await context.bot.send_message(chat_id=chat_id, text="Please try again", reply_to_message_id=update.message.message_id)
                elif "Failed to extract YouTube transcript" in error_message:
                    await context.bot.send_message(chat_id=chat_id, text="Failed to extract YouTube transcript. Please try again later or with a different video.", reply_to_message_id=update.message.message_id)
                else:
                    await context.bot.send_message(chat_id=chat_id, text=f"An error occurred: {error_message}", reply_to_message_id=update.message.message_id)
        elif command == 'file':
            file_path = f"{update.message.document.file_unique_id}.pdf"
            print("file_path=", file_path)

            file = await context.bot.get_file(update.message.document)
            await file.download_to_drive(file_path)

            content_hash = get_hash(file_path)
            cached_summary = get_cached_summary(content_hash)

            if cached_summary:
                add_user_request(user_id, content_hash)
                await context.bot.send_message(chat_id=chat_id, text=cached_summary, reply_to_message_id=update.message.message_id, parse_mode="HTML",disable_web_page_preview = True)
                os.remove(file_path)
                return

            text_array = []
            reader = PdfReader(file_path)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                text_array.append(text)

            await context.bot.send_chat_action(chat_id=chat_id, action="TYPING")
            summary = summarize(text_array)

            cache_summary(content_hash, summary)
            add_user_request(user_id, content_hash)

            await context.bot.send_message(chat_id=chat_id, text=f"{summary}", reply_to_message_id=update.message.message_id, parse_mode="HTML", disable_web_page_preview = True)

            # remove temp file after sending message
            os.remove(file_path)
        elif command == 'button_click':
            original_message_text = update.callback_query.message.text
            await context.bot.send_chat_action(chat_id=chat_id, action="TYPING")

            if update.callback_query.data == "explore_similar":
                keywords = call_gpt_api(f"{original_message_text}\nBased on the content above, give me the top 5 important keywords with commas.", [
                    {"role": "system", "content": f"You will print keywords only."}
                ])

                tasks = [search_results(keywords)]
                results = await asyncio.gather(*tasks)
                print(results)

                links = ''
                for r in results[0]:
                    links += f"{r['title']}\n{r['href']}\n"

                await context.bot.send_message(chat_id=chat_id, text=links, reply_to_message_id=update.callback_query.message.message_id, disable_web_page_preview=True)

            if update.callback_query.data == "why_it_matters":
                result = call_gpt_api(f"{original_message_text}\nBased on the content above, tell me why it matters as an expert.", [
                    {"role": "system", "content": f"You will show the result in English."}
                ])
                await context.bot.send_message(chat_id=chat_id, text=result, reply_to_message_id=update.callback_query.message.message_id, disable_web_page_preview = True)
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error: {e}\n{error_traceback}")


def process_user_input(user_input):
    print("Вызвана функция process_user_input")
    global youtube_pattern
    url_pattern = re.compile(r"https?://")

    if youtube_pattern.match(user_input):
        try:
            text_array = retrieve_yt_transcript_from_url(user_input)
            video_info = get_youtube_video_info(user_input)
            return text_array, video_info
        except ValueError as e:
            print(f"Error processing YouTube input: {e}")
            raise ValueError("Please try again")
    elif url_pattern.match(user_input):
        text_array = scrape_text_from_url(user_input)
    else:
        text_array = split_user_input(user_input)

    return text_array, None

def get_inline_keyboard_buttons():
    print("Вызвана функция get_inline_keyboard_buttons")
    keyboard = [
        [InlineKeyboardButton("Explore Similar", callback_data="explore_similar")],
        [InlineKeyboardButton("Why It Matters", callback_data="why_it_matters")],
    ]
    return InlineKeyboardMarkup(keyboard)

def get_hash(content):
    print("Вызвана функция get_hash")
    return hashlib.md5(content.encode()).hexdigest()

async def get_cached_summary(content_hash):
    print("Вызвана функция get_cached_summary")
    cached = await redis_client.hget(f'study_buddy_youtube_info:{content_hash}', 'summary')
    return cached.decode('utf-8') if cached else None

async def cache_summary(content_hash, summary):
    print("Вызвана функция cache_summary")
    await redis_client.hset(f'study_buddy_youtube_info:{content_hash}', 'summary', summary)

async def add_user_request(user_id, content_hash):
    print("Вызвана функция add_user_request")
    await redis_client.sadd(f'study_buddy_users:{user_id}:requests', content_hash)

async def get_user_requests(user_id):
    print("Вызвана функция get_user_requests")
    requests = await redis_client.smembers(f'study_buddy_users:{user_id}:requests')
    return [request.decode('utf-8') if isinstance(request, bytes) else request for request in requests]


async def get_cached_data(content_hash):
    print("Вызвана функция get_cached_data")
    fields = ['author', 'title', 'duration', 'publish_date', 'description', 'summary', 'url']
    values = await redis_client.hmget(f'study_buddy_youtube_info:{content_hash}', fields)
    result = {}
    for field, value in zip(fields, values):
        if value:
            result[field] = value.decode('utf-8') if isinstance(value, bytes) else value
    return result

def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def construct_video_info_text(video_info):
    response_text = ""
    if video_info:
        if 'author' in video_info:
            response_text += f"<b>Author: </b>{video_info['author']}\n"
        if 'title' in video_info:
            response_text += f"<b>Title: </b>{video_info['title']}\n"
        if 'duration' in video_info:
            formatted_duration = format_duration(video_info['duration'])
            response_text += f"<b>Duration: </b>{formatted_duration}\n"
        if 'publish_date' in video_info:
            response_text += f"<b>Publish date: </b>{video_info['publish_date']}\n"
        if 'description' in video_info:
            response_text += f"<b>Description: </b><blockquote expandable> {video_info['description']}</blockquote>\n"
    return response_text

async def cache_data(content_hash, summary, video_info=None):
    print("Вызвана функция cache_data")
    data = {'summary': summary}
    if video_info:
        data.update({k: str(v) for k, v in video_info.items()})
    await redis_client.hset(f'study_buddy_youtube_info:{content_hash}', mapping=data)

def main():
    try:
        init_redis()
        application = ApplicationBuilder().token(telegram_token).build()
        start_handler = CommandHandler('start', handle_start)
        help_handler = CommandHandler('help', handle_help)
        showall_handler = CommandHandler('showall', handle_showall)
        summarize_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_summarize)
        file_handler = MessageHandler(filters.Document.PDF, handle_file)
        button_click_handler = CallbackQueryHandler(handle_button_click)
        application.add_handler(file_handler)
        application.add_handler(start_handler)
        application.add_handler(help_handler)
        application.add_handler(showall_handler)
        application.add_handler(summarize_handler)
        application.add_handler(button_click_handler)
        application.run_polling()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    #asyncio.run()
    main()

