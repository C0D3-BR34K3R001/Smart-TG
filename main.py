# do this in new terminal:  pip install aiohttp psutil

"""
Smart Telegram Bot with Enhanced AI, Owner Features, and OpenRouter Integration
Requires: pip install python-telegram-bot python-dotenv requests beautifulsoup4 yt-dlp aiohttp
"""

import os
import logging
import aiohttp
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, filters, ContextTypes
)
import json
from pathlib import Path
import platform
import psutil

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_VERSION = "2.0.0"
ADMIN_IDS = [7528793664]  # Add your admin user IDs here
OWNER_ID = 7528793664  # Replace with your user ID
USERS_DB = "users.json"

# OpenRouter configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
AI_MODELS = {
    "gpt-4": "GPT-4",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "gemini-pro": "Gemini Pro",
    "llama-2-70b": "Llama 2 70B",
    "mistral-7b": "Mistral 7B"
}

# States for conversations
DOWNLOAD_VIDEO, DOWNLOAD_MUSIC, DOWNLOAD_IMAGE = range(3)
AI_CHAT, AI_TRANSLATE, AI_SUMMARIZE, AI_VISION, AI_CODE = range(3, 8)

# ==================== DATABASE FUNCTIONS ====================

def load_users():
    if Path(USERS_DB).exists():
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=2)

def add_user(user_id, username=""):
    users = load_users()
    if str(user_id) not in users:
        users[str(user_id)] = {
            "username": username,
            "joined": datetime.now().isoformat(),
            "commands_used": 0,
            "is_banned": False,
            "ai_model": "gpt-4",
            "message_history": []
        }
        save_users(users)

def update_user_stats(user_id):
    users = load_users()
    if str(user_id) in users:
        users[str[user_id]]["commands_used"] += 1
        save_users(users)

# ==================== AI FUNCTIONS ====================

async def openrouter_chat_completion(messages, model="gpt-4", max_tokens=1000):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://telegram-bot.com",
        "X-Title": "Smart Telegram Bot"
    }

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    return f"❌ AI Error: {response.status} - {error_text}"
    except:
        return "❌ AI request failed. Try again later."

async def ai_chat_response(user_id, message, context=None):
    users = load_users()
    user_data = users.get(str(user_id), {})
    model = user_data.get("ai_model", "gpt-4")
    message_history = user_data.get("message_history", [])

    message_history.append({"role": "user", "content": message})
    message_history = message_history[-10:]

    messages = [{"role": "system", "content": "You are a helpful AI."}] + message_history
    response = await openrouter_chat_completion(messages, model)

    if not response.startswith("❌"):
        message_history.append({"role": "assistant", "content": response})
        users[str(user_id)]["message_history"] = message_history
        save_users(users)

    return response

async def ai_translate(text, target_language):
    messages = [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": f"Translate to {target_language}: {text}"}
    ]
    return await openrouter_chat_completion(messages)

async def ai_summarize(text):
    messages = [
        {"role": "system", "content": "You summarize text."},
        {"role": "user", "content": f"Summarize this:\n{text}"}
    ]
    return await openrouter_chat_completion(messages)

async def ai_code_assist(code, language=None):
    messages = [
        {"role": "system", "content": "You assist with coding."},
        {"role": "user", "content": f"Help with this ({language}):\n{code}"}
    ]
    return await openrouter_chat_completion(messages)

async def ai_vision_analysis(image_url, question=None):
    messages = [
        {"role": "system", "content": "You analyze images."},
        {"role": "user", "content": f"Analyze image: {image_url}. Question: {question}"}
    ]
    return await openrouter_chat_completion(messages)

# ==================== REQUIRED BOT COMMANDS ====================

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
<b>🤖 Smart Bot Help</b>

Available commands:
/start - Open main menu
/help - Show this help message
/cancel - Cancel current action

Use the menu buttons to access:
• AI Chat
• Translate
• Summarize
• Vision AI
• Code Assistant
• Downloads
• Fun Menu
• Stats, Admin & Owner Tools
"""
    await update.message.reply_html(text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["awaiting_input"] = None
    await update.message.reply_text("❌ Operation cancelled.")

# ==================== PLACEHOLDER FUNCTIONS (MISSING IN YOUR FILE) ====================
# These prevent crashes due to undefined references.

async def user_menu(query, context):
    await query.edit_message_text("👤 User Menu (placeholder)")

async def download_menu(query, context):
    context.user_data["awaiting_input"] = "download_video"
    await query.edit_message_text("📥 Send the video URL to download:")

async def fun_menu(query, context):
    await query.edit_message_text("🎮 Fun Menu")

async def admin_menu(query, context):
    await query.edit_message_text("🛡️ Admin Panel")

async def show_stats(query, context):
    users = load_users()
    txt = f"""
<b>📊 Bot Statistics</b>

Users: {len(users)}
Admins: {len(ADMIN_IDS)}
Bot Version: {BOT_VERSION}
"""
    await query.edit_message_text(txt, parse_mode="HTML")

async def profile_command(query, context):
    await query.edit_message_text("👤 Profile (coming soon)")

async def chat_ai_handler(query, context):
    context.user_data["awaiting_input"] = "chat"
    await query.edit_message_text("💬 Send a message to chat with AI")

async def translate_handler(query, context):
    context.user_data["awaiting_input"] = "translate"
    await query.edit_message_text("🌐 Send: text | language")

async def summarize_handler(query, context):
    context.user_data["awaiting_input"] = "summarize"
    await query.edit_message_text("📝 Send text to summarize")

async def joke_command(query, context):
    await query.edit_message_text("😂 Funny joke coming soon!")

async def quote_command(query, context):
    await query.edit_message_text("💬 Inspirational quote coming soon!")

async def dice_command(query, context):
    await query.edit_message_text("🎲 You rolled a dice!")

async def magic8_command(query, context):
    await query.edit_message_text("🎱 Magic 8 ball says... yes!")

# ==================== OWNER MENU FIXES ====================

async def analytics(query, context):
    await query.edit_message_text("📈 Analytics coming soon")

async def bot_settings(query, context):
    await query.edit_message_text("⚙️ Bot settings coming soon")

async def maintenance(query, context):
    await query.edit_message_text("🧹 Maintenance tools coming soon")

async def user_management(query, context):
    await query.edit_message_text("📋 User management coming soon")

async def clear_cache(query, context):
    await query.edit_message_text("🧹 Cache cleared!")

async def restart_bot(query, context):
    await query.edit_message_text("🔄 Restarting bot... (not implemented)")

async def stop_bot(query, context):
    await query.edit_message_text("⏹ Bot stopping... (not implemented)")

async def owner_broadcast(query, context):
    context.user_data["awaiting_input"] = "broadcast"
    await query.edit_message_text("📢 Send broadcast message text")

# ==================== MAIN MENU ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    add_user(user.id, user.username)

    keyboard = [
        [InlineKeyboardButton("👤 User Menu", callback_data="user_menu"),
         InlineKeyboardButton("🤖 AI Commands", callback_data="ai_menu")],
        [InlineKeyboardButton("📥 Download Menu", callback_data="download_menu"),
         InlineKeyboardButton("🎮 Fun Commands", callback_data="fun_menu")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats")]
    ]

    if user.id in ADMIN_IDS:
        keyboard.append([InlineKeyboardButton("🛡️ Admin Panel", callback_data="admin_menu")])
    if user.id == OWNER_ID:
        keyboard.append([InlineKeyboardButton("👑 Owner Panel", callback_data="owner_menu")])

    reply_markup = InlineKeyboardMarkup(keyboard)

    text = f"""
<b>🤖 Welcome to Smart Bot v{BOT_VERSION}</b>

Hello {user.first_name} 👋

Choose an option below.
"""
    await update.message.reply_html(text, reply_markup=reply_markup)

# ==================== AI MENU ====================

async def ai_menu(query, context):
    keyboard = [
        [InlineKeyboardButton("💬 AI Chat", callback_data="chat_ai"),
         InlineKeyboardButton("🌐 Translate", callback_data="translate_ai")],
        [InlineKeyboardButton("📝 Summarize", callback_data="summarize_ai"),
         InlineKeyboardButton("👁️ Vision", callback_data="vision_ai")],
        [InlineKeyboardButton("💻 Code Assistant", callback_data="code_ai"),
         InlineKeyboardButton("🤖 Change AI Model", callback_data="change_model")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "<b>🤖 AI Tools Menu</b>",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )

async def change_model_handler(query, context):
    model_buttons = []
    users = load_users()
    current = users.get(str(query.from_user.id), {}).get("ai_model", "gpt-4")

    for mid, name in AI_MODELS.items():
        mark = " ✅" if mid == current else ""
        model_buttons.append([InlineKeyboardButton(f"{name}{mark}", callback_data=f"model_{mid}")])

    model_buttons.append([InlineKeyboardButton("🔙 Back", callback_data="ai_menu")])

    await query.edit_message_text(
        "<b>Select AI Model</b>",
        reply_markup=InlineKeyboardMarkup(model_buttons),
        parse_mode="HTML"
    )

async def model_selection_handler(query, context):
    model = query.data.replace("model_", "")
    users = load_users()
    if str(query.from_user.id) in users:
        users[str(query.from_user.id)]["ai_model"] = model
        save_users(users)
    await query.answer("Model updated!")
    await ai_menu(query, context)

async def vision_ai_handler(query, context):
    context.user_data["awaiting_input"] = "vision_ai"
    await query.edit_message_text("👁 Send: image_url | question")

async def code_ai_handler(query, context):
    context.user_data["awaiting_input"] = "code_ai"
    await query.edit_message_text("💻 Send: code | language")

# ==================== HANDLE USER MESSAGES ====================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text

    users = load_users()
    user_data = users.get(str(user.id), {})

    if user_data.get("is_banned"):
        await update.message.reply_text("❌ You are banned.")
        return

    awaiting = context.user_data.get("awaiting_input")
    update_user_stats(user.id)

    if awaiting == "chat":
        await update.message.reply_chat_action("typing")
        response = await ai_chat_response(user.id, text)
        await update.message.reply_text(response)

    elif awaiting == "translate":
        if "|" in text:
            t, lang = map(str.strip, text.split("|", 1))
            response = await ai_translate(t, lang)
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("Format: text | language")

    elif awaiting == "summarize":
        response = await ai_summarize(text)
        await update.message.reply_text(response)

    elif awaiting == "vision_ai":
        if "|" in text:
            url, question = map(str.strip, text.split("|", 1))
            response = await ai_vision_analysis(url, question)
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("Format: image_url | question")

    elif awaiting == "code_ai":
        if "|" in text:
            code, lang = map(str.strip, text.split("|", 1))
            response = await ai_code_assist(code, lang)
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("Format: code | language")

    context.user_data["awaiting_input"] = None

# ==================== CALLBACK HANDLER ====================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    routes = {
        "user_menu": user_menu,
        "ai_menu": ai_menu,
        "download_menu": download_menu,
        "fun_menu": fun_menu,
        "admin_menu": admin_menu,
        "owner_menu": owner_menu,
        "stats": show_stats,
        "profile": profile_command,
        "chat_ai": chat_ai_handler,
        "translate_ai": translate_handler,
        "summarize_ai": summarize_handler,
        "vision_ai": vision_ai_handler,
        "code_ai": code_ai_handler,
        "change_model": change_model_handler,
    }

    if query.data.startswith("model_"):
        await model_selection_handler(query, context)
        return

    if query.data in routes:
        await routes[query.data](query, context)

# ==================== MAIN APPLICATION ====================

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("cancel", cancel))

    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Smart Bot is starting...")
    print(f"🤖 Bot Version: {BOT_VERSION}")
    print(f"👑 Owner ID: {OWNER_ID}")
    print(f"🛡 Admin Count: {len(ADMIN_IDS)}")

    app.run_polling()

if __name__ == "__main__":
    main()
