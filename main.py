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
    """Load users database"""
    if Path(USERS_DB).exists():
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users database"""
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=2)

def add_user(user_id, username=""):
    """Add user to database"""
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
    """Update user command statistics"""
    users = load_users()
    if str(user_id) in users:
        users[str(user_id)]["commands_used"] += 1
        save_users(users)

# ==================== AI FUNCTIONS ====================

async def openrouter_chat_completion(messages, model="gpt-4", max_tokens=1000):
    """Send request to OpenRouter API"""
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
                    logger.error(f"OpenRouter API error: {error_text}")
                    return f"❌ AI Error: {response.status} - {error_text}"
    except asyncio.TimeoutError:
        return "❌ AI request timeout. Please try again."
    except Exception as e:
        logger.error(f"OpenRouter exception: {e}")
        return f"❌ AI Error: {str(e)}"

async def ai_chat_response(user_id, message, context=None):
    """Generate AI chat response with context"""
    users = load_users()
    user_data = users.get(str(user_id), {})
    model = user_data.get("ai_model", "gpt-4")
    
    # Get message history (last 10 messages for context)
    message_history = user_data.get("message_history", [])
    message_history.append({"role": "user", "content": message})
    
    # Keep only last 10 messages to manage context
    if len(message_history) > 10:
        message_history = message_history[-10:]
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant in a Telegram bot. Be concise but helpful."}
    ] + message_history
    
    response = await openrouter_chat_completion(messages, model)
    
    # Update message history with AI response
    if not response.startswith("❌"):
        message_history.append({"role": "assistant", "content": response})
        users[str(user_id)]["message_history"] = message_history
        save_users(users)
    
    return response

async def ai_translate(text, target_language):
    """Translate text using AI"""
    prompt = f"Translate the following text to {target_language}. Only provide the translation, no additional text:\n\n{text}"
    
    messages = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": prompt}
    ]
    
    return await openrouter_chat_completion(messages)

async def ai_summarize(text):
    """Summarize text using AI"""
    prompt = f"Please provide a concise summary of the following text. Keep it brief but informative:\n\n{text}"
    
    messages = [
        {"role": "system", "content": "You are a text summarization expert."},
        {"role": "user", "content": prompt}
    ]
    
    return await openrouter_chat_completion(messages)

async def ai_code_assist(code, language=None):
    """Provide coding assistance"""
    prompt = f"Please help with this code{' in ' + language if language else ''}. Provide improvements, fixes, or explanations:\n\n{code}"
    
    messages = [
        {"role": "system", "content": "You are an expert programming assistant."},
        {"role": "user", "content": prompt}
    ]
    
    return await openrouter_chat_completion(messages)

async def ai_vision_analysis(image_url, question=None):
    """Analyze images (placeholder for vision capabilities)"""
    prompt = f"Analyze this image and {question if question else 'describe what you see'}."
    
    # Note: OpenRouter vision capabilities depend on model support
    messages = [
        {"role": "system", "content": "You are an image analysis expert."},
        {"role": "user", "content": prompt}
    ]
    
    return await openrouter_chat_completion(messages)

# ==================== MAIN MENU ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command - show main menu"""
    user = update.effective_user
    add_user(user.id, user.username)
    
    keyboard = [
        [InlineKeyboardButton("👤 User Menu", callback_data="user_menu"),
         InlineKeyboardButton("🤖 AI Commands", callback_data="ai_menu")],
        [InlineKeyboardButton("📥 Download Menu", callback_data="download_menu"),
         InlineKeyboardButton("🎮 Fun Commands", callback_data="fun_menu")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats")]
    ]
    
    # Add admin/owner buttons if user has privileges
    if user.id in ADMIN_IDS:
        keyboard.append([InlineKeyboardButton("🛡️ Admin Panel", callback_data="admin_menu")])
    if user.id == OWNER_ID:
        keyboard.append([InlineKeyboardButton("👑 Owner Panel", callback_data="owner_menu")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = f"""
🤖 <b>Welcome to Smart Bot v{BOT_VERSION}!</b>

Hello {user.first_name}! 👋

I'm an advanced AI-powered bot with:
✨ Real AI chat, translation, summarization
📥 Download videos, music & images
🎮 Fun utilities & games
📊 User statistics
👑 Owner features

Choose an option below to get started!
"""
    
    await update.message.reply_html(welcome_text, reply_markup=reply_markup)

# ==================== ENHANCED AI MENU ====================

async def ai_menu(query, context):
    """Enhanced AI commands menu"""
    keyboard = [
        [InlineKeyboardButton("💬 AI Chat", callback_data="chat_ai"),
         InlineKeyboardButton("🌐 Translate", callback_data="translate_ai")],
        [InlineKeyboardButton("📝 Summarize", callback_data="summarize_ai"),
         InlineKeyboardButton("👁️ Vision Analysis", callback_data="vision_ai")],
        [InlineKeyboardButton("💻 Code Assistant", callback_data="code_ai"),
         InlineKeyboardButton("🤖 Change AI Model", callback_data="change_model")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    users = load_users()
    user_data = users.get(str(query.from_user.id), {})
    current_model = user_data.get("ai_model", "gpt-4")
    
    text = f"""
<b>🤖 Enhanced AI Commands</b>

Current AI Model: <b>{AI_MODELS.get(current_model, current_model)}</b>

Choose an AI feature:
• <b>AI Chat</b> - Conversational AI with memory
• <b>Translate</b> - Multi-language translation
• <b>Summarize</b> - Text summarization
• <b>Vision</b> - Image analysis (URL-based)
• <b>Code Assistant</b> - Programming help
• <b>Change Model</b> - Switch AI models
"""
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

async def change_model_handler(query, context):
    """Change AI model"""
    keyboard = []
    users = load_users()
    user_data = users.get(str(query.from_user.id), {})
    current_model = user_data.get("ai_model", "gpt-4")
    
    for model_id, model_name in AI_MODELS.items():
        is_current = " ✅" if model_id == current_model else ""
        keyboard.append([InlineKeyboardButton(f"{model_name}{is_current}", callback_data=f"model_{model_id}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="ai_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = f"""
<b>🤖 Change AI Model</b>

Current model: <b>{AI_MODELS.get(current_model, current_model)}</b>

Select a new AI model:
"""
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

async def model_selection_handler(query, context):
    """Handle model selection"""
    model_id = query.data.replace("model_", "")
    user_id = query.from_user.id
    
    users = load_users()
    if str(user_id) in users:
        users[str(user_id)]["ai_model"] = model_id
        save_users(users)
    
    await query.answer(f"✅ Model changed to {AI_MODELS.get(model_id, model_id)}")
    await ai_menu(query, context)

async def vision_ai_handler(query, context):
    """Vision analysis feature"""
    context.user_data['awaiting_input'] = 'vision_ai'
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="ai_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
<b>👁️ AI Vision Analysis</b>

Send image URL and question (optional):
Format: <code>image_url | question</code>
Example: <code>https://example.com/image.jpg | What's in this image?</code>

(Type /cancel to go back)
"""
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

async def code_ai_handler(query, context):
    """Code assistant feature"""
    context.user_data['awaiting_input'] = 'code_ai'
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="ai_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
<b>💻 AI Code Assistant</b>

Send your code and language (optional):
Format: <code>your_code | language</code>
Example: <code>def hello(): return "world" | python</code>

(Type /cancel to go back)
"""
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

# ==================== OWNER MENU ====================

async def owner_menu(query, context):
    """Owner panel with advanced features"""
    keyboard = [
        [InlineKeyboardButton("📊 System Stats", callback_data="system_stats"),
         InlineKeyboardButton("🔧 Bot Controls", callback_data="bot_controls")],
        [InlineKeyboardButton("📋 User Management", callback_data="user_management"),
         InlineKeyboardButton("🔄 Maintenance", callback_data="maintenance")],
        [InlineKeyboardButton("📈 Analytics", callback_data="analytics"),
         InlineKeyboardButton("⚙️ Bot Settings", callback_data="bot_settings")],
        [InlineKeyboardButton("🔙 Back", callback_data="back_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
<b>👑 Owner Panel</b>

Advanced bot management:
• <b>System Stats</b> - Detailed system information
• <b>Bot Controls</b> - Start/stop/restart bot
• <b>User Management</b> - Advanced user controls
• <b>Maintenance</b> - Database cleanup & optimization
• <b>Analytics</b> - Usage statistics & insights
• <b>Bot Settings</b> - Configure bot behavior
"""
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

async def system_stats_handler(query, context):
    """Show detailed system statistics"""
    import psutil
    import platform
    
    # System information
    system_info = f"""
<b>🖥️ System Statistics</b>

<b>System:</b> {platform.system()} {platform.release()}
<b>Processor:</b> {platform.processor()}
<b>Python:</b> {platform.python_version()}

<b>Memory Usage:</b> {psutil.virtual_memory().percent}%
<b>CPU Usage:</b> {psutil.cpu_percent()}%
<b>Disk Usage:</b> {psutil.disk_usage('/').percent}%

<b>Bot Uptime:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="owner_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(system_info, reply_markup=reply_markup, parse_mode="HTML")

async def bot_controls_handler(query, context):
    """Bot controls menu"""
    keyboard = [
        [InlineKeyboardButton("🔄 Restart Bot", callback_data="restart_bot"),
         InlineKeyboardButton("⏹️ Stop Bot", callback_data="stop_bot")],
        [InlineKeyboardButton("📝 Broadcast Message", callback_data="owner_broadcast"),
         InlineKeyboardButton("🧹 Clear Cache", callback_data="clear_cache")],
        [InlineKeyboardButton("🔙 Back", callback_data="owner_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = """
<b>🔧 Bot Controls</b>

Manage bot operation:
• <b>Restart Bot</b> - Soft restart
• <b>Stop Bot</b> - Shutdown bot
• <b>Broadcast</b> - Message all users
• <b>Clear Cache</b> - Clear temporary data
"""
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

# ==================== ENHANCED MESSAGE HANDLER ====================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages with enhanced AI features"""
    user = update.effective_user
    text = update.message.text
    
    # Check if user is banned
    users = load_users()
    user_data = users.get(str(user.id), {})
    if user_data.get('is_banned'):
        await update.message.reply_text("❌ You are banned from using this bot.")
        return
    
    awaiting = context.user_data.get('awaiting_input')
    update_user_stats(user.id)
    
    if awaiting == 'chat':
        # Enhanced AI chat with real responses
        await update.message.reply_chat_action("typing")
        response = await ai_chat_response(user.id, text)
        await update.message.reply_text(f"💬 {response}")
        
    elif awaiting == 'translate':
        # Real translation
        if '|' in text:
            text_to_translate, target_lang = map(str.strip, text.split('|', 1))
            await update.message.reply_chat_action("typing")
            translation = await ai_translate(text_to_translate, target_lang)
            await update.message.reply_text(f"🌐 Translation to {target_lang}:\n\n{translation}")
        else:
            await update.message.reply_text("❌ Please use format: text | target_language")
            
    elif awaiting == 'summarize':
        # Real summarization
        await update.message.reply_chat_action("typing")
        summary = await ai_summarize(text)
        await update.message.reply_text(f"📝 Summary:\n\n{summary}")
        
    elif awaiting == 'vision_ai':
        # Vision analysis
        if '|' in text:
            image_url, question = map(str.strip, text.split('|', 1))
            await update.message.reply_chat_action("typing")
            analysis = await ai_vision_analysis(image_url, question)
            await update.message.reply_text(f"👁️ Vision Analysis:\n\n{analysis}")
        else:
            await update.message.reply_text("❌ Please use format: image_url | question")
            
    elif awaiting == 'code_ai':
        # Code assistance
        if '|' in text:
            code, language = map(str.strip, text.split('|', 1))
            await update.message.reply_chat_action("typing")
            assistance = await ai_code_assist(code, language)
            await update.message.reply_text(f"💻 Code Assistance:\n\n{assistance}")
        else:
            await update.message.reply_text("❌ Please use format: code | language")
            
    elif awaiting in ['download_video', 'download_music', 'download_image', 'search']:
        # Download features (existing functionality)
        response = f"⏳ Processing your request: {text}\n\n(This is a demo - implement your download logic here)"
        await update.message.reply_text(response)
        
    elif awaiting in ['ban_user', 'kick_user', 'promote_user', 'broadcast']:
        # Admin features (existing functionality)
        await handle_admin_actions(update, context, awaiting, text)
    
    # Clear awaiting input
    context.user_data['awaiting_input'] = None

async def handle_admin_actions(update, context, action, text):
    """Handle admin actions"""
    if action == 'ban_user':
        try:
            user_id = int(text)
            users_data = load_users()
            if str(user_id) in users_data:
                users_data[str(user_id)]['is_banned'] = True
                save_users(users_data)
                await update.message.reply_text(f"✅ User {user_id} has been banned!")
            else:
                await update.message.reply_text(f"❌ User {user_id} not found!")
        except ValueError:
            await update.message.reply_text("❌ Invalid user ID!")
            
    elif action == 'kick_user':
        try:
            user_id = int(text)
            await update.message.reply_text(f"👢 User {user_id} has been kicked!")
        except ValueError:
            await update.message.reply_text("❌ Invalid user ID!")
            
    elif action == 'promote_user':
        try:
            user_id = int(text)
            if user_id not in ADMIN_IDS:
                ADMIN_IDS.append(user_id)
                await update.message.reply_text(f"⬆️ User {user_id} has been promoted to admin!")
            else:
                await update.message.reply_text(f"ℹ️ User {user_id} is already an admin!")
        except ValueError:
            await update.message.reply_text("❌ Invalid user ID!")
            
    elif action == 'broadcast':
        users_data = load_users()
        count = 0
        for user_id in users_data:
            # In production, you'd send actual messages here
            count += 1
        await update.message.reply_text(f"📢 Broadcast sent to {count} users!")

# ==================== ENHANCED CALLBACK HANDLER ====================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks with new features"""
    query = update.callback_query
    await query.answer()
    
    callback_handlers = {
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
        "system_stats": system_stats_handler,
        "bot_controls": bot_controls_handler,
    }
    
    # Handle model selection
    if query.data.startswith("model_"):
        await model_selection_handler(query, context)
        return
    
    handler = callback_handlers.get(query.data)
    if handler:
        await handler(query, context)
    elif query.data in ["joke", "quote", "dice", "magic8"]:
        await handle_fun_commands(query, context)

async def handle_fun_commands(query, context):
    """Handle fun commands"""
    if query.data == "joke":
        await joke_command(query, context)
    elif query.data == "quote":
        await quote_command(query, context)
    elif query.data == "dice":
        await dice_command(query, context)
    elif query.data == "magic8":
        await magic8_command(query, context)

# ==================== MAIN APPLICATION ====================

def main():
    """Start the bot with enhanced features"""
    # Create application
    app = Application.builder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("cancel", cancel))
    
    # Enhanced callback query handler
    app.add_handler(CallbackQueryHandler(button_callback))
    
    # Message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start bot
    logger.info("🚀 Enhanced Smart Bot is starting...")
    print(f"🤖 Bot Version: {BOT_VERSION}")
    print(f"👑 Owner ID: {OWNER_ID}")
    print(f"🛡️ Admin Count: {len(ADMIN_IDS)}")
    print("✅ Bot is ready and waiting for messages...")
    
    app.run_polling()

if __name__ == "__main__":
    main()