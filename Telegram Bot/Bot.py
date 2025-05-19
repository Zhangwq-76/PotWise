#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import asyncio
import os
import sys
import base64
import tempfile
from io import BytesIO

import nest_asyncio
nest_asyncio.apply()

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# 添加 YOLO 模型所在目录到 sys.path
YOLO_DIR = r"C:\Users\12821\Desktop\PotWise\YOLO"
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)
import model  # 导入我们写好的 model.py 模块

# 导入标签映射文件，标签映射文件位于 YOLO 文件夹下
from label_mapping import LABEL_MAPPING

# 添加 Deepseek 模块所在目录到 sys.path，并导入 get_hotpot_recommendation 方法
DEEPSEEK_DIR = r"C:\Users\12821\Desktop\PotWise\Deepseek"
if DEEPSEEK_DIR not in sys.path:
    sys.path.append(DEEPSEEK_DIR)
from UseDeepseek import get_hotpot_recommendation  # 导入火锅推荐方法

from GetSectence import generate_lover_prompt_sentence  # 从 get_sectence.py 中导入函数

FISHSPEECH_DIR = r"C:\Users\12821\Desktop\PotWise\Telegram Bot"
if FISHSPEECH_DIR not in sys.path:
    sys.path.append(FISHSPEECH_DIR)

from FishSpeech import generate_speech_from_text

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# /start 命令处理函数
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! I'm Potwise, your Hotpot Ingredient Assistant. Send me a photo of your hotpot, and I’ll recognize the ingredients for you in seconds!")

# /help 命令处理函数
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Just send a hotpot image, and I’ll detect the ingredients using our custom-trained YOLO model. You’ll receive a labeled image along with cooking suggestions.")

# /markdown 命令处理函数，回复一段测试的 Markdown 格式文本
async def markdown_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    test_markdown = (
        "*This is bold text*\n"
        "_This is italic text_\n"
        "`This is inline code`\n"
        "[This is a link](https://example.com)"
    )
    await update.message.reply_text(test_markdown, parse_mode='Markdown')

# 处理用户发送的图片
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message.photo:
        await update.message.reply_text("No image detected. Please send a photo.")
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        temp_path = tf.name

    await file.download_to_drive(custom_path=temp_path)
    logger.info(f"Image downloaded to {temp_path}")

    try:
        result = await asyncio.to_thread(model.predict, temp_path)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        await update.message.reply_text("Image processing failed. Please try again later.")
        os.remove(temp_path)
        return

    detections = result.get("detections", [])
    annotated_image_base64 = result.get("annotated_image")

    caption = "Detection Results:\n"
    if not detections:
        caption += "No objects detected."
    else:
        for det in detections:
            caption += f"Class: {det['class']}, Confidence: {det['confidence']:.2f}, Bounding Box: {det['bbox']}\n"

    try:
        annotated_bytes = base64.b64decode(annotated_image_base64)
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        await update.message.reply_text("Error occurred while processing the image.")
        os.remove(temp_path)
        return

    image_stream = BytesIO(annotated_bytes)
    image_stream.name = "result.jpg"

    # await update.message.reply_photo(photo=image_stream, caption=caption)
    await update.message.reply_photo(photo=image_stream)

    if detections:
        ingredients_dict = {}
        for idx, det in enumerate(detections, 1):
            mapped_label = LABEL_MAPPING.get(det['class'], "Unknown Label")
            ingredients_dict[str(idx)] = mapped_label

        print("Generated ingredients dict:", ingredients_dict)

        hotpot_reply = get_hotpot_recommendation(ingredients_dict)
        await update.message.reply_text(hotpot_reply, parse_mode='Markdown')

    try:
        lover_prompt = generate_lover_prompt_sentence(ingredients_dict)
        context.user_data['lover_prompt'] = lover_prompt

        audio_path = generate_speech_from_text(lover_prompt)
        if os.path.exists(audio_path):
            with open(audio_path, 'rb') as audio_file:
                await update.message.reply_voice(audio_file, caption=lover_prompt)
        else:
            await update.message.reply_text("Failed to generate the voice file.")

    except Exception as e:
        logger.error(f"Failed to generate lover prompt or voice: {e}")
        lover_prompt = "(Voice prompt generation failed)"
        await update.message.reply_text("Failed to generate voice prompt. Please try again later.")

    os.remove(temp_path)

async def main():
    TOKEN = "Your own bot token"
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("markdown", markdown_test))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    await application.run_polling(close_loop=False)

if __name__ == '__main__':
    asyncio.run(main())
