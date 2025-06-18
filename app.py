import os
import sqlite3
import pickle
import platform
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session, flash
from moviepy.editor import VideoFileClip, AudioFileClip
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
from datetime import datetime
from functools import wraps
import PyPDF2
import docx
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-change-this-in-production'

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384

def get_ffmpeg_executable():
    if platform.system() == "Windows":
        win_path = r"C:\\ffmpeg\\bin\\ffmpeg.exe"
        if os.path.exists(win_path):
            return win_path
    elif platform.system() == "Linux":
        if os.path.exists("/usr/bin/ffmpeg"):
            return "/usr/bin/ffmpeg"
        elif os.path.exists("/usr/local/bin/ffmpeg"):
            return "/usr/local/bin/ffmpeg"
    else:
        return "ffmpeg"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def init_db():
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            transcript TEXT NOT NULL,
            summary TEXT NOT NULL,
            sentiment_polarity REAL,
            sentiment_subjectivity REAL,
            sentiment_label TEXT,
            emotion_analysis TEXT,
            keywords TEXT,
            combined_text TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faiss_index_meta (
            id INTEGER PRIMARY KEY,
            index_size INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def load_faiss_index():
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT embedding, combined_text, summary FROM video_analysis ORDER BY id')
    rows = cursor.fetchall()
    if not rows:
        return faiss.IndexFlatL2(dimension), []
    index = faiss.IndexFlatL2(dimension)
    stored_data = []
    for row in rows:
        embedding_blob, combined_text, summary = row
        if embedding_blob:
            embedding = pickle.loads(embedding_blob)
            index.add(np.array([embedding]))
            stored_data.append({"text": combined_text, "summary": summary})
    conn.close()
    return index, stored_data

def save_analysis_to_db(filename, transcript, summary, sentiment, emotion_analysis, keywords, combined_text, embedding):
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    embedding_blob = pickle.dumps(embedding[0])
    cursor.execute('''
        INSERT INTO video_analysis 
        (filename, transcript, summary, sentiment_polarity, sentiment_subjectivity, 
         sentiment_label, emotion_analysis, keywords, combined_text, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename, transcript, summary,
        sentiment.polarity if sentiment else None,
        sentiment.subjectivity if sentiment else None,
        "Positive" if sentiment and sentiment.polarity > 0.1 else "Negative" if sentiment and sentiment.polarity < -0.1 else "Neutral",
        emotion_analysis, keywords, combined_text, embedding_blob
    ))
    conn.commit()
    conn.close()

def get_all_analysis_data():
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, filename, transcript, summary, sentiment_polarity, 
               sentiment_subjectivity, sentiment_label, emotion_analysis, 
               keywords, created_at
        FROM video_analysis 
        ORDER BY created_at DESC
    ''')
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    result = [dict(zip(columns, row)) for row in rows]
    conn.close()
    return result

def extract_text_from_document(filepath, extension):
    text = ""
    if extension == '.pdf':
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif extension in ['.doc', '.docx']:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
        text = df.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ')
    return text.strip()

init_db()
index, stored_data = load_faiss_index()

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out successfully!', 'info')
    return redirect(url_for('login'))

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        file = request.files.get("media")
        selected_media_type = request.form.get("media_type")

        if not file:
            return "No file uploaded.", 400

        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()

        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        audio_formats = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']
        document_formats = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']

        if file_extension not in video_formats + audio_formats + document_formats:
            return "Unsupported file format. Please upload a video, audio or document file.", 400

        is_video = file_extension in video_formats
        is_audio = file_extension in audio_formats
        is_document = file_extension in document_formats

        if is_video:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"video{file_extension}")
        elif is_audio:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"audio{file_extension}")
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"document{file_extension}")

        file.save(filepath)

        try:
            if is_document:
                transcript = extract_text_from_document(filepath, file_extension)
                if not transcript.strip():
                    return "Failed to extract text from the document or document is empty.", 400
            else:
                ffmpeg = get_ffmpeg_executable()
                temp_audio_path = "temp_audio.wav"
                compressed_audio_path = "compressed_audio.wav"

                if is_video:
                    video_clip = VideoFileClip(filepath)
                    video_clip.audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
                    video_clip.close()
                elif is_audio:
                    try:
                        audio_clip = AudioFileClip(filepath)
                        audio_clip.write_audiofile(temp_audio_path, logger=None, verbose=False)
                        audio_clip.close()
                    except Exception:
                        if file_extension == '.wav':
                            import shutil
                            shutil.copy2(filepath, temp_audio_path)
                        else:
                            result = subprocess.run([ffmpeg, '-i', filepath, '-y', temp_audio_path], capture_output=True)
                            if result.returncode != 0:
                                raise Exception(f"Failed to convert audio file: {filename}\n{result.stderr.decode()}")

                result = subprocess.run([ffmpeg, '-i', temp_audio_path, '-ac', '1', '-ar', '16000', '-y', compressed_audio_path], capture_output=True)
                if result.returncode != 0 or not os.path.exists(compressed_audio_path):
                    compressed_audio_path = temp_audio_path

                with open(compressed_audio_path, "rb") as af:
                    transcript_response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=af
                    )
                transcript = transcript_response.text

            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant that summarizes audio/video/document content concisely and accurately."},
                    {"role": "user", "content": f"Provide a comprehensive summary of this {selected_media_type.lower()} transcript/text:\n\n{transcript}"}
                ]
            )
            summary = summary_response.choices[0].message.content

            sentiment = TextBlob(transcript).sentiment if transcript else None
            polarity = sentiment.polarity if sentiment else 0
            subjectivity = sentiment.subjectivity if sentiment else 0
            sentiment_label = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"

            emotion_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in emotional intelligence and psychology. Analyze the emotional tone and provide insights."},
                    {"role": "user", "content": f"Analyze the emotional tone of this transcript/text and provide detailed insights:\n\n{transcript}"}
                ]
            )
            emotion_analysis = emotion_response.choices[0].message.content

            keywords_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract key topics, important keywords, and main themes from the provided text."},
                    {"role": "user", "content": f"Extract and list the most important keywords and themes from this transcript/text:\n\n{transcript}"}
                ]
            )
            keywords = keywords_response.choices[0].message.content

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)
            wordcloud_path = os.path.join("static", "wordcloud.png")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig(wordcloud_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()

            media_type = selected_media_type if selected_media_type else ("Video" if is_video else "Audio" if is_audio else "Document")
            combined_text = f"""Transcript ({media_type}): {transcript}\n\nSummary: {summary}\n\nSentiment Analysis: Polarity={polarity:.2f}, Subjectivity={subjectivity:.2f}, Label={sentiment_label}\n\nEmotion Analysis: {emotion_analysis}\n\nKeywords: {keywords}"""

            embedding_vector = embedding_model.encode([combined_text])
            global index, stored_data
            index.add(np.array(embedding_vector))
            stored_data.append({"text": combined_text, "summary": summary})

            save_analysis_to_db(
                filename, transcript, summary, sentiment, 
                emotion_analysis, keywords, combined_text, embedding_vector
            )

            for temp_file in [filepath, "temp_audio.wav", "compressed_audio.wav"]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

            return render_template("result.html", 
                                 transcript=transcript, 
                                 summary=summary,
                                 polarity=polarity, 
                                 subjectivity=subjectivity,
                                 sentiment=sentiment_label, 
                                 emotion=emotion_analysis,
                                 keywords=keywords, 
                                 wordcloud_path=wordcloud_path,
                                 filename=filename,
                                 media_type=media_type)

        except Exception as e:
            for temp_file in ["temp_audio.wav", "compressed_audio.wav", filepath]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            return f"Error processing {selected_media_type or ('video' if is_video else 'audio' if is_audio else 'document')}: {str(e)}", 500

    recent_analysis = get_all_analysis_data()
    return render_template("index.html", recent_analysis=recent_analysis[:5])

@app.route("/download/<filename>")
@login_required
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)


@app.route("/get-video-list")
@login_required
def get_video_list():
    """Get list of all analyzed videos for the filter dropdown"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, filename FROM video_analysis ORDER BY created_at DESC')
    videos = cursor.fetchall()
    
    conn.close()
    return jsonify([{"id": video[0], "filename": video[1]} for video in videos])

@app.route("/ask-audio", methods=["POST"])
@login_required
def ask_audio():
    audio_file = request.files.get("audio")
    selected_filename = request.form.get("filename")  # get filename from frontend form

    if not audio_file:
        return jsonify({"error": "No audio file provided."}), 400

    temp_path = "temp_mic.wav"
    audio_file.save(temp_path)

    try:
        with open(temp_path, "rb") as af:
            whisper_result = client.audio.transcriptions.create(
                model="whisper-1",
                file=af
            )
        mic_text = whisper_result.text

        if not stored_data:
            return jsonify({"answer": "No data available. Please upload media first."})

        conn = sqlite3.connect('video_analysis.db')
        cursor = conn.cursor()

        if selected_filename and selected_filename.lower() != "all":
            cursor.execute('SELECT combined_text, embedding FROM video_analysis WHERE filename = ?', (selected_filename,))
            rows = cursor.fetchall()
            if not rows:
                return jsonify({"answer": f"No analysis data found for file: {selected_filename}"})

            temp_index = faiss.IndexFlatL2(dimension)
            temp_stored_data = []
            embeddings = []

            for combined_text, emb_blob in rows:
                emb = pickle.loads(emb_blob)
                embeddings.append(emb)
                temp_stored_data.append({"text": combined_text})

            embeddings_np = np.array(embeddings).astype('float32')
            temp_index.add(embeddings_np)

            question_embedding = embedding_model.encode([mic_text]).astype('float32')
            D, I = temp_index.search(question_embedding, k=1)

            if len(I[0]) == 0 or I[0][0] == -1:
                return jsonify({"answer": "No relevant information found in the selected file."})

            context_text = temp_stored_data[I[0][0]]['text']
            similarity_score = float(D[0][0])
        else:
            query_vector = embedding_model.encode([mic_text])
            D, I = index.search(np.array(query_vector), k=1)
            if len(I[0]) == 0:
                return jsonify({"answer": "No relevant information found in the database."})
            context_text = stored_data[I[0][0]]['text']
            similarity_score = float(D[0][0])

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant answering questions based on video analysis data. Provide accurate, detailed responses based on the context provided."},
            {"role": "user", "content": f"Context from analyzed media:\n{context_text}\n\nUser Question: {mic_text}\n\nPlease provide a comprehensive answer based on the media content."}
        ]

        print("Sending chat messages to OpenAI GPT:", messages)

        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        answer = chat_response.choices[0].message.content

        return jsonify({
            "question": mic_text,
            "answer": answer,
            "similarity_score": similarity_score
        })

    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/ask-text", methods=["POST"])
@login_required
def ask_text():
    data = request.get_json()
    question = data.get("question", "").strip()
    selected_filename = data.get("filename")  # The filename selected by user

    if not question:
        return jsonify({"error": "No question provided."}), 400

    if not stored_data:
        return jsonify({"answer": "No data available. Please upload media first."})

    try:
        conn = sqlite3.connect('video_analysis.db')
        cursor = conn.cursor()

        if selected_filename and selected_filename.lower() != "all":
            # Fetch all rows matching the selected filename
            cursor.execute('SELECT combined_text, embedding FROM video_analysis WHERE filename = ?', (selected_filename,))
            rows = cursor.fetchall()
            if not rows:
                return jsonify({"answer": f"No analysis data found for file: {selected_filename}"})

            # Build FAISS index for this file's embeddings only
            temp_index = faiss.IndexFlatL2(dimension)
            temp_stored_data = []
            embeddings = []

            for combined_text, emb_blob in rows:
                emb = pickle.loads(emb_blob)
                embeddings.append(emb)
                temp_stored_data.append({"text": combined_text})

            embeddings_np = np.array(embeddings).astype('float32')
            temp_index.add(embeddings_np)

            # Encode question embedding
            question_embedding = embedding_model.encode([question]).astype('float32')

            # Search in temporary index
            D, I = temp_index.search(question_embedding, k=1)

            if len(I[0]) == 0 or I[0][0] == -1:
                return jsonify({"answer": "No relevant information found in the selected file."})

            context_text = temp_stored_data[I[0][0]]['text']
            similarity_score = float(D[0][0])

        else:
            # If "all" or no filename selected, search global index as usual
            query_vector = embedding_model.encode([question])
            D, I = index.search(np.array(query_vector), k=1)
            if len(I[0]) == 0:
                return jsonify({"answer": "No relevant information found in the database."})
            context_text = stored_data[I[0][0]]['text']
            similarity_score = float(D[0][0])

        # Prepare messages for GPT
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant answering questions based on video analysis data. Provide accurate, detailed responses based on the context provided."},
            {"role": "user", "content": f"Context from analyzed media:\n{context_text}\n\nUser Question: {question}\n\nPlease provide a comprehensive answer based on the media content."}
        ]

        print("Sending chat messages to OpenAI GPT:", messages)

        chat_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        answer = chat_response.choices[0].message.content

        return jsonify({
            "question": question,
            "answer": answer,
            "similarity_score": similarity_score
        })

    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500


@app.route("/analysis-history")
@login_required
def analysis_history():
    """View all stored video analysis"""
    all_analysis = get_all_analysis_data()
    return render_template("history.html", analysis_data=all_analysis)


@app.route("/clear-data", methods=["POST"])
@login_required
def clear_data():
    """Clear all stored data and reset FAISS index"""
    global index, stored_data
    
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM video_analysis')
    cursor.execute('DELETE FROM faiss_index_meta')
    conn.commit()
    conn.close()
    
    # Reset FAISS index and stored data
    index = faiss.IndexFlatL2(dimension)
    stored_data = []
    
    return jsonify({"message": "All data cleared successfully."})


if __name__ == "__main__":
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print("🚀 Starting Flask Video Analyzer...")
    print(f"📊 Loaded {len(stored_data)} existing analysis records")
    print(f"🔐 Login credentials: Username: {ADMIN_USERNAME}, Password: {ADMIN_PASSWORD}")
    
    app.run(host="0.0.0.0", port=5000)