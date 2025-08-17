
# Stream a preview of the original (uploaded) file for the owner

PREVIEW_DURATION_SECONDS = 30

# Stream a preview of the original (uploaded) file for the owner
import logging

async def stream_original_audio_preview(db, file_id: str, user_id: str, duration: int = PREVIEW_DURATION_SECONDS):
	"""
	Stream a preview (first N seconds) of the original uploaded audio file for the owner.
	Accepts either an original file ID or a normalized file ID (resolves to original).
	"""
	logger = logging.getLogger("audionorm.preview")
	logger.setLevel(logging.INFO)
	def preview_log(msg):
		logger.info(msg)
		print(f"[PREVIEW] {msg}")

	preview_log(f"Incoming file_id: {file_id}")
	parsed_id = parse_file_id(file_id)
	preview_log(f"Parsed file_id: {parsed_id} (type: {type(parsed_id)})")
	bucket = AsyncIOMotorGridFSBucket(db)
	# Try to find as original upload first
	file_doc = await db["audio_files"].find_one({"_id": parsed_id})
	preview_log(f"Lookup in audio_files with _id={parsed_id}: {file_doc}")
	# If not found, try as normalized file and resolve original_upload_id
	if not file_doc:
		norm_doc = await db["audio_normalizations"].find_one({"_id": parsed_id})
		preview_log(f"Lookup in audio_normalizations with _id={parsed_id}: {norm_doc}")
		if norm_doc and norm_doc.get("original_upload_id"):
			preview_log(f"Found normalized file, resolving original_upload_id: {norm_doc['original_upload_id']}")
			file_doc = await db["audio_files"].find_one({"_id": parse_file_id(norm_doc["original_upload_id"])})
			preview_log(f"Lookup in audio_files with _id={norm_doc['original_upload_id']}: {file_doc}")
	if not file_doc:
		preview_log(f"No file found for id: {file_id} (parsed: {parsed_id})")
		return JSONResponse(status_code=404, content={"error": "Original file not found"})
	preview_log(f"Found file with _id: {file_doc.get('_id')}, user_id: {file_doc.get('user_id')}, gridfs_id: {file_doc.get('gridfs_id')}, filename: {file_doc.get('original_filename')}")
	if file_doc.get("user_id") != user_id:
		preview_log(f"User mismatch: file user_id={file_doc.get('user_id')} vs request user_id={user_id}")
		return JSONResponse(status_code=403, content={"error": "You can only preview your own files"})
	gridfs_id = file_doc.get("gridfs_id")
	if not gridfs_id:
		preview_log(f"File found but gridfs_id missing for _id: {file_doc.get('_id')}")
		return JSONResponse(status_code=404, content={"error": "File data not found"})
	sr = file_doc.get("sample_rate", 48000)
	channels = file_doc.get("channels", 1)
	import re
	original_filename = file_doc.get("original_filename", "audio.wav").lower()
	# Sanitize filename to ASCII only for Content-Disposition
	safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)
	# Allow preview for WAV, MP3, FLAC (convert to WAV on-the-fly if needed)
	SUPPORTED_EXTS = ['.wav', '.mp3', '.flac']
	ext = None
	for e in SUPPORTED_EXTS:
		if original_filename.endswith(e):
			ext = e
			break
	if not ext:
		preview_log(f"File extension not supported: {original_filename}")
		return JSONResponse(status_code=400, content={"error": "Preview only supported for WAV, MP3, FLAC files (by extension)"})

	grid_out = await bucket.open_download_stream(gridfs_id)
	grid_out.seek(0)
	audio_bytes = await grid_out.read()

	# If WAV, stream bytes directly (fast path)
	if ext == '.wav':
		preview_samples = int(sr * duration)
		preview_bytes = preview_samples * channels * BYTES_PER_SAMPLE
		preview_data = audio_bytes[:preview_bytes]
		headers = {
			"Content-Type": "audio/wav",
			"Content-Disposition": 'inline; filename="preview.wav"'
		}
		return StreamingResponse(
			iter([preview_data]),
			headers=headers,
			media_type="audio/wav"
		)

	# For MP3/FLAC, decode and convert to WAV in memory
	if not AUDIO_DEPS_AVAILABLE:
		return JSONResponse(status_code=500, content={"error": f"Audio dependencies not available: {LIBROSA_ERR}"})
	try:
		audio_buffer = io.BytesIO(audio_bytes)
		audio, sr = librosa.load(audio_buffer, sr=None, mono=True)
		preview_samples = int(sr * duration)
		audio_preview = audio[:preview_samples]
		wav_buffer = io.BytesIO()
		sf.write(wav_buffer, audio_preview, sr, format='WAV')
		wav_buffer.seek(0)
	except Exception as e:
		return JSONResponse(status_code=400, content={"error": f"Failed to decode/convert audio: {str(e)}"})
	headers = {
		"Content-Type": "audio/wav",
		"Content-Disposition": 'inline; filename="preview.wav"'
	}
	return StreamingResponse(
		wav_buffer,
		headers=headers,
		media_type="audio/wav"
	)
import io
try:
	import librosa
	import soundfile as sf
	import pyloudnorm as pyln
	AUDIO_DEPS_AVAILABLE = True
except ImportError as e:
	AUDIO_DEPS_AVAILABLE = False
	LIBROSA_ERR = str(e)

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from bson import ObjectId, errors as bson_errors
def parse_file_id(file_id):
	try:
		return ObjectId(file_id)
	except (bson_errors.InvalidId, TypeError):
		return file_id  # treat as string/UUID
from fastapi.responses import StreamingResponse, JSONResponse

PREVIEW_DURATION_SECONDS = 30
BYTES_PER_SAMPLE = 2  # 16-bit PCM

async def generate_normalized_preview(db, file_id: str, user_id: str, target_lufs: float, duration: int = PREVIEW_DURATION_SECONDS):
	"""
	Generate and stream a preview (first N seconds) of the audio normalized to the selected LUFS level.
	"""
	if not AUDIO_DEPS_AVAILABLE:
		return JSONResponse(status_code=500, content={"error": f"Audio dependencies not available: {LIBROSA_ERR}"})
	bucket = AsyncIOMotorGridFSBucket(db)
	file_doc = await db["audio_files"].find_one({"_id": parse_file_id(file_id)})
	if not file_doc:
		return JSONResponse(status_code=404, content={"error": "Audio file not found"})
	if file_doc.get("user_id") != user_id:
		return JSONResponse(status_code=403, content={"error": "You can only preview your own files"})
	gridfs_id = file_doc.get("gridfs_id")
	if not gridfs_id:
		return JSONResponse(status_code=404, content={"error": "File data not found"})
	# Read the file from GridFS
	try:
		grid_out = await bucket.open_download_stream(gridfs_id)
		audio_bytes = await grid_out.read()
	except Exception as e:
		return JSONResponse(status_code=500, content={"error": f"Failed to read audio from storage: {str(e)}"})
	# Load audio with librosa
	audio_buffer = io.BytesIO(audio_bytes)
	try:
		audio, sr = librosa.load(audio_buffer, sr=None, mono=True)
	except Exception as e:
		return JSONResponse(status_code=400, content={"error": f"Failed to load audio: {str(e)}"})
	# Take preview segment
	preview_samples = int(sr * duration)
	audio_preview = audio[:preview_samples]
	# Normalize to target LUFS
	try:
		meter = pyln.Meter(sr)
		loudness = meter.integrated_loudness(audio_preview)
		normalized_audio = pyln.normalize.loudness(audio_preview, loudness, target_lufs)
	except Exception as e:
		return JSONResponse(status_code=500, content={"error": f"Failed to normalize audio: {str(e)}"})
	# Write to WAV in memory
	try:
		wav_buffer = io.BytesIO()
		sf.write(wav_buffer, normalized_audio, sr, format='WAV')
		wav_buffer.seek(0)
	except Exception as e:
		return JSONResponse(status_code=500, content={"error": f"Failed to write WAV: {str(e)}"})
	headers = {
		"Content-Type": "audio/wav",
		"Content-Disposition": 'inline; filename="preview.wav"'
	}
	return StreamingResponse(
		wav_buffer,
		headers=headers,
		media_type="audio/wav"
	)



async def stream_audio_preview(db, file_id: str, user_id: str, duration: int = PREVIEW_DURATION_SECONDS):
	"""
	Stream a preview (first N seconds) of the normalized audio file for the owner.
	"""
	bucket = AsyncIOMotorGridFSBucket(db)
	file_doc = await db["audio_normalizations"].find_one({
		"$or": [
			{"_id": parse_file_id(file_id)},
			{"file_id": file_id}
		]
	})
	if not file_doc:
		return JSONResponse(status_code=404, content={"error": "Normalized file not found"})
	if file_doc.get("user_id") != user_id:
		return JSONResponse(status_code=403, content={"error": "You can only preview your own files"})
	gridfs_id = file_doc.get("gridfs_id")
	if not gridfs_id:
		return JSONResponse(status_code=404, content={"error": "File data not found"})
	sr = file_doc.get("sample_rate", 48000)
	channels = file_doc.get("channels", 1)
	normalized_filename = file_doc.get("normalized_filename", "").lower()
	if not normalized_filename.endswith('.wav'):
		return JSONResponse(status_code=400, content={"error": "Preview only supported for WAV files (by extension)"})
	preview_samples = int(sr * duration)
	preview_bytes = preview_samples * channels * BYTES_PER_SAMPLE
	grid_out = await bucket.open_download_stream(gridfs_id)
	grid_out.seek(0)
	preview_data = await grid_out.read(preview_bytes)
	headers = {
		"Content-Type": "audio/wav",
		"Content-Disposition": 'inline; filename="preview.wav"'
	}
	return StreamingResponse(
		iter([preview_data]),
		headers=headers,
		media_type="audio/wav"
	)
