from espnet2.bin.tts_inference import Text2Speech
import time
import torch
import soundfile as sf

fs, lang = 44100, "Japanese"
model= "downloads/f3698edf589206588f58f5ec837fa516/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.total_count.ave_10best.pth"

text2speech = Text2Speech.from_pretrained(
    model_file=model,
    device="cpu",
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

print(f"Input your favorite sentence in {lang}.")
x ="こんにちは。本日はいい天気ですね。"

with torch.no_grad():
    start = time.time()
    wav = text2speech(x)["wav"]
rtf = (time.time() - start) / (len(wav) / text2speech.fs)
print(f"RTF = {rtf:5f}")

wavdata = wav.view(-1).cpu().numpy()
samplerate=text2speech.fs

sf.write('normal_vits_10_train.wav', wavdata, samplerate, subtype='PCM_24')

