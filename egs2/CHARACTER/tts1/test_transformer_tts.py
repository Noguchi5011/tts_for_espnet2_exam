from espnet2.bin.tts_inference import Text2Speech
import time
import torch
import soundfile as sf

fs, lang = 44100, "Japanese"
model= "exp/tts_finetune_10epoch_transformer_tts_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave.pth"

text2speech = Text2Speech.from_pretrained(
    model_file=model,
    device="cpu",
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

print(f"Input your favorite sentence in {lang}.")
# x = input()
x ="水をマレーシアから買わなくてはならないのです。"

with torch.no_grad():
    start = time.time()
    wav = text2speech(x)["wav"]
rtf = (time.time() - start) / (len(wav) / text2speech.fs)
print(f"RTF = {rtf:5f}")

wavdata = wav.view(-1).cpu().numpy()
samplerate=text2speech.fs

sf.write('tsukuyomi_transformer_tts_20_train.wav', wavdata, samplerate, subtype='PCM_24')