import argparse, json
from .mfcc144 import extract_mfcc_144
from .mel144 import extract_logmel_144

def main():
    ap = argparse.ArgumentParser(description="Extract fixed 144D audio features (MFCC or Log-Mel).")
    ap.add_argument("wav", help="Path to .wav file")
    ap.add_argument("--mode", choices=["mfcc", "logmel"], default="mfcc")
    ap.add_argument("--pcen", action="store_true", help="Use PCEN (only for logmel)")
    ap.add_argument("--no-down16k", action="store_true", help="Do not force downsample to 16 kHz when sr>16k")
    ap.add_argument("--out", default="", help="Save JSON to file instead of printing")
    args = ap.parse_args()

    if args.mode == "mfcc":
        vec, sr, band = extract_mfcc_144(args.wav, force_down_to_16k=not args.no_down16k)
    else:
        vec, sr, band = extract_logmel_144(args.wav, use_pcen=args.pcen, force_down_to_16k=not args.no_down16k)

    payload = {"sr": int(sr), "band": band, "mode": args.mode, "shape": [144], "features": list(map(float, vec))}
    text = json.dumps(payload)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)

if __name__ == "__main__":
    main()
