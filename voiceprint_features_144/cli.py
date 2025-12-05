import argparse, json
from .mfcc144 import extract_mfcc_144
from .mel144 import extract_logmel_144
from .extract_health_matrix import extract_health_matrix

def main():
    ap = argparse.ArgumentParser(description="Extract 144D audio features (vector or per-frame matrix).")
    ap.add_argument("wav", help="Path to .wav file")
    ap.add_argument("--mode", choices=["mfcc", "logmel", "health_matrix"], default="mfcc")
    ap.add_argument("--pcen", action="store_true", help="Use PCEN (logmel or health_matrix)")
    ap.add_argument("--no-down16k", action="store_true", help="Do not force downsample to 16 kHz when sr>16k")
    ap.add_argument("--n-frames", type=int, default=400, help="Target frames for health_matrix (default: 400)")
    ap.add_argument("--fmin", type=int, default=100, help="Min frequency (health_matrix)")
    ap.add_argument("--fmax", type=int, default=7200, help="Max frequency (health_matrix)")
    ap.add_argument("--out", default="", help="Save JSON to file instead of printing")
    args = ap.parse_args()

    if args.mode == "mfcc":
        vec, sr, band = extract_mfcc_144(args.wav, force_down_to_16k=not args.no_down16k)
        payload = {"sr": int(sr), "band": band, "mode": args.mode, "shape": [144], "features": list(map(float, vec))}
    elif args.mode == "logmel":
        vec, sr, band = extract_logmel_144(args.wav, use_pcen=args.pcen, force_down_to_16k=not args.no_down16k)
        payload = {"sr": int(sr), "band": band, "mode": args.mode, "pcen": bool(args.pcen), "shape": [144], "features": list(map(float, vec))}
    else:
        mat, sr, band = extract_health_matrix(
            args.wav,
            target_frames=args.n_frames,
            use_pcen=args.pcen,
            force_down_to_16k=not args.no_down16k,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        payload = {
            "sr": int(sr),
            "band": band,
            "mode": args.mode,
            "pcen": bool(args.pcen),
            "shape": [int(args.n_frames), 144],
            "features": mat.tolist(),
        }

    text = json.dumps(payload)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)

if __name__ == "__main__":
    main()
