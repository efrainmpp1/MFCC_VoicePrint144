import argparse
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voiceprint_features_144.extract_mfcc_matrix import extract_mfcc_matrix


def matrix_to_txtgz_bytes(matrix) -> bytes:
    content = "\n".join(" ".join(str(int(v)) for v in row) for row in matrix) + "\n"
    return gzip.compress(content.encode("utf-8"))


def convert_file(
    wav_path: Path, input_dir: Path, out_dir: Path, n_frames: int, fmin: int, fmax: int
) -> Path:
    matrix, _sr, _band = extract_mfcc_matrix(
        str(wav_path), target_frames=n_frames, fmin=fmin, fmax=fmax
    )
    relative_dir = wav_path.parent.relative_to(input_dir)
    target_dir = out_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{wav_path.stem}.txt.gz"
    out_path.write_bytes(matrix_to_txtgz_bytes(matrix))
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Converte um lote de .wav locais para .txt.gz (modo mfcc_matrix, 144D)."
    )
    ap.add_argument("input_dir", help="Pasta com os arquivos .wav")
    ap.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Pasta de saída para os .txt.gz (default: mesma pasta de entrada)",
    )
    ap.add_argument(
        "--n-frames",
        type=int,
        default=20000,
        help="Frames alvo da matriz (default: 20000, igual ao padrão do extrator)",
    )
    ap.add_argument("--fmin", type=int, default=100, help="Frequência mínima (default: 100)")
    ap.add_argument("--fmax", type=int, default=7000, help="Frequência máxima (default: 7000)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.rglob("*.wav"))
    if not wav_files:
        print(f"Nenhum .wav encontrado em {input_dir}")
        return

    for wav_path in wav_files:
        try:
            out_path = convert_file(
                wav_path, input_dir, output_dir, args.n_frames, args.fmin, args.fmax
            )
            print(f"OK  {wav_path.name} -> {out_path}")
        except Exception as exc:
            print(f"ERRO {wav_path.name}: {exc}")


if __name__ == "__main__":
    main()
