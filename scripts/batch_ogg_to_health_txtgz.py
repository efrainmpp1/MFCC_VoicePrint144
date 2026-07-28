import argparse
import gzip
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voiceprint_features_144.extract_health_matrix import extract_health_matrix


def matrix_to_txtgz_bytes(matrix) -> bytes:
    content = "\n".join(" ".join(str(int(v)) for v in row) for row in matrix) + "\n"
    return gzip.compress(content.encode("utf-8"))


def convert_ogg_to_wav(ogg_path: Path, wav_path: Path) -> None:
    # Mesmos parâmetros usados em produção (TransformAudioService.js) para
    # que áudio de origem externa produza um sinal comparável ao de produção:
    # 44100Hz, PCM 16-bit, mono.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(ogg_path),
            "-ac", "1",
            "-ar", "44100",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def convert_file(
    ogg_path: Path,
    input_dir: Path,
    out_dir: Path,
    out_name_stem: str,
    n_frames: int,
    fmin: int,
    fmax: int,
    pcen: bool,
    down16k: bool,
) -> Path:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / f"{ogg_path.stem}.wav"
        convert_ogg_to_wav(ogg_path, wav_path)

        matrix, _sr, _band = extract_health_matrix(
            str(wav_path),
            target_frames=n_frames,
            use_pcen=pcen,
            force_down_to_16k=down16k,
            fmin=fmin,
            fmax=fmax,
        )

    relative_dir = ogg_path.parent.relative_to(input_dir)
    target_dir = out_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{out_name_stem}.txt.gz"
    out_path.write_bytes(matrix_to_txtgz_bytes(matrix))
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Converte um lote de .ogg (ex: notas de voz do WhatsApp) para .wav "
            "e extrai health_matrix (.txt.gz), preservando a estrutura de "
            "subpastas (estado/label) da entrada."
        )
    )
    ap.add_argument("input_dir", help="Pasta com subpastas de .ogg (uma por estado/label)")
    ap.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Pasta de saída para os .txt.gz (default: mesma pasta de entrada)",
    )
    ap.add_argument(
        "--n-frames",
        type=int,
        default=400,
        help="Frames alvo da matriz (default: 400, igual ao padrão do extrator)",
    )
    ap.add_argument("--fmin", type=int, default=100, help="Frequência mínima (default: 100)")
    ap.add_argument("--fmax", type=int, default=7200, help="Frequência máxima (default: 7200)")
    ap.add_argument("--pcen", action="store_true", help="Usar PCEN em vez de dB nas bandas Mel")
    ap.add_argument(
        "--no-down16k",
        action="store_true",
        help="Não forçar downsample para 16kHz quando sr > 16kHz",
    )
    ap.add_argument(
        "--name-prefix",
        default="efrainmpp",
        help="Prefixo do nome de saída: <prefixo>_test_<estado>_<NN>.txt.gz (default: efrainmpp)",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ogg_files = sorted(input_dir.rglob("*.ogg"))
    if not ogg_files:
        print(f"Nenhum .ogg encontrado em {input_dir}")
        return

    # Agrupa por pasta (estado) para numerar sequencialmente dentro de cada uma.
    counters = {}
    for ogg_path in ogg_files:
        estado = ogg_path.parent.name.lower()
        counters[estado] = counters.get(estado, 0) + 1
        idx = counters[estado]
        out_name_stem = f"{args.name_prefix}_test_{estado}_{idx:02d}"

        try:
            out_path = convert_file(
                ogg_path,
                input_dir,
                output_dir,
                out_name_stem,
                args.n_frames,
                args.fmin,
                args.fmax,
                args.pcen,
                not args.no_down16k,
            )
            print(f"OK  {ogg_path.relative_to(input_dir)} -> {out_path}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
            print(f"ERRO {ogg_path.relative_to(input_dir)}: ffmpeg falhou: {stderr.strip()[-300:]}")
        except Exception as exc:
            print(f"ERRO {ogg_path.relative_to(input_dir)}: {exc}")


if __name__ == "__main__":
    main()
