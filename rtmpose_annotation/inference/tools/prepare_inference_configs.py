# inference/tools/prepare_inference_configs.py
import argparse
from mmengine.config import Config

def flatten_and_strip(src: str, dst: str):
    # 1) flatten (resolve _base_ / imports into one file)
    cfg = Config.fromfile(src)
    cfg.dump(dst)

    # 2) remove training-only init_cfg so inference won't look for extra pretrained files
    cfg2 = Config.fromfile(dst)

    def strip_init_cfg(d):
        if isinstance(d, dict) and 'init_cfg' in d:
            d['init_cfg'] = None

    m = cfg2.get('model', {})
    if isinstance(m, dict):
        strip_init_cfg(m.get('backbone', {}))
        strip_init_cfg(m.get('neck', {}))
        strip_init_cfg(m.get('head', {}))
        strip_init_cfg(m.get('pose_head', {}))
        # also handle lists of components (rare)
        for k, v in list(m.items()):
            if isinstance(v, list):
                for part in v:
                    strip_init_cfg(part)

    cfg2.dump(dst)
    print(f"[OK] Wrote flattened config: {dst}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-src", required=True)
    ap.add_argument("--pose-dst", required=True)
    ap.add_argument("--det-src", required=True)
    ap.add_argument("--det-dst", required=True)
    args = ap.parse_args()

    flatten_and_strip(args.pose_src, args.pose_dst)
    flatten_and_strip(args.det_src, args.det_dst)

if __name__ == "__main__":
    main()
