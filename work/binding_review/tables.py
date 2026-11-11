"""Exact seed-level tables for the revised manuscript."""
import json

from offline import OUT, ROOT


def display(value, scale=1, digits=3):
    return "—" if value is None else f"{value*scale:.{digits}f}"


def main():
    runs = json.loads((OUT / "analysis.json").read_text(encoding="utf8"))
    lines = [
        "# 修订实验逐种子全表", "",
        "所有新增读数使用 FP32 前向。密集轨迹在每键两个测试上下文评估，固定第 16→64 步变化使用六个上下文。",
        "缺口闭合表的主要目标为 Maple 验证正确完成序列平均 logP −1.0，−0.5 为敏感性目标；100% 是共同绝对目标的辅助比较。单元格包含首次达标区间内的插值，实测点另列。",
        "JS 的第二个数是加入其余桶最坏内部信息后得到的上界。所有数字以种子为单位，不把上下文当作独立重复。", "",
        "| 分支 | 完成 | 最后步 | 最后 Maple logP | 固定 16→64 ΔI | 固定 16→64 Δ旧质量 pp |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        last = run["trace"][-1]
        change = run["fixed_change"] or {}
        lines.append(f"| {name} | {run['complete']} | {last['step']} | {last['maple']:.3f} | "
                     f"{display(change.get('information'))} | {display(change.get('old_correct'),100)} |")
    lines += ["", "| 分支 | 目标 | 进度 | 插值步 | 旧正确质量 % | I(K;Y) nats | JS / 上界 nats | 预定活跃键同键新值 % / 活跃键内置换 % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        for p in run["progress"]:
            if p["comparable"]:
                m = p["interpolated"]
                lines.append(f"| {name} | {p['target']} | {p['fraction']:.0%} | {p['step']:.2f} | "
                             f"{m['old_correct']*100:.2f} | {m['information']:.3f} | "
                             f"{m['js']:.4f} / {m['js_upper']:.4f} | "
                             f"{m['trained_paired']*100:.2f} / {m['trained_null']*100:.2f} |")
            else:
                lines.append(f"| {name} | {p['target']} | {p['fraction']:.0%} | 未覆盖 | — | — | — | — |")
    lines += ["","## 共同绝对目标的首次实测点","",
              "| 分支 | 目标 | 实测步 | 验证 Maple logP | 实测旧质量 % | 其余均值 % / 最大键 % |",
              "|---|---:|---:|---:|---:|---:|"]
    for name,run in runs.items():
        for p in run["progress"]:
            if p["fraction"]!=1 or not p["comparable"]:
                continue
            step = p["observed_step"]
            row = next(r for r in run["trace"] if r["step"]==step)
            m = p["observed"]
            lines.append(f"| {name} | {p['target']} | {step} | {row['maple']:.3f} | "
                         f"{m['old_correct']*100:.2f} | {m['other']*100:.2f} / {m['other_max']*100:.2f} |")
    target = ROOT / "paper/review_tables.md"
    target.write_text("\n".join(lines)+"\n", encoding="utf8")
    print(target)


if __name__ == "__main__":
    main()
