"""Streamlit presentation for realtime online-adaptation telemetry."""

from __future__ import annotations

from typing import Any

_DEFAULT_LABELS = ("LEFT", "RIGHT", "IDLE")
_CUE_SYMBOLS = {"left": "←", "right": "→", "idle": "○"}


def render_online_cue_panel(status: dict[str, Any] | None, *, ui: Any) -> None:
    """Render the automatic experiment cue source."""

    if not isinstance(status, dict) or status.get("source") != "cued-protocol":
        return
    phase = str(status.get("phase", "preparing"))
    label_name = str(status.get("label_name", "idle"))
    remaining = float(status.get("phase_remaining_sec", 0.0))
    phase_text = {
        "preparing": "准备",
        "fixation": "注视",
        "cue": "提示",
        "control": "运动想象 / 小车控制",
        "iti": "间隔休息",
        "done": "实验完成",
    }.get(phase, phase)
    if phase in {"cue", "control"}:
        prompt = f"{_CUE_SYMBOLS.get(label_name, '○')}  {label_name.upper()}"
    elif phase == "fixation":
        prompt = "+"
    elif phase == "done":
        prompt = "✓"
    else:
        prompt = "·"
    ui.markdown("### 连续自动 Cue")
    columns = ui.columns(4)
    columns[0].metric("阶段", phase_text)
    if bool(status.get("continuous", False)):
        trial_label = "累计 Trial"
        trial_text: str | int = int(status.get("trial_number", 0))
    else:
        trial_label = "Trial"
        trial_text = f"{int(status.get('trial_number', 0))}/{int(status.get('total_trials', 0))}"
    columns[1].metric(
        trial_label,
        trial_text,
    )
    columns[2].metric("目标", label_name.upper())
    columns[3].metric("剩余", f"{remaining:.1f}s")
    ui.markdown(f"<div style='text-align:center;font-size:5rem'>{prompt}</div>", unsafe_allow_html=True)


def render_online_adaptation_panel(adaptation: dict[str, Any] | None, *, ui: Any) -> None:
    """Render the dashboard for the active adaptation strategy, if any."""

    if not isinstance(adaptation, dict) or not adaptation.get("enabled"):
        return
    if str(adaptation.get("strategy", "periodic_head")) == "neuroonline":
        _render_neuroonline(adaptation, ui=ui)
    else:
        _render_periodic_head(adaptation, ui=ui)


def _render_periodic_head(adaptation: dict[str, Any], *, ui: Any) -> None:
    ui.markdown("### 10分钟周期模型更新")
    columns = ui.columns(4)
    columns[0].metric("状态", str(adaptation.get("state", "-")))
    columns[1].metric("模型版本", f"v{int(adaptation.get('model_version', 0))}")
    columns[2].metric("有效窗口", int(adaptation.get("buffered_windows", 0)))
    remaining = float(adaptation.get("seconds_until_update", 0.0))
    columns[3].metric("距下次检查", f"{remaining / 60.0:.1f} min")
    counts = adaptation.get("class_counts", {}) or {}
    ui.caption(
        "类别窗口 LEFT / RIGHT / IDLE: "
        f"{counts.get('0', 0)} / {counts.get('1', 0)} / {counts.get('2', 0)}"
    )
    last_result = adaptation.get("last_result")
    if isinstance(last_result, dict) and last_result.get("accepted"):
        ui.success(
            "最近一次更新已接受，balanced accuracy 提升 "
            f"{float(last_result.get('balanced_accuracy_gain', 0.0)):+.3f}"
        )
    elif isinstance(last_result, dict) and last_result.get("error"):
        ui.warning(f"最近一次更新失败: {last_result['error']}")
    elif last_result:
        ui.warning("最近一次候选模型未通过验证，继续使用旧模型。")


def _render_neuroonline(adaptation: dict[str, Any], *, ui: Any) -> None:
    ui.markdown("### NeuroOnline 在线适配")
    prequential = adaptation.get("prequential", {}) or {}
    last_result = adaptation.get("last_result") or {}
    top = ui.columns(5)
    top[0].metric("状态", str(adaptation.get("state", "-")))
    top[1].metric("更新次数", int(adaptation.get("update_count", 0)))
    top[2].metric("缓冲窗口", int(adaptation.get("buffered_windows", 0)))
    top[3].metric("在线 Bal.Acc.", f"{float(prequential.get('balanced_accuracy', 0.0)):.3f}")
    top[4].metric("最近更新耗时", f"{float(last_result.get('duration_sec', 0.0)):.2f}s")

    progress = float(adaptation.get("progress", 0.0))
    ui.progress(min(max(progress, 0.0), 1.0))
    ui.caption(
        f"累计有标签窗口 {int(adaptation.get('seen_labeled_windows', 0))} · "
        f"距下次更新 {int(adaptation.get('samples_until_update', 0))} 个样本 · "
        f"下一触发步 {int(adaptation.get('next_update_step', 0))}"
    )

    labels = _labels_for(adaptation, prequential)
    detail_left, detail_right = ui.columns(2)
    with detail_left:
        ui.caption("类别覆盖与累计表现")
        ui.dataframe(_class_rows(adaptation, prequential, labels), hide_index=True, width="stretch")
    with detail_right:
        ui.caption("累计混淆矩阵（行=真实，列=预测）")
        ui.dataframe(_confusion_rows(prequential, labels), hide_index=True, width="stretch")

    history = adaptation.get("update_history", []) or []
    if history:
        ui.caption("更新损失轨迹")
        ui.line_chart(
            history,
            x="update",
            y=["loss", "classification_loss", "consistency_loss"],
            width="stretch",
        )
        chart_left, chart_right = ui.columns(2)
        with chart_left:
            ui.caption("CRM gate 轨迹")
            ui.line_chart(history, x="update", y=["gate_alpha", "gate_beta"], width="stretch")
        with chart_right:
            ui.caption("在线累计性能")
            ui.line_chart(
                history,
                x="update",
                y=["prequential_accuracy", "prequential_balanced_accuracy"],
                width="stretch",
            )

    if last_result:
        ui.success(
            "最近一次更新完成："
            f"loss={float(last_result.get('loss', 0.0)):.4f}，"
            f"classification={float(last_result.get('classification_loss', 0.0)):.4f}，"
            f"consistency={float(last_result.get('consistency_loss', 0.0)):.4f}"
        )


def _labels_for(adaptation: dict[str, Any], prequential: dict[str, Any]) -> tuple[str, ...]:
    counts = adaptation.get("class_counts", {}) or {}
    confusion = prequential.get("confusion_matrix", []) or []
    class_count = max(len(counts), len(confusion), len(_DEFAULT_LABELS))
    return tuple(
        _DEFAULT_LABELS[index] if index < len(_DEFAULT_LABELS) else f"class-{index}"
        for index in range(class_count)
    )


def _class_rows(
    adaptation: dict[str, Any],
    prequential: dict[str, Any],
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    counts = adaptation.get("class_counts", {}) or {}
    per_class = prequential.get("per_class_accuracy", {}) or {}
    return [
        {
            "类别": label,
            "缓冲窗口": int(counts.get(str(index), 0)),
            "在线准确率": float(per_class.get(str(index), 0.0)),
        }
        for index, label in enumerate(labels)
    ]


def _confusion_rows(
    prequential: dict[str, Any],
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    confusion = prequential.get("confusion_matrix", []) or []
    rows: list[dict[str, Any]] = []
    for true_index, values in enumerate(confusion):
        true_label = labels[true_index] if true_index < len(labels) else f"class-{true_index}"
        row: dict[str, Any] = {"真实类别": true_label}
        for predicted_index, value in enumerate(values):
            predicted_label = (
                labels[predicted_index]
                if predicted_index < len(labels)
                else f"class-{predicted_index}"
            )
            row[f"预测 {predicted_label}"] = int(value)
        rows.append(row)
    return rows
