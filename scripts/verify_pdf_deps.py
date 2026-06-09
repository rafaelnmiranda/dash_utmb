#!/usr/bin/env python3
"""Smoke test das dependências de PDF (kaleido + WeasyPrint)."""

from __future__ import annotations

import sys


def main() -> int:
    errors: list[str] = []

    # 1) Kaleido + Plotly PNG
    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(x=["A"], y=[1]))
        png = fig.to_image(format="png", scale=1)
        if not png or png[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append("kaleido/plotly: PNG inválido")
        else:
            print(f"OK kaleido/plotly ({len(png)} bytes PNG)")
    except Exception as exc:
        errors.append(f"kaleido/plotly: {exc}")

    # 2) WeasyPrint import + PDF mínimo
    try:
        from weasyprint import HTML

        pdf = HTML(string="<html><body><h1>Teste</h1></body></html>").write_pdf()
        if not pdf or pdf[:4] != b"%PDF":
            errors.append("weasyprint: PDF inválido")
        else:
            print(f"OK weasyprint ({len(pdf)} bytes PDF)")
    except Exception as exc:
        errors.append(f"weasyprint: {exc}")

    # 3) pdf_export.render_pdf_bytes
    try:
        from pdf_export import render_pdf_bytes

        pdf = render_pdf_bytes("<html><body><p>pdf_export</p></body></html>")
        if pdf[:4] != b"%PDF":
            errors.append("pdf_export: PDF inválido")
        else:
            print(f"OK pdf_export.render_pdf_bytes ({len(pdf)} bytes)")
    except Exception as exc:
        errors.append(f"pdf_export: {exc}")

    if errors:
        print("\nFALHAS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nTodas as verificações passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
