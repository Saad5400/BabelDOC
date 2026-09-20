"""Drawing assignment must preserve objects without quadratic deep comparisons."""

from types import SimpleNamespace

from babeldoc.format.pdf.document_il.midend.styles_and_formulas import StylesAndFormulas


class Element:
    def __eq__(self, other):
        raise AssertionError("Drawing removal must not compare nested PDF objects")


def test_assigned_drawings_move_once_and_unassigned_order_is_preserved(monkeypatch):
    styles = StylesAndFormulas.__new__(StylesAndFormulas)
    curves = [Element() for _ in range(1000)]
    forms = [Element() for _ in range(1000)]
    formula = SimpleNamespace(pdf_curve=[], pdf_form=[])
    page = SimpleNamespace(
        pdf_paragraph=[object()], pdf_curve=curves[:], pdf_form=forms[:]
    )
    curve_list, form_list = page.pdf_curve, page.pdf_form
    monkeypatch.setattr(
        styles,
        "_collect_element_formula_candidates",
        lambda _page: (
            [(formula, None)],
            {i: (c, [(0, 1.0, "iou_exact")]) for i, c in enumerate(curves) if i % 2},
            {i: (f, [(0, 1.0, "iou_exact")]) for i, f in enumerate(forms) if i % 2},
        ),
    )
    styles.collect_contained_elements(page)
    assert page.pdf_curve is curve_list and page.pdf_form is form_list
    assert list(map(id, page.pdf_curve)) == list(map(id, curves[::2]))
    assert list(map(id, page.pdf_form)) == list(map(id, forms[::2]))
    assert list(map(id, formula.pdf_curve)) == list(map(id, curves[1::2]))
    assert list(map(id, formula.pdf_form)) == list(map(id, forms[1::2]))
