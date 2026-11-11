"""Create an editable PowerPoint presentation from generated slide content."""

from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def create_presentation(slides, paper_title="Research Paper Presentation"):
    """Return the generated PPTX as bytes for Streamlit download."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Title slide
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = str(paper_title)[:120]
    title_slide.placeholders[1].text = "Paper-to-Presentation & Viva Agent"
    for paragraph in title_slide.shapes.title.text_frame.paragraphs:
        paragraph.font.size = Pt(30)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor(31, 78, 121)

    for item in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = str(item.get("title", "Key points"))[:120]
        body = slide.placeholders[1].text_frame
        body.clear()

        bullets = item.get("bullets", [])
        if isinstance(bullets, str):
            bullets = [bullets]

        for index, bullet in enumerate(bullets[:6]):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            paragraph.text = str(bullet)
            paragraph.level = 0
            paragraph.font.size = Pt(21)

        notes = item.get("speaker_notes", "")
        if notes:
            try:
                slide.notes_slide.notes_text_frame.text = str(notes)
            except Exception:
                pass

    output = BytesIO()
    prs.save(output)
    return output.getvalue()
