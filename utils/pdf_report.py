from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from datetime import datetime

def generate_prediction_pdf(
    file_path,
    stock_name,
    prediction_date,
    predicted_price,
    confidence,
    chart_path=None
):
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Stock Market Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Metadata
    elements.append(Paragraph(f"<b>Stock:</b> {stock_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Prediction Date:</b> {prediction_date}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    # Prediction details
    elements.append(Paragraph("<b>Prediction Summary</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Predicted Price: <b>${predicted_price}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Model Confidence: <b>{confidence}%</b>", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Chart (optional)
    if chart_path:
        elements.append(Paragraph("<b>Prediction Chart</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image(chart_path, width=5 * inch, height=3 * inch))

    doc.build(elements)
