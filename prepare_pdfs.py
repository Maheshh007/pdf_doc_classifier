"""Step 1: build the PDF corpus.
 - 20 REAL receipt photos (CVAT receipt dataset) -> scanned PDFs  (data/raw_pdfs/receipt/real_*.pdf)
 - SYNTHETIC invoices / bills / receipts (reportlab), ~1/3 rasterised + degraded to mimic scans.
   Utility bills are generated from rows of the supplied Household energy bill CSV.
Synthetic files are prefixed `syn_` so they can always be told apart from real ones.
"""
import io, random, glob, os
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
import pypdfium2 as pdfium
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

random.seed(7); np.random.seed(7)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_pdfs")
N_PER_CLASS = 110

COMPANIES = ["Apex Industrial Supply", "BlueLine Logistics", "Nimbus Cloud Services", "Orion Consulting Group",
             "Greenfield Packaging", "Summit Software Ltd", "Pioneer Print & Design", "Kestrel Engineering",
             "Lotus Office Solutions", "Vertex Marketing Agency", "Harbor Freight Partners", "Delta Components Inc"]
CLIENTS = ["Acme Corp", "Brightway Traders", "Cedar Health Clinic", "Dune Analytics", "Everest Retail LLC",
           "Fusion Labs", "Granite Builders", "Horizon Foods"]
SERVICES = ["Consulting services", "Software licence (annual)", "Cloud hosting", "Design work", "Freight charges",
            "Maintenance contract", "Training workshop", "Hardware components", "Printing services", "Support hours"]
UTILS = ["City Power & Light", "MetroGas Energy", "AquaFlow Water Board", "NetStream Broadband", "Sunrise Mobile",
         "Valley Electricity Co", "ClearView Telecom", "GreenGrid Utilities"]
STORES = ["FRESH MART", "QUICK STOP", "CITY PHARMACY", "BURGER HUB", "CAFE LUNA", "BOOK NOOK", "SUPER SAVER",
          "DAILY GROCER", "TECH ZONE", "PIZZA PALACE", "GAS N GO", "HOME DEPOT-ISH"]
ITEMS = ["MILK 1L", "BREAD", "EGGS 12", "COFFEE", "NOTEBOOK", "USB CABLE", "SANDWICH", "COLA", "BATTERIES",
         "SOAP", "RICE 5KG", "CHIPS", "PEN SET", "WATER 500ML", "BANANAS", "CHEESE"]

def d():  # random date string in varied formats
    y, m, dd = random.randint(2019, 2024), random.randint(1, 12), random.randint(1, 28)
    return random.choice([f"{dd:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{dd:02d}", f"{m:02d}-{dd:02d}-{y}"])

def money(x): return f"{x:,.2f}"

# ---------------------------------------------------------------- invoices
def make_invoice(c):
    W, H = A4
    co, cl = random.choice(COMPANIES), random.choice(CLIENTS)
    title = random.choice(["INVOICE", "INVOICE", "TAX INVOICE", "Invoice"])
    y = H - 60
    c.setFont("Helvetica-Bold", 22); c.drawString(50, y, title)
    c.setFont("Helvetica", 10); c.drawRightString(W - 50, y, co)
    y -= 35
    c.drawString(50, y, f"Invoice No: INV-{random.randint(1000, 99999)}")
    c.drawString(250, y, f"Invoice Date: {d()}")
    if random.random() < .7: c.drawString(420, y, f"Due Date: {d()}")
    y -= 18
    if random.random() < .5: c.drawString(50, y, f"PO Number: PO-{random.randint(100, 9999)}")
    c.drawString(250, y, random.choice(["Terms: Net 30", "Terms: Net 15", "Payment terms: Due on receipt"]))
    y -= 30
    c.setFont("Helvetica-Bold", 10); c.drawString(50, y, "Bill To:")
    c.setFont("Helvetica", 10); c.drawString(50, y - 14, cl); c.drawString(50, y - 28, f"{random.randint(1,999)} Market Street")
    y -= 70
    c.setFont("Helvetica-Bold", 10)
    for x, h in [(50, "Description"), (300, "Qty"), (360, "Unit Price"), (460, "Amount")]: c.drawString(x, y, h)
    c.line(50, y - 4, W - 50, y - 4)
    y -= 20; sub = 0; c.setFont("Helvetica", 10)
    for _ in range(random.randint(2, 7)):
        q, p = random.randint(1, 20), random.uniform(15, 900); amt = q * p; sub += amt
        c.drawString(50, y, random.choice(SERVICES)); c.drawString(300, y, str(q))
        c.drawString(360, y, money(p)); c.drawString(460, y, money(amt)); y -= 16
    tax = sub * random.choice([.05, .08, .1, .18])
    y -= 10
    c.drawString(360, y, "Subtotal:"); c.drawString(460, y, money(sub)); y -= 16
    c.drawString(360, y, random.choice(["VAT:", "GST:", "Sales Tax:"])); c.drawString(460, y, money(tax)); y -= 18
    c.setFont("Helvetica-Bold", 11); c.drawString(360, y, "Total Due:"); c.drawString(460, y, money(sub + tax))
    c.setFont("Helvetica", 9)
    c.drawString(50, 90, random.choice(["Please remit payment within the stated terms.", "Bank transfer details: IBAN XX00 0000 0000",
                                         "Make cheques payable to " + co]))
    c.drawString(50, 76, "Thank you for your business.")

# ---------------------------------------------------------------- bills (utility / telecom / medical)
def make_bill(c, row):
    W, H = A4
    ut = random.choice(UTILS); y = H - 60
    c.setFont("Helvetica-Bold", 20); c.drawString(50, y, ut)
    c.setFont("Helvetica-Bold", 13); c.drawRightString(W - 50, y, random.choice(["Monthly Bill", "Billing Statement", "Your Bill", "Statement of Account"]))
    c.setFont("Helvetica", 10); y -= 32
    c.drawString(50, y, f"Account Number: {random.randint(10**8, 10**9)}")
    c.drawString(300, y, f"Billing Period: {d()} to {d()}"); y -= 16
    c.drawString(50, y, f"Customer: {random.choice(CLIENTS)} household"); c.drawString(300, y, f"Bill Date: {d()}"); y -= 30
    c.setFont("Helvetica-Bold", 10); c.drawString(50, y, "Usage Summary"); c.setFont("Helvetica", 10); y -= 16
    units = max(30, float(row["housearea"]) * .35 + float(row["num_people"]) * 20)
    for lab, val in [("Meter reading (previous)", random.randint(1000, 9000)), ("Meter reading (current)", random.randint(9000, 12000)),
                     ("Units consumed (kWh)", f"{units:.0f}"), ("Rate per unit", f"{random.uniform(.08,.3):.3f}")]:
        c.drawString(50, y, lab); c.drawString(300, y, str(val)); y -= 15
    y -= 10
    amt = float(row["amount_paid"])
    for lab, val in [("Previous balance", random.uniform(0, 80)), ("Current charges", amt),
                     ("Late payment fee", random.choice([0, 0, 5.0])), ("Service charge", random.uniform(2, 12))]:
        c.drawString(50, y, lab); c.drawString(300, y, money(val)); y -= 15
    c.setFont("Helvetica-Bold", 12); y -= 6
    c.drawString(50, y, "Total Amount Due:"); c.drawString(300, y, money(amt + random.uniform(2, 90)))
    c.setFont("Helvetica", 9); y -= 24
    c.drawString(50, y, f"Pay by {d()} to avoid disconnection.")
    c.drawString(50, y - 14, random.choice(["Pay online, by phone or at any authorised payment centre.", "Autopay customers: amount will be debited automatically.",
                                              "Questions about your bill? Call customer care."]))
    c.drawString(50, y - 28, f"Average daily usage: {units/30:.1f} kWh. Compare with last month and save energy.")

# ---------------------------------------------------------------- receipts (thermal style)
def make_receipt(c, w, h):
    st = random.choice(STORES); y = h - 25
    c.setFont("Courier-Bold", 12); c.drawCentredString(w / 2, y, st); y -= 14
    c.setFont("Courier", 8)
    c.drawCentredString(w / 2, y, f"{random.randint(1,999)} Main St  Tel {random.randint(100,999)}-{random.randint(1000,9999)}"); y -= 11
    c.drawString(10, y, f"{d()}  {random.randint(0,23):02d}:{random.randint(0,59):02d}"); c.drawRightString(w - 10, y, f"#{random.randint(1000,99999)}"); y -= 12
    c.drawString(10, y, "-" * 38); y -= 11; sub = 0
    for _ in range(random.randint(2, 9)):
        p = random.uniform(.5, 40); q = random.choice([1, 1, 1, 2, 3]); sub += p * q
        c.drawString(10, y, f"{random.choice(ITEMS)}" + (f" x{q}" if q > 1 else "")); c.drawRightString(w - 10, y, money(p * q)); y -= 11
    tax = sub * random.choice([.05, .07, .08]); tot = sub + tax
    c.drawString(10, y, "-" * 38); y -= 11
    c.drawString(10, y, "SUBTOTAL"); c.drawRightString(w - 10, y, money(sub)); y -= 11
    c.drawString(10, y, "TAX"); c.drawRightString(w - 10, y, money(tax)); y -= 11
    c.setFont("Courier-Bold", 9); c.drawString(10, y, "TOTAL"); c.drawRightString(w - 10, y, money(tot)); y -= 12
    c.setFont("Courier", 8)
    pay = random.choice(["CASH", "VISA ****%d" % random.randint(1000, 9999), "MASTERCARD ****%d" % random.randint(1000, 9999), "DEBIT CARD"])
    c.drawString(10, y, pay); c.drawRightString(w - 10, y, money(tot + (random.choice([0, 5, 10]) if pay == "CASH" else 0))); y -= 11
    if pay == "CASH": c.drawString(10, y, "CHANGE"); c.drawRightString(w - 10, y, money(random.choice([0, 5, 10]))); y -= 11
    y -= 6
    c.drawCentredString(w / 2, y, random.choice(["THANK YOU - COME AGAIN", "THANK YOU FOR SHOPPING", "HAVE A NICE DAY", "KEEP YOUR RECEIPT"]))
    if random.random() < .4: c.drawCentredString(w / 2, y - 11, "ITEMS SOLD: %d" % random.randint(2, 12))

def degrade(pdf_bytes):
    """Rasterise a digital PDF and make it look scanned (blur, noise, slight rotation, jpeg)."""
    img = pdfium.PdfDocument(pdf_bytes)[0].render(scale=2.2).to_pil().convert("L")
    img = img.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(.4, 1.0)))
    a = np.array(img).astype(np.float32) + np.random.normal(0, random.uniform(6, 18), (img.height, img.width))
    img = Image.fromarray(np.clip(a * random.uniform(.85, 1.0), 0, 255).astype(np.uint8))
    out = io.BytesIO(); img.save(out, "PDF", resolution=150, quality=55); return out.getvalue()

def render(fn, *args, size=A4):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=size); fn(c, *args); c.save(); return buf.getvalue()

def main():
    # 1. real receipts -> pdf
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "receipt_images", "*.jpg")), key=lambda p: int(os.path.basename(p)[:-4])):
        Image.open(f).convert("RGB").save(os.path.join(RAW, "receipt", f"real_{os.path.basename(f)[:-4]}.pdf"), "PDF", resolution=150)
    # 2. synthetic
    energy = pd.read_csv(os.path.join(ROOT, "data", "household_energy_bill_data.csv"))
    for i in range(N_PER_CLASS):
        for cls, gen in [("invoice", lambda: render(make_invoice)),
                         ("bill", lambda r=energy.iloc[i * 7 % len(energy)]: render(make_bill, r)),
                         ("receipt", lambda: (lambda w, h: render(make_receipt, w, h, size=(w, h)))(226, random.randint(300, 460)))]:
            b = gen()
            if random.random() < .35: b = degrade(b)
            open(os.path.join(RAW, cls, f"syn_{i:03d}.pdf"), "wb").write(b)
    # 3. two deliberately broken files so error handling is exercised
    open(os.path.join(RAW, "invoice", "syn_corrupt.pdf"), "wb").write(b"%PDF-1.4 this is not a real pdf")
    c = canvas.Canvas(os.path.join(RAW, "bill", "syn_blank.pdf")); c.showPage(); c.save()
    print({k: len(os.listdir(os.path.join(RAW, k))) for k in ["invoice", "bill", "receipt"]})

if __name__ == "__main__":
    main()
