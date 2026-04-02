"""
Generates 3 sample PDF documents for testing the Smart Research Assistant.
Topics: AI & Machine Learning, Climate Change, Quantum Computing
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

OUTPUT_DIR = "test_documents"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DocTitle",
        fontSize=20, fontName="Helvetica-Bold",
        spaceAfter=8, alignment=TA_CENTER, textColor=colors.HexColor("#1a1a2e")
    ))
    styles.add(ParagraphStyle(
        name="DocSubtitle",
        fontSize=12, fontName="Helvetica",
        spaceAfter=20, alignment=TA_CENTER, textColor=colors.HexColor("#555577")
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=13, fontName="Helvetica-Bold",
        spaceBefore=16, spaceAfter=6, textColor=colors.HexColor("#2c3e7a")
    ))
    styles.add(ParagraphStyle(
        name="Body",
        fontSize=10, fontName="Helvetica",
        spaceAfter=8, leading=15, alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name="BulletItem",
        fontSize=10, fontName="Helvetica",
        leftIndent=20, spaceAfter=5, leading=14,
        bulletIndent=10
    ))
    return styles


# ── Document 1: Transformer Architecture ──────────────────────────────────────
def create_doc1(styles):
    path = os.path.join(OUTPUT_DIR, "transformer_architecture_overview.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=0.9*inch, leftMargin=0.9*inch,
                            topMargin=0.9*inch, bottomMargin=0.9*inch)
    story = []
    B, S, SH, BL = styles["Body"], styles["DocSubtitle"], styles["SectionHeader"], styles["BulletItem"]

    story.append(Paragraph("Transformer Architecture: A Comprehensive Overview", styles["DocTitle"]))
    story.append(Paragraph("Research Summary · AI & Deep Learning Series · 2024", S))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccccdd")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Abstract", SH))
    story.append(Paragraph(
        "The Transformer architecture, introduced by Vaswani et al. in the landmark 2017 paper "
        "'Attention Is All You Need', has fundamentally reshaped the landscape of deep learning. "
        "Originally designed for natural language processing tasks, Transformers have since been "
        "adapted for computer vision, speech recognition, protein structure prediction, and "
        "reinforcement learning. This document provides a comprehensive technical overview of the "
        "architecture, its components, training strategies, and real-world applications.", B))

    story.append(Paragraph("1. Core Architecture Components", SH))
    story.append(Paragraph(
        "The Transformer model consists of an encoder-decoder structure, though many modern variants "
        "use only the encoder (BERT) or only the decoder (GPT). The architecture relies entirely on "
        "attention mechanisms, abandoning recurrent and convolutional layers entirely.", B))

    story.append(Paragraph("1.1 Self-Attention Mechanism", SH))
    story.append(Paragraph(
        "Self-attention allows each token in a sequence to attend to all other tokens simultaneously. "
        "Given an input matrix X, the mechanism computes Query (Q), Key (K), and Value (V) matrices "
        "via learned linear projections. The attention output is computed as: "
        "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V, where d_k is the key dimension. "
        "This scaled dot-product prevents vanishing gradients in high-dimensional spaces.", B))

    story.append(Paragraph("1.2 Multi-Head Attention", SH))
    story.append(Paragraph(
        "Rather than computing a single attention function, the Transformer uses h=8 parallel attention "
        "heads with dimension d_model/h = 64 each (for d_model=512). This allows the model to jointly "
        "attend to information from different representation subspaces at different positions. "
        "The outputs of all heads are concatenated and projected back to d_model dimensions.", B))

    story.append(Paragraph("1.3 Positional Encoding", SH))
    story.append(Paragraph(
        "Since the attention mechanism is permutation-invariant, positional encodings are added to "
        "input embeddings to inject sequence order information. The original paper uses sinusoidal "
        "encodings: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). "
        "Modern models like RoPE (Rotary Position Embeddings) and ALiBi have since improved upon this.", B))

    story.append(Paragraph("1.4 Feed-Forward Networks", SH))
    story.append(Paragraph(
        "Each Transformer layer includes a position-wise feed-forward network with two linear "
        "transformations and a ReLU activation: FFN(x) = max(0, xW1 + b1)W2 + b2. "
        "The inner dimension is typically 4x the model dimension (2048 for d_model=512). "
        "Layer normalization and residual connections are applied around both sub-layers.", B))

    story.append(Paragraph("2. Training Methodology", SH))
    story.append(Paragraph(
        "Transformers are trained using the Adam optimizer with a custom learning rate schedule: "
        "lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5)). "
        "The warmup_steps value is typically 4000. Dropout is applied at rate 0.1 on attention "
        "weights and feed-forward layers. Label smoothing of 0.1 is used for regularization.", B))

    story.append(Paragraph("3. Landmark Model Families", SH))
    data = [
        ["Model", "Year", "Parameters", "Key Innovation"],
        ["BERT-Large", "2018", "340M", "Bidirectional masked LM pre-training"],
        ["GPT-3", "2020", "175B", "Few-shot in-context learning"],
        ["T5-XXL", "2020", "11B", "Text-to-text unified framework"],
        ["LLaMA-3", "2024", "70B", "Open-weight, efficient fine-tuning"],
        ["Gemini 1.5", "2024", "N/A", "1M token context window"],
    ]
    table = Table(data, colWidths=[1.2*inch, 0.8*inch, 1.1*inch, 3.1*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e7a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f0f8"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#ccccdd")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Computational Requirements", SH))
    story.append(Paragraph(
        "The self-attention mechanism has O(n^2 * d) time and space complexity with respect to "
        "sequence length n, making it expensive for very long sequences. Efficient variants like "
        "Longformer (sliding window attention), Linformer (linear approximation), and FlashAttention "
        "(IO-aware exact attention) have reduced this to near-linear complexity. A standard GPT-3 "
        "training run consumed approximately 3,640 petaflop/s-days and cost an estimated $4.6M.", B))

    story.append(Paragraph("5. Applications Beyond NLP", SH))
    story.append(Paragraph(
        "Vision Transformer (ViT) applies the Transformer to 16x16 image patches and achieves "
        "state-of-the-art results on ImageNet when pre-trained on large datasets. AlphaFold2 uses "
        "a Transformer-based architecture (Evoformer) to predict protein 3D structures with atomic "
        "accuracy, winning the CASP14 competition. Decision Transformer formulates reinforcement "
        "learning as a sequence modeling problem.", B))

    story.append(Paragraph("6. Limitations and Open Research Problems", SH))
    for item in [
        "Quadratic attention complexity limits context length without approximations.",
        "Large models require significant compute for both training and inference.",
        "Emergent capabilities are difficult to predict before training completes.",
        "Hallucination and factual grounding remain active research areas.",
        "Interpretability of attention patterns is still not well understood.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Spacer(1, 10))
    story.append(Paragraph("References", SH))
    for ref in [
        "Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.",
        "Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.",
        "Brown et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.",
        "Dosovitskiy et al. (2020). An Image is Worth 16x16 Words. ICLR 2021.",
        "Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention. NeurIPS 2022.",
    ]:
        story.append(Paragraph(f"• {ref}", BL))

    doc.build(story)
    return path


# ── Document 2: Climate Change & Agriculture ──────────────────────────────────
def create_doc2(styles):
    path = os.path.join(OUTPUT_DIR, "climate_change_agriculture_impact.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=0.9*inch, leftMargin=0.9*inch,
                            topMargin=0.9*inch, bottomMargin=0.9*inch)
    story = []
    B, S, SH, BL = styles["Body"], styles["DocSubtitle"], styles["SectionHeader"], styles["BulletItem"]

    story.append(Paragraph("Climate Change Impacts on Global Agriculture", styles["DocTitle"]))
    story.append(Paragraph("Environmental Science Policy Brief · IPCC Data Synthesis · 2024", S))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccccdd")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", SH))
    story.append(Paragraph(
        "Global agricultural systems face unprecedented challenges from accelerating climate change. "
        "This report synthesizes findings from the IPCC Sixth Assessment Report (AR6), FAO global "
        "food security analyses, and peer-reviewed literature to quantify projected impacts on crop "
        "yields, water availability, and food security through 2100. Under high-emission scenarios "
        "(SSP5-8.5), global staple crop yields could decline by 2–6% per decade while demand "
        "continues to grow at 1.3% annually, creating a structural supply-demand gap.", B))

    story.append(Paragraph("1. Temperature Effects on Crop Yields", SH))
    story.append(Paragraph(
        "Each 1°C increase in growing-season temperatures reduces global wheat yields by 6.0%, "
        "rice yields by 3.2%, maize yields by 7.4%, and soybean yields by 3.1% (Zhao et al., 2017). "
        "These relationships are non-linear: crops near their thermal optimum suffer disproportionate "
        "losses from extreme heat events. Nighttime warming is particularly damaging — rice spikelet "
        "sterility increases significantly when nighttime temperatures exceed 26°C during flowering.", B))

    story.append(Paragraph("1.1 CO₂ Fertilization Effect", SH))
    story.append(Paragraph(
        "Elevated atmospheric CO₂ (currently 422 ppm, rising at 2.4 ppm/year) stimulates "
        "photosynthesis in C3 crops (wheat, rice, soybean) through the CO₂ fertilization effect. "
        "FACE (Free-Air CO₂ Enrichment) experiments show 8–15% yield increases at 550 ppm for C3 "
        "crops. However, C4 crops (maize, sorghum, sugarcane) show minimal response (~0–5%). "
        "Critically, elevated CO₂ reduces protein and micronutrient concentrations in grains by "
        "5–10%, degrading nutritional quality even when yields increase.", B))

    story.append(Paragraph("2. Water Stress and Drought", SH))
    story.append(Paragraph(
        "Agriculture accounts for 70% of global freshwater withdrawals. Climate change is projected "
        "to intensify the global hydrological cycle: wet regions generally become wetter, dry regions "
        "drier. The Mediterranean basin, southwestern North America, southern Africa, and "
        "northeastern Brazil face 20–40% reductions in precipitation by 2100 under SSP3-7.0. "
        "Groundwater depletion is already critical: the High Plains Aquifer (US), North China Plain, "
        "and Indus Basin are being depleted at rates 3–10x natural recharge.", B))

    story.append(Paragraph("3. Regional Impact Projections", SH))
    data = [
        ["Region", "Crop", "Yield Change by 2050", "Primary Driver"],
        ["Sub-Saharan Africa", "Maize", "-20% to -40%", "Temperature + drought"],
        ["South Asia", "Rice", "-10% to -30%", "Heat stress at flowering"],
        ["North America (Great Plains)", "Wheat", "-5% to +5%", "CO₂ offset by heat"],
        ["Mediterranean", "All cereals", "-15% to -25%", "Precipitation decline"],
        ["Northern Europe", "Wheat", "+5% to +15%", "Longer growing season"],
        ["China (North)", "Maize", "-8% to -18%", "Groundwater depletion"],
    ]
    table = Table(data, colWidths=[1.6*inch, 0.9*inch, 1.4*inch, 2.3*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2d6a4f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f7f0"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b7d5c2")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Food Security Implications", SH))
    story.append(Paragraph(
        "The World Food Programme estimates that climate change could push an additional 80–130 "
        "million people into hunger by 2050 under moderate warming scenarios (2°C), with the "
        "majority in sub-Saharan Africa and South Asia. Food price volatility is expected to "
        "increase significantly: a 1°C rise in temperature is associated with a 1–3% increase in "
        "wheat price volatility globally. Low-income, import-dependent nations are most vulnerable "
        "to these price shocks as they spend 40–60% of household income on food.", B))

    story.append(Paragraph("5. Adaptation Strategies", SH))
    story.append(Paragraph(
        "The agricultural sector has several evidence-based adaptation pathways available:", B))
    for item in [
        "Breeding heat and drought tolerant crop varieties (e.g., CIMMYT's drought-tolerant maize "
          "varieties show 20–30% yield advantages under stress conditions).",
        "Shifting planting dates by 2–4 weeks earlier to avoid peak summer heat during flowering.",
        "Precision irrigation reducing water use by 30–50% while maintaining yields.",
        "Agroforestry systems that buffer microclimates and diversify farm income.",
        "Conservation agriculture (no-till, cover crops) improving soil water retention by 15–25%.",
        "Crop diversification reducing single-crop failure risk at farm and regional scale.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("6. Economic Costs", SH))
    story.append(Paragraph(
        "Global economic losses from climate-related agricultural disruptions are projected at "
        "$700 billion–$1.2 trillion annually by 2050 (2020 USD). Adaptation investments of "
        "$70–100 billion/year through 2030 could offset losses of $300–500 billion/year — a "
        "benefit-cost ratio of 3–5:1. Current global adaptation finance flows to agriculture "
        "are estimated at only $4.3 billion/year, representing a significant financing gap.", B))

    story.append(Paragraph("References", SH))
    for ref in [
        "IPCC (2022). Sixth Assessment Report, Working Group II: Impacts, Adaptation and Vulnerability.",
        "Zhao et al. (2017). Temperature increase reduces global yields of major crops. Nature Plants.",
        "FAO (2023). The State of Food Security and Nutrition in the World 2023.",
        "Myers et al. (2017). Climate Change and Global Food Systems. Annual Review of Public Health.",
        "WFP (2024). Climate and Food Crises: Global Report on Food Crises 2024.",
    ]:
        story.append(Paragraph(f"• {ref}", BL))

    doc.build(story)
    return path


# ── Document 3: Quantum Computing ─────────────────────────────────────────────
def create_doc3(styles):
    path = os.path.join(OUTPUT_DIR, "quantum_computing_fundamentals.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=0.9*inch, leftMargin=0.9*inch,
                            topMargin=0.9*inch, bottomMargin=0.9*inch)
    story = []
    B, S, SH, BL = styles["Body"], styles["DocSubtitle"], styles["SectionHeader"], styles["BulletItem"]

    story.append(Paragraph("Quantum Computing: Fundamentals and Near-Term Applications", styles["DocTitle"]))
    story.append(Paragraph("Technical Overview · Quantum Information Science · 2024", S))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccccdd")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Abstract", SH))
    story.append(Paragraph(
        "Quantum computing exploits quantum mechanical phenomena — superposition, entanglement, and "
        "interference — to perform computations that are intractable for classical computers. "
        "This document covers the theoretical foundations of quantum computing, current hardware "
        "platforms, the concept of quantum advantage, near-term NISQ (Noisy Intermediate-Scale "
        "Quantum) applications, and the timeline toward fault-tolerant quantum computation. "
        "We analyze Google's 2023 quantum supremacy demonstrations, IBM's 433-qubit Osprey "
        "processor, and the practical roadmap for quantum advantage in optimization and chemistry.", B))

    story.append(Paragraph("1. Quantum Mechanical Foundations", SH))
    story.append(Paragraph("1.1 Qubits and Superposition", SH))
    story.append(Paragraph(
        "Classical computers use bits that are either 0 or 1. A qubit (quantum bit) can exist in "
        "a superposition of both states simultaneously: |ψ⟩ = α|0⟩ + β|1⟩, where α and β are "
        "complex probability amplitudes satisfying |α|² + |β|² = 1. Upon measurement, the qubit "
        "collapses to |0⟩ with probability |α|² or |1⟩ with probability |β|². A register of n "
        "qubits can represent 2ⁿ states simultaneously, enabling exponential parallelism in "
        "specific computational tasks.", B))

    story.append(Paragraph("1.2 Quantum Entanglement", SH))
    story.append(Paragraph(
        "Entanglement is a uniquely quantum correlation between qubits: measuring one instantaneously "
        "determines the state of its entangled partners, regardless of physical separation. "
        "A maximally entangled two-qubit state (Bell state) is: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2. "
        "Entanglement is the key resource enabling quantum teleportation, superdense coding, and "
        "quantum error correction. IBM's 2023 error mitigation experiments demonstrated that "
        "entangled circuits with 127 qubits could exceed classical simulation capabilities.", B))

    story.append(Paragraph("1.3 Quantum Interference", SH))
    story.append(Paragraph(
        "Quantum algorithms are designed so that incorrect answer paths interfere destructively "
        "(canceling out) while correct answer paths interfere constructively (amplifying). "
        "Grover's algorithm uses amplitude amplification to search an unsorted database of N "
        "items in O(√N) time vs. O(N) classically — a quadratic speedup. Shor's algorithm uses "
        "the quantum Fourier transform to factor N-bit integers in O(N³) quantum operations vs. "
        "sub-exponential classical algorithms, threatening RSA-2048 encryption.", B))

    story.append(Paragraph("2. Hardware Platforms", SH))
    data = [
        ["Technology", "Company", "Best Qubit Count", "Coherence Time", "Gate Fidelity"],
        ["Superconducting", "IBM / Google", "1000+ (IBM Condor)", "~100 μs", "99.5%"],
        ["Trapped Ion", "IonQ / Quantinuum", "32 (IonQ Aria)", "minutes", "99.9%"],
        ["Photonic", "PsiQuantum", "Prototype", "N/A (room temp)", "~99%"],
        ["Neutral Atom", "QuEra / Pasqal", "256 (QuEra)", "seconds", "99.5%"],
        ["Topological", "Microsoft", "Research phase", "TBD", "TBD"],
    ]
    table = Table(data, colWidths=[1.2*inch, 1.3*inch, 1.2*inch, 1.0*inch, 1.1*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4a1d8a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f4f0f8"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#c9b8e8")),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Quantum Algorithms and Speedups", SH))
    story.append(Paragraph(
        "Not all problems benefit from quantum computing. Proven quantum speedups exist for:", B))
    for item in [
        "Integer factorization: Shor's algorithm — exponential speedup (threatens public-key cryptography).",
        "Unstructured search: Grover's algorithm — quadratic speedup.",
        "Quantum simulation: Exact simulation of molecular systems — exponential speedup for chemistry.",
        "Linear systems: HHL algorithm — exponential speedup under certain sparsity conditions.",
        "Optimization: QAOA (Quantum Approximate Optimization Algorithm) — near-term heuristic speedups.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("4. NISQ Era and Current Limitations", SH))
    story.append(Paragraph(
        "We are currently in the NISQ (Noisy Intermediate-Scale Quantum) era — systems with "
        "50–1000 qubits that are too noisy for full error correction. Physical qubits have error "
        "rates of 0.1–1% per gate, meaning circuits deeper than ~100 gates accumulate too much "
        "noise to be reliable. Achieving fault-tolerant quantum computing requires logical qubits "
        "built from ~1000 physical qubits each (surface code), implying millions of physical qubits "
        "for practically useful fault-tolerant algorithms. Current best estimates place this "
        "milestone between 2030–2040.", B))

    story.append(Paragraph("5. Near-Term Applications", SH))
    story.append(Paragraph(
        "Despite NISQ limitations, several application domains show promise before full error "
        "correction is achieved:", B))
    for item in [
        "Quantum chemistry: Variational Quantum Eigensolver (VQE) for drug discovery and catalyst design. "
          "Accurately simulating the FeMo-cofactor of nitrogenase could transform nitrogen fixation.",
        "Financial optimization: Portfolio optimization and risk analysis using QAOA on 100–1000 asset universes.",
        "Machine learning: Quantum kernels and quantum-enhanced feature spaces (potential speedup unconfirmed).",
        "Cryptography: Post-quantum cryptographic standards (NIST finalized CRYSTALS-Kyber and CRYSTALS-Dilithium in 2024).",
        "Logistics: Vehicle routing and supply chain optimization as a near-term QAOA target.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("6. The Path to Quantum Advantage", SH))
    story.append(Paragraph(
        "Quantum advantage (outperforming all classical computers on a practically useful task) "
        "has not yet been conclusively demonstrated. Google's 2019 'quantum supremacy' claim "
        "(Sycamore processor, 53 qubits, random circuit sampling) was subsequently challenged by "
        "improved classical algorithms. IBM's goal is to demonstrate 'quantum utility' — "
        "meaningful quantum advantage on real-world problems — by 2026. The global quantum "
        "computing market is projected to reach $65 billion by 2030 (McKinsey, 2023), "
        "attracting over $35 billion in public and private investment since 2020.", B))

    story.append(Paragraph("References", SH))
    for ref in [
        "Arute et al. (2019). Quantum supremacy using a programmable superconducting processor. Nature.",
        "Preskill, J. (2018). Quantum Computing in the NISQ Era and Beyond. Quantum.",
        "NIST (2024). Post-Quantum Cryptography Standardization: Final Standards.",
        "IBM Quantum (2023). IBM Quantum System Two and the Path to 100,000 Qubits.",
        "McKinsey & Company (2023). Quantum Technology Monitor 2023.",
    ]:
        story.append(Paragraph(f"• {ref}", BL))

    doc.build(story)
    return path


# ── Document 4: Large Language Models & RAG ───────────────────────────────────
def create_doc4(styles):
    path = os.path.join(OUTPUT_DIR, "large_language_models_and_rag.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=0.9*inch, leftMargin=0.9*inch,
                            topMargin=0.9*inch, bottomMargin=0.9*inch)
    story = []
    B, S, SH, BL = styles["Body"], styles["DocSubtitle"], styles["SectionHeader"], styles["BulletItem"]

    story.append(Paragraph("Large Language Models and Retrieval-Augmented Generation", styles["DocTitle"]))
    story.append(Paragraph("Applied NLP Research Report · Information Retrieval Series · 2024", S))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccccdd")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Abstract", SH))
    story.append(Paragraph(
        "Large Language Models (LLMs) have demonstrated remarkable capabilities in language "
        "understanding and generation, yet they suffer from critical limitations: static knowledge "
        "cutoffs, hallucination of plausible-sounding but incorrect facts, and inability to access "
        "private or proprietary data. Retrieval-Augmented Generation (RAG) addresses these "
        "limitations by grounding LLM responses in dynamically retrieved external documents. "
        "This report covers RAG architecture, chunking strategies, embedding models, vector "
        "databases, retrieval algorithms, and evaluation methodologies for production RAG systems.", B))

    story.append(Paragraph("1. LLM Limitations That RAG Addresses", SH))
    story.append(Paragraph(
        "Despite their impressive performance, LLMs have well-documented failure modes that limit "
        "their deployment in knowledge-intensive applications:", B))
    for item in [
        "Knowledge cutoff: GPT-4's training data ends in April 2023; it cannot answer questions "
          "about events after that date without external augmentation.",
        "Hallucination: LLMs generate confident, grammatically correct responses that are factually "
          "wrong. Studies estimate hallucination rates of 3–27% depending on domain and model.",
        "Context window limits: Even with 128K token windows, LLMs cannot hold entire document "
          "repositories in context simultaneously.",
        "No access to private data: Enterprise knowledge bases, internal documents, and proprietary "
          "research are not part of public training corpora.",
        "Lack of citations: Base LLMs cannot attribute claims to specific source documents, "
          "making verification and trust-building difficult.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("2. RAG Architecture Overview", SH))
    story.append(Paragraph(
        "A RAG system consists of two main phases: offline indexing and online retrieval-generation. "
        "During indexing, source documents are chunked, embedded into dense vectors, and stored in "
        "a vector database. At query time, the user's question is embedded with the same model, "
        "the vector database is searched for semantically similar chunks (top-k retrieval), and "
        "the retrieved chunks are injected into the LLM prompt as context. The LLM then generates "
        "an answer grounded in the retrieved content.", B))

    story.append(Paragraph("2.1 Chunking Strategies", SH))
    story.append(Paragraph(
        "Document chunking critically impacts retrieval quality. Common strategies include:", B))
    for item in [
        "Fixed-size chunking: Split every N characters regardless of sentence boundaries. "
          "Simple but often breaks semantic units. Typical size: 512–1024 tokens.",
        "Recursive character splitting: Split on paragraph → sentence → word boundaries in order. "
          "Preserves semantic coherence. Recommended chunk size: 800 chars with 150-char overlap.",
        "Semantic chunking: Use embedding similarity to detect topic boundaries and split there. "
          "Best quality but 3–5x more expensive due to embedding each sentence.",
        "Hierarchical chunking (parent-child): Store large parent chunks for context, "
          "index smaller child chunks for precision. Retrieves child, returns parent to LLM.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("2.2 Embedding Models", SH))
    data = [
        ["Model", "Provider", "Dimensions", "Max Tokens", "Best For"],
        ["text-embedding-3-large", "OpenAI", "3072", "8191", "General English"],
        ["gemini-embedding-001", "Google", "3072", "2048", "Multilingual"],
        ["e5-large-v2", "Microsoft", "1024", "512", "Open-source"],
        ["bge-m3", "BAAI", "1024", "8192", "Multilingual OSS"],
        ["nomic-embed-text", "Nomic", "768", "8192", "Local deployment"],
    ]
    table = Table(data, colWidths=[1.6*inch, 1.0*inch, 0.9*inch, 0.9*inch, 1.8*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6b3a8a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f5f0f8"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#c9b8e8")),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Retrieval Algorithms", SH))
    story.append(Paragraph(
        "The retrieval step determines which chunks are passed to the LLM. Several strategies "
        "exist with different tradeoffs between precision and recall:", B))
    for item in [
        "Dense retrieval (cosine similarity): Embed query, find nearest vectors. Fast (ANN search "
          "in sub-millisecond for millions of vectors) but misses exact keyword matches.",
        "Sparse retrieval (BM25): TF-IDF variant capturing exact term frequency. Excellent for "
          "technical jargon and proper nouns but misses paraphrases.",
        "Hybrid retrieval: Combine dense + sparse scores using Reciprocal Rank Fusion (RRF). "
          "Consistently outperforms either method alone by 5–15% on BEIR benchmarks.",
        "MMR (Maximal Marginal Relevance): Selects diverse top-k results to prevent returning "
          "5 copies of the same paragraph. Uses lambda parameter to balance relevance vs. diversity.",
        "Multi-query retrieval: Generate 3–5 query paraphrases via LLM, retrieve for each, "
          "deduplicate. Improves recall by 20–35% for paraphrased or implicit questions.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("4. Advanced RAG Techniques", SH))
    story.append(Paragraph(
        "Production RAG systems go beyond basic retrieve-and-read with several enhancements:", B))
    for item in [
        "Re-ranking: After initial retrieval, apply a cross-encoder (e.g., ms-marco-MiniLM) to "
          "re-score candidates. Improves precision by 10–20% at the cost of ~50ms latency.",
        "HyDE (Hypothetical Document Embeddings): Generate a hypothetical answer first, embed it, "
          "use that embedding for retrieval. Bridges vocabulary gap between query and documents.",
        "FLARE (Forward-Looking Active Retrieval): Trigger retrieval only when model uncertainty "
          "is high (low token probabilities), reducing unnecessary retrieval calls.",
        "Self-RAG: Train the LLM to decide when to retrieve, critique its own outputs, and "
          "selectively use retrieved passages — all in a single model.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("5. Evaluation Metrics", SH))
    story.append(Paragraph(
        "RAG systems require evaluation at both the retrieval and generation stages. "
        "The RAGAS framework (Es et al., 2023) defines four key metrics:", B))
    for item in [
        "Faithfulness: Fraction of answer claims that are supported by retrieved context. "
          "Measures hallucination rate. Target: >0.90.",
        "Answer Relevancy: Semantic similarity between answer and question (ignoring context). "
          "Measures whether the answer addresses what was asked. Target: >0.85.",
        "Context Precision: Fraction of retrieved chunks that are actually relevant to the question. "
          "Measures retrieval noise. Target: >0.80.",
        "Context Recall: Fraction of ground-truth information covered by retrieved chunks. "
          "Measures retrieval completeness. Target: >0.75.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("References", SH))
    for ref in [
        "Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.",
        "Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024.",
        "Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey.",
        "Ma et al. (2023). Query Rewriting for Retrieval-Augmented Large Language Models. EMNLP.",
        "Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique. ICLR 2024.",
    ]:
        story.append(Paragraph(f"• {ref}", BL))

    doc.build(story)
    return path


# ── Document 5: Cybersecurity & Zero Trust ────────────────────────────────────
def create_doc5(styles):
    path = os.path.join(OUTPUT_DIR, "cybersecurity_zero_trust_framework.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=0.9*inch, leftMargin=0.9*inch,
                            topMargin=0.9*inch, bottomMargin=0.9*inch)
    story = []
    B, S, SH, BL = styles["Body"], styles["DocSubtitle"], styles["SectionHeader"], styles["BulletItem"]

    story.append(Paragraph("Cybersecurity and the Zero Trust Architecture Framework", styles["DocTitle"]))
    story.append(Paragraph("Enterprise Security Policy White Paper · NIST SP 800-207 Analysis · 2024", S))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccccdd")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", SH))
    story.append(Paragraph(
        "The traditional perimeter-based security model — trust everything inside the network, "
        "distrust everything outside — has collapsed under the weight of cloud adoption, remote "
        "work, and sophisticated supply chain attacks. Zero Trust Architecture (ZTA), formalized "
        "in NIST Special Publication 800-207 (2020), replaces implicit trust with continuous "
        "verification: 'never trust, always verify.' This white paper analyzes the Zero Trust "
        "principles, implementation pillars, maturity model, and measured outcomes from "
        "enterprise deployments, alongside the 2024 threat landscape that makes ZTA adoption urgent.", B))

    story.append(Paragraph("1. The 2024 Cybersecurity Threat Landscape", SH))
    story.append(Paragraph(
        "Cybercrime costs reached an estimated $8 trillion globally in 2023 and are projected to "
        "hit $10.5 trillion annually by 2025 (Cybersecurity Ventures). Key threat vectors include:", B))
    for item in [
        "Ransomware: Average ransom payment reached $1.54 million in 2023 (Sophos State of "
          "Ransomware). Healthcare, manufacturing, and critical infrastructure are top targets.",
        "Supply chain attacks: The SolarWinds (2020) and XZ Utils (2024) incidents demonstrated "
          "that trusted software can be weaponized, bypassing traditional perimeter defenses.",
        "Identity-based attacks: 80% of breaches involve compromised credentials (Verizon DBIR 2024). "
          "Credential stuffing, phishing, and MFA fatigue attacks are dominant vectors.",
        "AI-assisted attacks: LLMs enable highly personalized phishing at scale, automated "
          "vulnerability discovery, and deepfake-based social engineering attacks.",
        "Cloud misconfigurations: 82% of data breaches involve cloud assets, with misconfiguration "
          "as the leading cause (IBM Cost of Data Breach Report 2024).",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("2. Zero Trust Core Principles", SH))
    story.append(Paragraph(
        "NIST SP 800-207 defines Zero Trust around seven core tenets:", B))
    for item in [
        "All data sources and computing services are considered resources, regardless of location.",
        "All communication is secured regardless of network location (no implicit trust for internal traffic).",
        "Access to individual enterprise resources is granted on a per-session basis.",
        "Access to resources is determined by dynamic policy including observable client identity, "
          "application, and the requesting asset's state and behavior.",
        "The enterprise monitors and measures the integrity and security posture of all owned and "
          "associated assets.",
        "All resource authentication and authorization are dynamic and strictly enforced before "
          "access is allowed.",
        "The enterprise collects as much information as possible about the current state of assets, "
          "network infrastructure, and communications, and uses it to improve its security posture.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("3. The Five Pillars of Zero Trust", SH))
    data = [
        ["Pillar", "Core Capability", "Key Technologies"],
        ["Identity", "Verify every user and device", "MFA, SSO, PAM, Identity Governance"],
        ["Device", "Validate endpoint health", "EDR, MDM, Compliance Policies"],
        ["Network", "Micro-segment traffic", "ZTNA, SD-WAN, DNS Filtering"],
        ["Application", "Secure app access", "CASB, WAF, API Gateway"],
        ["Data", "Classify and protect data", "DLP, Encryption, DSPM"],
    ]
    table = Table(data, colWidths=[1.1*inch, 1.9*inch, 3.2*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#8b1a1a")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#fdf0f0"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e0b8b8")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. CISA Zero Trust Maturity Model", SH))
    story.append(Paragraph(
        "CISA's Zero Trust Maturity Model (ZTMM v2.0, 2023) defines four maturity stages across "
        "the five pillars: Traditional, Initial, Advanced, and Optimal. Organizations at the "
        "Traditional stage rely on static policies and perimeter controls. Initial adopters have "
        "implemented MFA and basic segmentation. Advanced organizations use risk-adaptive access "
        "policies and automated response. Optimal organizations have fully automated, continuously "
        "validated Zero Trust controls across all pillars with real-time threat integration.", B))

    story.append(Paragraph("5. Implementation Roadmap", SH))
    story.append(Paragraph(
        "A phased Zero Trust implementation typically follows this sequence:", B))
    for item in [
        "Phase 1 — Identity foundation (months 1–3): Deploy MFA organization-wide, implement "
          "Privileged Access Management (PAM), establish Identity Governance. Cost: $50K–$500K.",
        "Phase 2 — Device compliance (months 3–6): Enroll all endpoints in MDM, deploy EDR, "
          "enforce health checks as access conditions. Block non-compliant devices.",
        "Phase 3 — Network micro-segmentation (months 6–12): Replace VPN with ZTNA, implement "
          "east-west traffic inspection, deploy DNS-layer security.",
        "Phase 4 — Application access control (months 9–15): Instrument all apps with SSO, "
          "deploy CASB for SaaS, implement API gateway with per-request authorization.",
        "Phase 5 — Data protection (months 12–18): Classify all sensitive data, deploy DLP, "
          "enforce encryption at rest and in transit with managed key lifecycle.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("6. Measured Outcomes from Enterprise Deployments", SH))
    story.append(Paragraph(
        "Microsoft's internal Zero Trust implementation (Project Armada, 2019–2022) across "
        "220,000 employees reported: 50% reduction in security incidents, 90% reduction in "
        "lateral movement incidents, 75% faster incident response time, and $1.2 billion "
        "estimated savings from breach cost avoidance. Google's BeyondCorp initiative, the "
        "original Zero Trust implementation, enabled all employees to work securely from "
        "untrusted networks — a foundation for their successful pandemic response in 2020.", B))

    story.append(Paragraph("7. Common Implementation Challenges", SH))
    for item in [
        "Legacy application compatibility: Applications that rely on implicit network trust "
          "require re-architecting or wrapping with a proxy layer.",
        "User experience impact: Excessive authentication prompts increase friction and drive "
          "shadow IT. Adaptive authentication (step-up MFA) must balance security and UX.",
        "Visibility gaps: 43% of organizations report incomplete asset inventory, making "
          "comprehensive Zero Trust impossible without proper CMDB and discovery tooling.",
        "Organizational resistance: Zero Trust requires cross-functional coordination between "
          "security, IT, networking, and application teams — breaking traditional silos.",
    ]:
        story.append(Paragraph(f"• {item}", BL))

    story.append(Paragraph("References", SH))
    for ref in [
        "NIST SP 800-207 (2020). Zero Trust Architecture. National Institute of Standards and Technology.",
        "CISA (2023). Zero Trust Maturity Model Version 2.0.",
        "Verizon (2024). Data Breach Investigations Report 2024.",
        "IBM Security (2024). Cost of a Data Breach Report 2024.",
        "Cybersecurity Ventures (2024). Cybercrime Report 2024.",
    ]:
        story.append(Paragraph(f"• {ref}", BL))

    doc.build(story)
    return path


if __name__ == "__main__":
    styles = make_styles()
    docs = [
        ("transformer_architecture_overview.pdf", create_doc1(styles)),
        ("climate_change_agriculture_impact.pdf", create_doc2(styles)),
        ("quantum_computing_fundamentals.pdf", create_doc3(styles)),
        ("large_language_models_and_rag.pdf", create_doc4(styles)),
        ("cybersecurity_zero_trust_framework.pdf", create_doc5(styles)),
    ]
    print("Generated test documents:")
    for name, path in docs:
        size = os.path.getsize(path) / 1024
        print(f"  {path}  ({size:.1f} KB)")
