import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = path.resolve("outputs");
const OUT_FILE = path.join(OUT_DIR, "finqa_presentation.pptx");

const COLORS = {
  bg: "#F4F1EA",
  surface: "#FFFDF9",
  ink: "#132A3A",
  muted: "#4D6473",
  accent: "#B65A3A",
  accent2: "#2D7A72",
  line: "#D8CFC3",
};

const FONT = {
  title: "Poppins",
  body: "Lato",
};

function addTextBlock(slide, {
  left,
  top,
  width,
  height,
  text,
  fontSize = 24,
  color = COLORS.ink,
  bold = false,
  typeface = FONT.body,
  fill = null,
  radius = false,
}) {
  const shape = slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill: fill ?? { color: "#000000", transparency: 100000 },
    line: { width: 0, fill: "#00000000" },
  });
  shape.text = text;
  shape.text.fontSize = fontSize;
  shape.text.color = color;
  shape.text.bold = bold;
  shape.text.typeface = typeface;
  shape.text.insets = { left: 18, right: 18, top: 12, bottom: 12 };
  return shape;
}

function addSlideTitle(slide, eyebrow, title, subtitle = "") {
  addTextBlock(slide, {
    left: 72,
    top: 50,
    width: 1140,
    height: 40,
    text: eyebrow.toUpperCase(),
    fontSize: 16,
    color: COLORS.accent,
    bold: true,
    typeface: FONT.body,
  });
  addTextBlock(slide, {
    left: 72,
    top: 86,
    width: 1140,
    height: 86,
    text: title,
    fontSize: 34,
    color: COLORS.ink,
    bold: true,
    typeface: FONT.title,
  });
  if (subtitle) {
    addTextBlock(slide, {
      left: 72,
      top: 162,
      width: 980,
      height: 48,
      text: subtitle,
      fontSize: 18,
      color: COLORS.muted,
      typeface: FONT.body,
    });
  }
}

function addBulletCard(slide, left, top, width, title, bullets) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height: 240 },
    fill: COLORS.surface,
    line: { width: 1, fill: COLORS.line },
  });
  addTextBlock(slide, {
    left: left + 18,
    top: top + 14,
    width: width - 36,
    height: 40,
    text: title,
    fontSize: 20,
    color: COLORS.ink,
    bold: true,
    typeface: FONT.title,
  });
  addTextBlock(slide, {
    left: left + 18,
    top: top + 52,
    width: width - 36,
    height: 170,
    text: bullets.map((item) => `• ${item}`).join("\n"),
    fontSize: 17,
    color: COLORS.ink,
    typeface: FONT.body,
  });
}

function addSectionSlide(presentation, eyebrow, title, subtitle, columns) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addSlideTitle(slide, eyebrow, title, subtitle);
  let left = 72;
  for (const column of columns) {
    addBulletCard(slide, left, 240, 360, column.title, column.bullets);
    left += 388;
  }
  return slide;
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });

  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  presentation.theme.colorScheme = {
    name: "FinQA",
    themeColors: {
      accent1: COLORS.accent,
      accent2: COLORS.accent2,
      bg1: COLORS.bg,
      bg2: COLORS.surface,
      tx1: COLORS.ink,
      tx2: COLORS.muted,
    },
  };

  const title = presentation.slides.add();
  title.background.fill = COLORS.bg;
  title.shapes.add({
    geometry: "roundRect",
    position: { left: 56, top: 56, width: 1168, height: 608 },
    fill: COLORS.surface,
    line: { width: 1, fill: COLORS.line },
  });
  addTextBlock(title, {
    left: 92,
    top: 110,
    width: 760,
    height: 40,
    text: "RUNARA.AI TAKE-HOME",
    fontSize: 18,
    color: COLORS.accent,
    bold: true,
  });
  addTextBlock(title, {
    left: 92,
    top: 150,
    width: 760,
    height: 150,
    text: "FinQA Numerical Reasoning Chatbot",
    fontSize: 42,
    color: COLORS.ink,
    bold: true,
    typeface: FONT.title,
  });
  addTextBlock(title, {
    left: 92,
    top: 300,
    width: 720,
    height: 120,
    text: "LangGraph orchestration, LangChain model integration, and vLLM-backed Hugging Face serving for financial question answering over tables and narrative evidence.",
    fontSize: 20,
    color: COLORS.muted,
  });
  addBulletCard(title, 860, 118, 290, "Key themes", [
    "Financial QA is grounded numerical reasoning",
    "Document-local retrieval fixes the main baseline flaw",
    "Evaluation must split retrieval, reasoning, and answer quality",
  ]);
  title.speakerNotes.setText("Open with the business framing, then state the core architectural correction: retrieve from the active document, not from unrelated train QA pairs.");

  addSectionSlide(
    presentation,
    "Problem",
    "Why FinQA needs a specialized QA design",
    "The benchmark mixes long financial prose, tables, and executable arithmetic programs.",
    [
      {
        title: "Dataset facts",
        bullets: [
          "6,251 train examples and 883 validation examples",
          "Average context length is about 4,000 characters",
          "Average table has 5.3 data rows",
        ],
      },
      {
        title: "Reasoning profile",
        bullets: [
          "Gold programs are dominated by divide and subtract",
          "Units matter: percent, millions, billions",
          "Errors usually come from operand selection, not wording",
        ],
      },
      {
        title: "Business risk",
        bullets: [
          "Hallucinated values can still sound plausible",
          "Wrong arithmetic destroys trust quickly",
          "Answers need evidence and auditability",
        ],
      },
    ],
  ).speakerNotes.setText("Use this slide to distinguish FinQA from generic QA. The main point is that explicit numerical reasoning is the task, not an edge case.");

  addSectionSlide(
    presentation,
    "Correction",
    "The main repository issue was grounding",
    "The previous retrieval path searched other train questions instead of the source document being asked about.",
    [
      {
        title: "What was wrong",
        bullets: [
          "Retrieving train QA examples does not recover the target evidence",
          "It can leak patterns without grounding the answer",
          "It makes evaluation harder to trust",
        ],
      },
      {
        title: "What changed",
        bullets: [
          "Chunk the current document into pre-text, table rows, and post-text",
          "Retrieve evidence only from those chunks",
          "Keep train data for future few-shot use, not factual lookup",
        ],
      },
      {
        title: "Impact",
        bullets: [
          "Cleaner FinQA methodology",
          "Better observability and attribution",
          "More credible path to production",
        ],
      },
    ],
  ).speakerNotes.setText("Spend extra time here. This is the strongest evidence of critical thinking in the submission.");

  addSectionSlide(
    presentation,
    "Architecture",
    "Final pipeline",
    "LangGraph coordinates evidence retrieval, reasoning, arithmetic, verification, and output formatting.",
    [
      {
        title: "Serving stack",
        bullets: [
          "Hugging Face model served through vLLM",
          "LangChain ChatOpenAI wrapper targets the vLLM endpoint",
          "Same code works with local or remote OpenAI-compatible serving",
        ],
      },
      {
        title: "Graph nodes",
        bullets: [
          "retrieve -> reason -> calculate -> verify -> answer",
          "Retry loop triggers on verification failure",
          "Node traces are logged for latency and debugging",
        ],
      },
      {
        title: "Reasoning controls",
        bullets: [
          "Reason node emits evidence IDs and arithmetic expression",
          "Calculator executes safely with sympy fallback",
          "Verifier combines LLM review with deterministic checks",
        ],
      },
    ],
  ).speakerNotes.setText("Walk the audience through why LangGraph is justified here: explicit retries, node-level metrics, and clean extensibility.");

  addSectionSlide(
    presentation,
    "Model",
    "Why Qwen2.5-7B-Instruct",
    "The default model is chosen as the practical open-source baseline for GPU-backed deployment.",
    [
      {
        title: "Choice rationale",
        bullets: [
          "Broad HF and vLLM support",
          "Reasonable quality / latency tradeoff",
          "Fits a single L4 or A10G class GPU",
        ],
      },
      {
        title: "Alternatives considered",
        bullets: [
          "Tiny 1.5B models are too weak for reliable numerical reasoning",
          "Larger 32B models may improve quality but cost more",
          "AWQ variant is the memory-optimized path",
        ],
      },
      {
        title: "Provisioning",
        bullets: [
          "Recommended flavors: l4x1 or a10g-large",
          "Serve with vLLM OpenAI-compatible endpoint",
          "Switching endpoints only requires .env changes",
        ],
      },
    ],
  ).speakerNotes.setText("Frame 7B as the operational baseline, not the theoretical maximum.");

  addSectionSlide(
    presentation,
    "Evaluation",
    "How the system should be judged",
    "A single accuracy number is not enough for FinQA.",
    [
      {
        title: "Answer quality",
        bullets: [
          "Exact match",
          "Numeric match",
          "Tolerance match within 1 percent",
        ],
      },
      {
        title: "Reasoning quality",
        bullets: [
          "Calculator precision and recall",
          "Operator match against gold programs",
          "Verification pass / fail / uncertain rates",
        ],
      },
      {
        title: "Operational view",
        bullets: [
          "Latency and retry count",
          "Retrieval quality at top-k",
          "Drift in operator mix and retrieval scores",
        ],
      },
    ],
  ).speakerNotes.setText("This slide should set up the evaluation story even if benchmark runs happen later in the target GPU environment.");

  addSectionSlide(
    presentation,
    "Production",
    "Monitoring and maintenance plan",
    "The repo includes monitoring scaffolding and a concrete maintenance roadmap.",
    [
      {
        title: "Monitoring",
        bullets: [
          "Prometheus and Grafana stack in docker-compose",
          "Latency, GPU memory, verification outcomes, retries",
          "Structured logs for per-node debugging",
        ],
      },
      {
        title: "Drift detection",
        bullets: [
          "Track distribution shifts in questions and operators",
          "Alert on retrieval-score collapse",
          "Escalate when tolerance match rate drops materially",
        ],
      },
      {
        title: "Next improvements",
        bullets: [
          "Benchmark 7B vs AWQ vs larger models",
          "Add stronger row-level retrieval evaluation",
          "Consider fine-tuning for program prediction later",
        ],
      },
    ],
  ).speakerNotes.setText("Close by showing the project is operationally thought through, not just a notebook demo.");

  const finalSlide = presentation.slides.add();
  finalSlide.background.fill = COLORS.bg;
  addSlideTitle(
    finalSlide,
    "Summary",
    "What this project delivers",
    "A corrected FinQA pipeline that is more methodologically sound, more inspectable, and ready for GPU benchmarking."
  );
  addBulletCard(finalSlide, 72, 240, 540, "Delivered now", [
    "Reworked implementation in Python",
    "Technical report and presentation materials",
    "GPU-serving plan around vLLM and Hugging Face",
  ]);
  addBulletCard(finalSlide, 668, 240, 540, "Final submission step", [
    "Run the evaluation harness against a live vLLM endpoint",
    "Record 100-example and full-validation metrics",
    "Use the deck to present the design and tradeoffs",
  ]);
  finalSlide.speakerNotes.setText("End with the practical handoff: the repo is ready, and the remaining benchmark execution belongs in the target GPU environment.");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT_FILE);
  console.log(OUT_FILE);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
