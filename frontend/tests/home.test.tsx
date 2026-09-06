import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";

import Home from "../app/page";
import { demoRuns } from "../demo";

describe("Home", () => {
  it("renders the research workflow shell", () => {
    render(<Home />);
    expect(screen.getByText("ResearchPilot")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start research/i })).toBeInTheDocument();
    expect(screen.getByText(/multi-agent research co-pilot/i)).toBeInTheDocument();
  });

  it("shows a recorded run on first paint, with no request to the backend", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<Home />);

    // The page ships its own data, so the first screen is a full report.
    expect(screen.getByText(demoRuns[0].report.related_work_markdown.split("\n")[0].replace(/^#+\s*/, ""))).toBeInTheDocument();
    // Passive history/config loads are the only calls, and they are optional.
    for (const call of fetchSpy.mock.calls) {
      expect(String(call[0])).not.toContain("/research");
    }
  });

  it("switches the displayed report when another recorded run is picked", () => {
    render(<Home />);
    const target = demoRuns[1];

    fireEvent.click(screen.getByRole("button", { name: target.question }));

    expect(screen.getByRole("textbox", { name: /research question/i })).toHaveValue(target.question);
    expect(screen.getAllByText(target.question).length).toBeGreaterThan(0);
  });

  it("reports which sources contributed and which one failed", () => {
    render(<Home />);
    const panel = screen.getByText("Retrieval").parentElement as HTMLElement;

    expect(within(panel).getByText("arXiv")).toBeInTheDocument();
    expect(within(panel).getByText("OpenAlex")).toBeInTheDocument();
    expect(within(panel).getByText(/Semantic Scholar unavailable/)).toBeInTheDocument();
    expect(
      within(panel).getByText(/queried in parallel and each one is allowed to fail/i),
    ).toBeInTheDocument();
  });

  it("marks the selected recorded run for assistive technology", () => {
    render(<Home />);

    expect(screen.getByRole("button", { name: demoRuns[0].question })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: demoRuns[1].question })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});

describe("backend configuration", () => {
  it("makes no backend request when no API base is configured", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<Home />);

    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("explains why Start Research does nothing without a backend", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<Home />);

    fireEvent.click(screen.getByRole("button", { name: /start research/i }));

    expect(await screen.findByText(/No backend configured/i)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("refuses a plain http backend from an https page", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<Home />);

    fireEvent.click(screen.getByRole("button", { name: /run settings/i }));
    fireEvent.change(screen.getByRole("textbox", { name: /researchpilot api base/i }), {
      target: { value: "http://example.com:8000" },
    });
    fireEvent.click(screen.getByRole("button", { name: /start research/i }));

    // Shown twice on purpose: inline under the field as guidance, and as the
    // error banner once Start Research is pressed.
    expect(await screen.findAllByText(/browser will block a plain http backend/i)).toHaveLength(2);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("allows a plain http localhost backend from an https page", () => {
    render(<Home />);

    fireEvent.click(screen.getByRole("button", { name: /run settings/i }));
    fireEvent.change(screen.getByRole("textbox", { name: /researchpilot api base/i }), {
      target: { value: "http://localhost:8000" },
    });

    expect(screen.queryByText(/browser will block a plain http backend/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Live runs, saved history and report search/i)).toBeInTheDocument();
  });
});

describe("citation", () => {
  it("links the paper the artifact accompanies", () => {
    render(<Home />);

    // The site had no link to the paper at all, which was the largest gap for
    // a reader who wanted to cite the work.
    expect(screen.getByRole("link", { name: /^Abstract$/ })).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2603.14629",
    );
    expect(screen.getByRole("link", { name: /^PDF$/ })).toHaveAttribute(
      "href",
      "https://arxiv.org/pdf/2603.14629",
    );
  });

  it("offers a BibTeX entry that is on screen, not only on the clipboard", () => {
    render(<Home />);

    // Clipboard access can be refused, so the entry has to be selectable too.
    expect(screen.getByText(/@misc\{zhang2026researchpilot/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /copy bibtex/i })).toBeInTheDocument();
  });

  it("keeps the paper below the recorded runs rather than above them", () => {
    render(<Home />);

    const runs = document.querySelector("#recorded-runs") as HTMLElement;
    const paper = screen.getByRole("heading", { name: /^Cite this work$/ });
    // A demo page whose first screen is a paper header has become a product
    // page for its own demo.
    expect(runs.compareDocumentPosition(paper) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});

describe("browsing versus running", () => {
  it("says plainly that the runs are recorded, before any of them is read", () => {
    render(<Home />);

    const notice = screen.getByText(/You are reading recorded runs, not live ones/i);
    expect(notice).toBeInTheDocument();
    expect(
      screen.getByText(/Browsing them needs no API key and no backend/i),
    ).toBeInTheDocument();
  });

  it("offers Paper, Code, Examples and Cite from the top", () => {
    render(<Home />);

    const nav = screen.getByRole("navigation", { name: /primary/i });
    expect(within(nav).getByRole("link", { name: "Paper" })).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2603.14629",
    );
    expect(within(nav).getByRole("link", { name: "Code" })).toHaveAttribute(
      "href",
      "https://github.com/peng1z/ResearchPilot",
    );
    expect(within(nav).getByRole("link", { name: "Examples" })).toHaveAttribute("href", "#recorded-runs");
    expect(within(nav).getByRole("link", { name: "Cite" })).toHaveAttribute("href", "#cite");
  });

  it("states the version boundary between the paper and this build", () => {
    render(<Home />);

    // The paper describes Semantic Scholar and arXiv; these recordings also
    // carry OpenAlex results, which the paper does not report.
    expect(
      screen.getByText(/not part of what the paper reports/i),
    ).toBeInTheDocument();
  });

  it("says the method paper is not a source for the topics it drafts", () => {
    render(<Home />);

    expect(
      screen.getByText(/does not belong in the reference list of a review drafted/i),
    ).toBeInTheDocument();
  });

  it("cites the canonical BibTeX, not a hand-written copy", () => {
    render(<Home />);

    // The canonical entry says cs.IR; a hand-written copy said cs.CL.
    expect(screen.getByText(/primaryClass = \{cs\.IR\}/)).toBeInTheDocument();
    expect(screen.getByText(/doi = \{10\.48550\/arXiv\.2603\.14629\}/)).toBeInTheDocument();
  });
});

describe("case pages", () => {
  it("links every recorded run to its own page", () => {
    render(<Home />);

    for (const run of demoRuns) {
      const link = screen.getByRole("link", { name: run.slug });
      // Directory-style, because the export writes runs/<slug>/index.html and
      // an extensionless path 404s on a plain static host.
      expect(link).toHaveAttribute("href", `/runs/${run.slug}/`);
    }
  });
});

describe("running is opt-in", () => {
  it("keeps the run form behind a disclosure, closed by default", () => {
    render(<Home />);

    // Match the disclosure's own summary, not prose that mentions running.
    const summary = [...document.querySelectorAll("summary")].find((node) =>
      /Run your own question/i.test(node.textContent ?? ""),
    );
    expect(summary).toBeDefined();
    const disclosure = summary!.closest("details");
    expect(disclosure).not.toBeNull();
    // Browsing the recordings is the default; running is a deliberate act that
    // needs a backend the visitor supplies.
    expect(disclosure).not.toHaveAttribute("open");
    expect(screen.getByText(/This deployment hosts no backend and starts nothing/i)).toBeInTheDocument();
  });
});

describe("run metadata", () => {
  it("carries the date, model and software version of each recording", () => {
    for (const run of demoRuns) {
      // These come from the run itself. The fixtures carried none of them
      // until the pipeline started recording what produced a report.
      expect(run.report.created_at).toMatch(/^\d{4}-\d{2}-\d{2}/);
      expect(run.report.tool?.model).toBeTruthy();
      expect(run.report.tool?.version).toBeTruthy();
      expect(run.report.tool?.name).toBe("ResearchPilot");
    }
  });

  it("names the method paper on every run without making it a source", () => {
    for (const run of demoRuns) {
      expect(run.report.tool?.method_paper).toBe("https://arxiv.org/abs/2603.14629");
      expect(run.report.tool?.method_paper_note).toMatch(/not a source for the topic/i);
      // The drafted references are the topic's sources; the method paper is
      // not among them.
      const cited = run.report.references.map((reference) => reference.title.toLowerCase());
      expect(cited.some((title) => title.includes("researchpilot"))).toBe(false);
    }
  });
});
