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
