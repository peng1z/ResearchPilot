import React from "react";
import { render, screen } from "@testing-library/react";

import Home from "../app/page";

describe("Home", () => {
  it("renders the research workflow shell", () => {
    render(<Home />);
    expect(screen.getByText("ResearchPilot")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start research/i })).toBeInTheDocument();
    expect(screen.getByText(/multi-agent research co-pilot/i)).toBeInTheDocument();
  });
});
