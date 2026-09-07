# Design

Written from the built pages, not before them.

## What this surface is

The demo is the paper's runnable artifact. A researcher arrives from
arXiv:2603.14629 or from an assistant that surfaced it, and their job is to
judge whether the system did what the paper claims. So the mode is Read: the
recorded synthesis leads, and everything else gets out of its way.

The arrangement this category ships -- paper title, author line, four buttons,
abstract, then the demo somewhere below -- is refused. That is a product page
wearing an artifact's clothes, and it pushes the only thing worth reading
under the fold.

## Ground and colour

White, not the cream this started as. The reader has a PDF open beside this
page; matching that page is worth more than atmosphere, and cream plus a serif
is the look every generated page in this category arrives wearing.

Restrained: neutrals and one accent, `--accent: #1a4d8f`, the colour a citation
takes in a well-set document. It appears on links and focus rings and nowhere
else. Nothing is tinted to signal a mood.

| token | value | for |
|---|---|---|
| `--ink` | `#161513` | body text, a hair off black so long passages do not vibrate |
| `--ink-2` | `#4d4a45` | secondary prose |
| `--ink-3` | `#7b7770` | labels, metadata, disclosure markers |
| `--paper` | `#ffffff` | ground |
| `--rule` | `#e6e3de` | section divisions |
| `--rule-strong` | `#cbc7c0` | inputs, quote rules, scrollbar thumb |

## Type

Body is a Charter-class serif at 17px on 1.72, because the page is read
continuously rather than scanned. Headings are the system sans: here a heading
is navigation, not voice, and the inversion keeps the page out of the
serif-display cliche without reaching for a novelty face.

Measure is `min(34rem, 100%)`, about 64ch at the body size. `min()` rather than
a flat value so the measure is a reading comfort and never a floor that pushes
text off a narrow screen.

Figures are tabular throughout: paper counts, durations and years line up down
a column.

## Controls

`.label`, `.control`, `.btn`, `.group` and `.entry` in `globals.css` are the
whole control vocabulary: a rule, a piece of type, and nothing enclosing
anything. A record in a list -- a recorded run, a saved report, a reference --
is an `.entry`: a row under a rule, not a bordered tile.

These rules are unlayered, so they beat Tailwind's layered utilities on any
property they both set. `button { color: inherit }` in particular swallows a
`text-*` utility without complaint, which is why selected and muted states are
declared as classes here rather than left in a `className` that silently does
nothing.

## Structure

Rules and space divide; nothing encloses. There are no cards, and the panels
this replaced were the reason two unrelated projects looked identical.

No eyebrow above any heading. The heading carries its own weight, and a
repeated product name above it carried nothing.

Sections separate with a top rule and 44px above the heading, 10px below it, so
a heading joins what follows rather than trailing what came before.

One column, centred, 46rem wide. An earlier arrangement put the agent feed and
the saved-report list in a left column beside the reading, which meant the
default state of the page -- a recorded run, nothing live -- was half a screen
of white beside text pinned to the right. The live-run apparatus now sits
inside the run panel, where a run is actually started, and the reading has the
page.

Sub-headings are set as headings, at 1.02rem in the sans. They were small grey
letterspaced caps, which reads as a form label rather than a turn in an
argument, and which is the badge every generated page in this category wears.

## Motion

One authored moment: a disclosure opening, 220ms on an exponential ease-out
from an already-visible default, behind `prefers-reduced-motion`. Nothing
enters on scroll.

## Browser surfaces

Selection, focus ring, and scrollbar are themed from the palette. Left at their
defaults they belong to no design system, and they are the cheapest signal that
a page was assembled rather than built.

## Accessibility

Every state that colour distinguishes is also stated in words. `not recorded`
is rendered rather than hidden, because a hidden gap reads as complete
information.

## What is deliberately absent

Gradients, glass, blur, shadow, rounded panels, kickers, and any card whose
only job is to hold a heading and a paragraph.

## Disclosure

A recorded run is a drafted section, its reference list, the papers behind it,
and a synthesis over them -- 8,534px printed in full, of which a reader sees
the first screen. The synthesis is what a visitor came to judge, so it is the
only thing open. The draft and the papers are drawers, and the draft's summary
carries its reading time so opening it is a decision and not a dare.

The citation footer collapses the same way: the paper line and the buttons
stay, the version caveat and the BibTeX entry go behind labelled summaries.
Behind a summary is not gone -- the clipboard can be refused, so the entry has
to stay selectable, and it is, one click away.

The page is 3,599px, down from 8,534. The per-run permalink is 2,722px, down
from 6,982.

## Colour

Three kinds of claim, and which kind it is changes how much weight it carries:
consensus, contradiction, open gap. Each gets a rule down its left edge --
green, amber, accent -- and its own label in the same colour. The word is
always there, so the colour is a second reading of the label and never the only
carrier.

The strip above the synthesis is the one raised voice: papers read, papers
cited, sources that answered, and the model. Set as a paper, the page never
said what this run cost before the reading began.

## What this is not

This page and the review tool in the sibling project had converged on one
component set -- same masthead, same stat strip on the same accent wash, same
drawer, same white ground, two blues six hex points apart. One component
invented once and pasted twice is how two unrelated things end up reading as
one template.

This one is a document, so it is furnished like an offprint:

- A **running head**, not a title bar: small caps, the work and its arXiv id, a
  hairline rule. A document does not have chrome.
- **Warm paper**, `#faf8f4`. The tool keeps the white.
- A **sepia accent**, `#6b3f2a` -- the ink of a bound volume rather than a
  terminal's blue.
- **Serif throughout**, headings included. The earlier note here argued sans
  headings on the grounds that a heading is navigation rather than voice; that
  was written before the tool arrived wearing the same white ground and the
  same sans title. A document sets its headings in the face it is read in.
- **Particulars in a ruled band**, small caps and figures between two rules --
  the way a journal states a paper's specifics. No cards, no fill, no chips.
- **Numbered sections**, hanging in the margin where the viewport has room.

Only the form controls stay in the sans: they are the one machine-facing part
of the page, and a serif input reads as a typo.

## Mark

`app/icon.svg`: a confluence. Three sources searched in parallel, joining into
one synthesis, which is the whole pipeline and the one shape the review tool
does not have. Sepia on paper, so the mark belongs to the document rather than
to an icon set.

It is drawn heavy and nearly edge to edge. The tool's chip is filled
near-black and reads as a block at any size; this one is a light chip, so the
mark has to carry itself. The first version used the page's hairline weights,
sat in the middle of a lot of paper, and vanished at 16px, which is the only
size that matters.

`app/apple-icon.png` is the same drawing rendered at 180px.
