/**
 * How many of the retrieved papers the draft actually cites.
 *
 * The strip read `references.length` under the label "Cited", which is the
 * length of the reference list the pipeline attached, not the number of works
 * the prose cites. The three recorded runs cite 9, 8 and 8 of their 10, so the
 * page was overstating its own coverage on the one line a reader checks first.
 *
 * Returns null when the draft carries no [R#] markers at all rather than
 * falling back to the list length, which would put the wrong number back
 * under the right label. A missing figure is honest; a wrong one is not.
 */
export function citedInDraft(markdown: string): number | null {
  // The reference list at the foot of the draft opens every entry with its
  // own [R#], so counting the whole document counts each entry once for
  // existing rather than for being cited. It only shows up where an entry is
  // never cited in the prose -- which is the one case this figure exists to
  // reveal, so the first version read correct on two of three runs and hid
  // the third.
  const heading = markdown.search(/^#{1,6}[ \t]*References[ \t]*$/im);
  const body = heading >= 0 ? markdown.slice(0, heading) : markdown;
  const labels = new Set([...body.matchAll(/\[R(\d+)\]/g)].map((match) => match[1]));
  return labels.size > 0 ? labels.size : null;
}
