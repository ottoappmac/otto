export interface FuzzyMatch {
  score: number;
  /** Indices of matched characters in the text. */
  indices: number[];
}

/** Subsequence match, case-insensitive. Rewards consecutive runs and matches in the file name. */
export function fuzzyMatch(query: string, text: string): FuzzyMatch | null {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (!q) return { score: 0, indices: [] };
  const base = t.lastIndexOf("/") + 1;
  const indices: number[] = [];
  let from = 0;
  let score = 0;
  let prev = -2;
  for (const ch of q) {
    const at = t.indexOf(ch, from);
    if (at === -1) return null;
    indices.push(at);
    score += at === prev + 1 ? 5 : 1;
    if (at === base || t[at - 1] === "/" || t[at - 1] === "_" || t[at - 1] === "-" || t[at - 1] === ".") score += 3;
    if (at >= base) score += 1;
    prev = at;
    from = at + 1;
  }
  score -= Math.min(text.length, 80) / 40;
  return { score, indices };
}

/** Filter and rank items by fuzzy match against `key(item)`. */
export function fuzzyFilter<T>(query: string, items: T[], key: (item: T) => string): { item: T; match: FuzzyMatch }[] {
  const out: { item: T; match: FuzzyMatch }[] = [];
  for (const item of items) {
    const match = fuzzyMatch(query, key(item));
    if (match) out.push({ item, match });
  }
  return out.sort((a, b) => b.match.score - a.match.score);
}
