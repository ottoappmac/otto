export interface PathKind {
  path: string;
  is_dir: boolean;
  is_file: boolean;
}

/** Split dragged paths using a stat result.

A name with no dot is not a folder. ``Makefile`` is a file. When the
stat call did not return a row, a trailing slash is a folder and
everything else is a file.
*/
export function splitDroppedPaths(
  paths: string[],
  kinds: PathKind[] | null,
): { files: string[]; folders: string[] } {
  const byPath = new Map((kinds ?? []).map((row) => [row.path, row]));
  const files: string[] = [];
  const folders: string[] = [];
  for (const path of paths) {
    const kind = byPath.get(path);
    if (kind?.is_dir) folders.push(path);
    else if (kind?.is_file) files.push(path);
    else if (path.endsWith("/") || path.endsWith("\\")) folders.push(path);
    else files.push(path);
  }
  return { files, folders };
}
